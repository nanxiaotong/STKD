import argparse
import os
import json
import numpy as np
import cv2
import pandas as pd
import torch
import torch.optim
import torch.utils.data
import cvtransforms as T
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torchnet as tnt
import math
from torchnet.engine import Engine
import torch.backends.cudnn as cudnn
from utils import conv_params, linear_params, bnparams, bnstats, at_loss, batch_norm, \
    distillation, rocket_distillation, cast, data_parallel, flatten_stats, flatten_params, mutual_distillation

# import tools
from tools import get_training_dataloader, get_test_dataloader

#网络的输入数据维度或类型上变化不大,可以这样设置来提升运行效率
cudnn.benchmark = True

# 定义一个常量
CONST_STEP_FLAG = 0

# argparse是一个Python模块:命令行选项、参数和子命令解析器
# 1.创建解析器
parser = argparse.ArgumentParser(description='Wide Residual Networks')

# Model options
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--student_depth', default=0, type=int) # 额外参数
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='SVHN', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--teacher_id', default='', type=str)

# Training options
parser.add_argument('--batchSize', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--weightDecay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--optim_method', default='SGD', type=str) # 额外参数
parser.add_argument('--randomcrop_pad', default=4, type=float)
parser.add_argument('--temperature', default=4, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='./', type=str, help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int, help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='2', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# wait to remove
parser.add_argument('--sigma_refine_step', default='[120,160,180]', type=str, help='json list with epochs to refine running_sigma') # 额外参数
parser.add_argument('--running_sigma', default=0, type=float) # 额外参数

# turing option
parser.add_argument('--alpha', default=0, type=float, help="weight for knowledge distilling")
parser.add_argument('--beta', default=0, type=float, help="weight for attention transfer")
parser.add_argument('--gamma', default=0, type=float, help="weight for hint loss") # 额外参数
parser.add_argument('--grad_block', default=True, type=bool, help="switch for gradient block") # 额外参数
parser.add_argument('--param_share', default=True, type=bool, help="switch for parameter sharing") # 额外参数
parser.add_argument('--dropout', default=0.0, type=float) # 额外参数

opt = parser.parse_args()
if not os.path.exists("./logs2"):
    os.mkdir("./logs2")

def resnet(depth, width, num_classes, stu_depth=0):
    """Wide ResNet model definition
    :param depth: total number of layers
    :param width: multiplier for number of feature planes in each convolution
    :param num_classes: number of output neurons in the top linear layer
    :return:
      f: function that defines the model
      params: optimizable parameters dict
      stats: batch normalization moments
    """
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6 # n残差块个数，//为向下取整法
    if stu_depth != 0:
        # student parameter limit perform in the number of group
        assert (stu_depth - 4) % 6 == 0, 'student depth should be 6n+4'
        n_s = (stu_depth - 4) // 6
    else:
        n_s = 0

    widths = torch.Tensor([16, 32, 64]).mul(width).int()

    # 初始化,生成残差块个数
    def gen_block_params(ni, no):
        return {'conv0': conv_params(ni, no, 3),
                'conv1': conv_params(no, no, 3),
                'bn0': bnparams(ni),
                'bn1': bnparams(no),
                'bns0': bnparams(ni),
                'bns1': bnparams(no),
                'convdim': conv_params(ni, no, 1) if ni != no else None,
                }

    #初始化,生成残差块组的参数,每组包含count个残差块
    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    #初始化,生成BN层组的参数,每组包含count个BN层
    def gen_group_stats(ni, no, count):
        return {'block%d' % i: {'bn0': bnstats(ni if i == 0 else no), 'bn1': bnstats(no), 'bns0': bnstats(ni if i == 0 else no), 'bns1': bnstats(no)}
                for i in range(count)}
    
    # 定义参数
    if stu_depth != 0 and opt.param_share:
        print("########################### stu_depth choose if ###########################")
        print(stu_depth)
        print(opt.param_share)
        params = {'conv0': conv_params(3, 16, 3),
                  'group0': gen_group_params(16, widths[0], n),
                  'group1': gen_group_params(widths[0], widths[1], n),
                  'group2': gen_group_params(widths[1], widths[2], n),
                  'groups0': gen_group_params(16, widths[0], n_s),
                  'groups1': gen_group_params(widths[0], widths[1], n_s),
                  'groups2': gen_group_params(widths[1], widths[2], n_s),
                  'bn': bnparams(widths[2]),
                  'bns': bnparams(widths[2]),
                  'fc': linear_params(widths[2], num_classes),
                  'fcs': linear_params(widths[2], num_classes),
                  }

        stats = {'group0': gen_group_stats(16, widths[0], n),
                 'group1': gen_group_stats(widths[0], widths[1], n),
                 'group2': gen_group_stats(widths[1], widths[2], n),
                 'groups0': gen_group_stats(16, widths[0], n_s),
                 'groups1': gen_group_stats(widths[0], widths[1], n_s),
                 'groups2': gen_group_stats(widths[1], widths[2], n_s),
                 'bn': bnstats(widths[2]),
                 'bns': bnstats(widths[2]),
                 }
    else:
        print("########################### stu_depth choose else ###########################")
        print(stu_depth)
        print(opt.param_share)
        params = {'conv0': conv_params(3, 16, 3),
                  'group0': gen_group_params(16, widths[0], n),
                  'group1': gen_group_params(widths[0], widths[1], n),
                  'group2': gen_group_params(widths[1], widths[2], n),
                  'bn': bnparams(widths[2]),
                  'bns': bnparams(widths[2]),
                  'fc': linear_params(widths[2], num_classes),
                  'fcs': linear_params(widths[2], num_classes),
                  }

        stats = {'group0': gen_group_stats(16, widths[0], n),
                 'group1': gen_group_stats(widths[0], widths[1], n),
                 'group2': gen_group_stats(widths[1], widths[2], n),
                 'bn': bnstats(widths[2]),
                 'bns': bnstats(widths[2]),
                 }

    #初始化整个网络框架的参数，并更新参数
    flat_params = flatten_params(params)
    flat_stats = flatten_stats(stats)

    # 构建一个残差块 RELU->卷积->RELU->卷积
    def block(x, params, stats, base, mode, stride, flag, drop_switch=True):
        if flag == 's':
            o1 = F.relu(batch_norm(x, params, stats, base + '.bns0', mode))
            y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
            o2 = F.relu(batch_norm(y, params, stats, base + '.bns1', mode))
            z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
            # 在shortcut通路加1x1卷积
            if base + '.convdim' in params:
                return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
            else:
                return z + x
        o1 = F.relu(batch_norm(x, params, stats, base + '.bn0', mode))
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(batch_norm(y, params, stats, base + '.bn1', mode))
        if opt.dropout > 0 and drop_switch:
            o2 = F.dropout(o2, p=opt.dropout, training=mode)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    # 构建一个残差组
    def group(o, params, stats, base, mode, stride):
        for i in range(n):
            o = block(o, params, stats, '%s.block%d' % (base, i),
                      mode, stride if i == 0 else 1, 't', False)
        return o

    def group_student(o, params, stats, base, mode, stride, n_layer):
        for i in range(n_layer):
            o = block(o, params, stats, '%s.block%d' % (base, i), 
                      mode, stride if i == 0 else 1, 's', False)
        return o

    # 构建整个网络：卷积-第一组-第二组-第三组-relu-池化-全连接
    def f(input, params, stats, mode, prefix=''):
        x = F.conv2d(input, params[prefix + 'conv0'], padding=1)
        g0 = group(x, params, stats, prefix + 'group0', mode, 1)
        g1 = group(g0, params, stats, prefix + 'group1', mode, 2)
        g2 = group(g1, params, stats, prefix + 'group2', mode, 2)
        o = F.relu(batch_norm(g2, params, stats, prefix + 'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params[prefix + 'fc.weight'], params[prefix + 'fc.bias'])

        if stu_depth != 0:
            if opt.param_share:
                # with parameter sharing
                gs0 = group_student(x, params, stats, prefix + 'group0', mode, 1, n_s)
                gs1 = group_student(gs0, params, stats, prefix + 'group1', mode, 2, n_s)
                gs2 = group_student(gs1, params, stats, prefix + 'group2', mode, 2, n_s)
            else:
                gs0 = group_student(x, params, stats, prefix + 'groups0', mode, 1, n_s)
                gs1 = group_student(gs0, params, stats, prefix + 'groups1', mode, 2, n_s)
                gs2 = group_student(gs1, params, stats, prefix + 'groups2', mode, 2, n_s)

            os = F.relu(batch_norm(gs2, params, stats, prefix + 'bns', mode))
            os = F.avg_pool2d(os, 8, 1, 0)
            os = os.view(os.size(0), -1)
            os = F.linear(os, params[prefix + 'fcs.weight'], params[prefix + 'fcs.bias'])
            return os, o, [g0, g1, g2, gs0, gs1, gs2]
        else:
            return o, [g0, g1, g2]
    return f, flat_params, flat_stats

def main():
    global CONST_STEP_FLAG
    CONST_STEP_FLAG = 0
    # opt = parser.parse_args()
    # option note
    # assert not(opt.beta and opt.gamma), "Can't support attention-transfer and rocket-launching together"
    print("##############################################################################")
    print('parsed options:', vars(opt)) 
    epoch_step = json.loads(opt.epoch_step)
    sigma_refine_step = json.loads(opt.sigma_refine_step)
    num_classes = 10 if opt.dataset == 'SVHN' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    
    #加载SVHN数据集
    SVHN_TRAIN_MEAN = (0.485, 0.456, 0.406) # 均值
    SVHN_TRAIN_STD = (0.229, 0.224, 0.225) # 方差

    # total training epoches
    svhn_training_loader = get_training_dataloader(
        SVHN_TRAIN_MEAN,
        SVHN_TRAIN_STD,
    )

    svhn_test_loader = get_test_dataloader(
        SVHN_TRAIN_MEAN,
        SVHN_TRAIN_STD,
    )
    train_loader = svhn_training_loader
    test_loader = svhn_test_loader

    # deal with student first
    f_s, params_s, stats_s = resnet(opt.depth, opt.width, num_classes, opt.student_depth)
    print("######################## deal with student finished ##########################")
    print(opt.student_depth)
    print(f_s)

    # 解包操作
    # deal with teacher
    if opt.teacher_id != '':
        print("######################## deal with teacher start ##########################")
        print(opt.teacher_id)

        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff: # 从文件中读取参数，构建教师网络
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:])
        f_t = resnet(info['depth'], info['width'], num_classes)[0] # pre-trained teacher
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7')) # 从文件中加载model.pt7模型，预训练好的模型，即expert teacher
        params_t = model_data['params']
        stats_t = model_data['stats']

        # merge teacher and student params and stats
        params = {'student.' + k: v for k, v in params_s.items()}
        for k, v in params_t.items():
            params['teacher.' + k] = Variable(v)
        stats = {'student.' + k: v for k, v in stats_s.items()}
        stats.update({'teacher.' + k: v for k, v in stats_t.items()})
        print("######################## merge teacher and student params and stats over ##########################")
        
        def f(inputs, params, stats, mode):        
            if opt.gamma: # 以此判断是否使用hint中间层信息，但最新改进是使用ML进行替换，即对g_s和g_t进行计算。
                y_s, y_t_auto, g_s = f_s(inputs, params, stats, mode, 'student.')
                y_t, g_t = f_t(inputs, params, stats, False, 'teacher.')
                return y_s, y_t_auto, y_t, [mutual_distillation(x, y) for x, y in zip(g_s[0:3], g_s[3:6])]
            else:
                y_s, g_s = f_s(inputs, params, stats, mode, 'student.')
                y_t, g_t = f_t(inputs, params, stats, False, 'teacher.')
                return y_s, y_t, [at_loss(x, y) for x, y in zip(g_s, g_t)]
    else:
        f, params, stats = f_s, params_s, stats_s
    # print(params)

    optimizable = [v for v in params.values() if v.requires_grad]
    
    # 定义优化器 optimizer
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        if opt.optim_method == 'SGD':
            return torch.optim.SGD(optimizable, lr, 0.9, weight_decay=opt.weightDecay)
        elif opt.optim_method == 'Adam':
            return torch.optim.Adam(optimizable, lr)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors, stats = state_dict['params'], state_dict['stats']
        for k, v in params.item()():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    print(pd.DataFrame([(key, v.size(), torch.typename(v.data)) for key, v in params.items()]))
    print('\nAdditional buffers:')
    print(pd.DataFrame([(key, v.size(), torch.typename(v)) for key, v in stats.items()]))

    n_parameters = sum(p.numel() for p in params_s.values())
    print('\nTotal number of parameters:', n_parameters)

    if opt.gamma:
        meter_loss_s = tnt.meter.AverageValueMeter()
        meter_loss_t = tnt.meter.AverageValueMeter()
        meter_loss_c = tnt.meter.AverageValueMeter()
        meter_loss_d = tnt.meter.AverageValueMeter()
        classacc_s = tnt.meter.ClassErrorMeter(accuracy=True)
        classacc_t = tnt.meter.ClassErrorMeter(accuracy=True)
        classacc_s_top1 = tnt.meter.ClassErrorMeter(topk=[1]) # 额外的参数，返回top-1的accuracy    
    else:
        classacc = tnt.meter.ClassErrorMeter(accuracy=True)

    meter_loss = tnt.meter.AverageValueMeter()
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = Variable(cast(sample[0], opt.dtype))
        targets = Variable(cast(sample[1], 'long'))
        if opt.teacher_id != '':
            if opt.gamma:                
                ys, y_t_auto, y_t, loss_groups = data_parallel(f, inputs, params, stats, sample[2], np.arange(opt.ngpu))[:4] # loss_groups 代表注意力转移损失
                loss_groups = [v.sum() for v in loss_groups]
                # [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
                loss_l2 = torch.nn.MSELoss()
                T = 4
                loss_student = F.cross_entropy(ys, targets)
                loss_teacher = F.cross_entropy(y_t_auto, targets)
                loss_course = F.kl_div(F.log_softmax(ys, dim = 1), F.softmax(Variable(y_t_auto), dim=1)) + F.kl_div(F.log_softmax(y_t_auto, dim = 1), F.softmax(Variable(ys), dim=1)) 
                y_tech_temp = torch.autograd.Variable(y_t_auto.data, requires_grad=False)
                log_kd = rocket_distillation(ys, y_t, targets, opt.temperature, opt.alpha)
                # + mutual_distillation(ys, y_tech_temp) DML对logits进行双向转移
                # return rocket_distillation(ys, y_t, targets, opt.temperature, opt.alpha) + F.cross_entropy(y_t_auto, targets) + F.cross_entropy(ys, targets) + mutual_distillation(ys, y_tech_temp), (ys, y_t_auto, loss_student, loss_teacher, loss_course, log_kd)
                # return opt.beta * sum(at_losses_groups)+ F.cross_entropy(y_t_auto, targets) + F.cross_entropy(ys, targets) + opt.gamma * sum(loss_groups), (ys, y_t_auto, loss_student, loss_teacher, loss_course, log_kd)
                # return rocket_distillation(ys, y_t, targets, opt.temperature, opt.alpha) + F.cross_entropy(y_t_auto, targets) + F.cross_entropy(ys, targets) + opt.gamma * sum(loss_groups), (ys, y_t_auto, loss_student, loss_teacher, loss_course, log_kd)
                # return rocket_distillation(ys, y_t, targets, opt.temperature, opt.alpha) + F.cross_entropy(y_t_auto, targets) + F.cross_entropy(ys, targets) +  opt.gamma * sum(loss_groups) + opt.gamma * ((y_tech_temp - ys) * (
                #         y_tech_temp - ys)).sum() / opt.batchSize, (ys, y_t_auto, loss_student, loss_teacher, loss_course, log_kd)
                # return rocket_distillation(ys, y_t, targets, opt.temperature, opt.alpha) + F.cross_entropy(y_t_auto, targets) + F.cross_entropy(ys, targets) + opt.gamma * (sum(loss_groups) + loss_course), (ys, y_t_auto, loss_student, loss_teacher, loss_course, log_kd)
                # return opt.beta * sum(loss_groups) + F.cross_entropy(y_t_auto, targets) + F.cross_entropy(ys, targets) + opt.gamma * ((y_tech_temp - ys) * (y_tech_temp - ys)).sum() / opt.batchSize, (ys, y_t_auto, loss_student, loss_teacher, loss_course, log_kd)
                return rocket_distillation(ys, y_t, targets, opt.temperature, opt.alpha) + opt.beta * sum(loss_groups) + F.cross_entropy(y_t_auto, targets) + F.cross_entropy(ys, targets) + opt.gamma * ((y_tech_temp - ys) * (y_tech_temp - ys)).sum() / opt.batchSize, (ys, y_t_auto, loss_student, loss_teacher, loss_course, log_kd)
            else:
                y_s, y_t, loss_groups = data_parallel(f, inputs, params, stats, sample[2], np.arange(opt.ngpu))
                loss_groups = [v.sum() for v in loss_groups]
                [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
                return distillation(y_s, y_t, targets, opt.temperature, opt.alpha) + opt.beta * sum(loss_groups), y_s # 没考虑hint loss情况, alpha!=0, KD Losss; beta !=0, at Losss.
        else:
            if opt.gamma:
                # print("-----------------Mutual Learning-----------------")
                ys, y = data_parallel(f, inputs, params, stats, sample[2], np.arange(opt.ngpu))[:2]
                loss_l2 = torch.nn.MSELoss()
                T = 4
                network_num = 2 
                loss_student = F.cross_entropy(ys, targets) #学生网络 交叉熵损失
                loss_teacher = F.cross_entropy(y, targets) #教师网络 交叉熵损失
                # loss_course = opt.gamma * ((y - ys) * (y - ys)).sum() / opt.batchSize  # hint loss
                loss_course = F.kl_div(F.log_softmax(ys, dim = 1), F.softmax(Variable(y), dim=1)) + F.kl_div(F.log_softmax(y, dim = 1), F.softmax(Variable(ys), dim=1)) 
                if opt.grad_block:
                    y_course = torch.autograd.Variable(y.data, requires_grad=False)
                else:
                    y_course = y
                # rocket launching loss 
                return F.cross_entropy(y, targets) + F.cross_entropy(ys, targets) + F.kl_div(F.log_softmax(ys, dim = 1), F.softmax(Variable(y_course), dim=1)) + F.kl_div(F.log_softmax(y_course, dim = 1), F.softmax(Variable(ys), dim=1)), (ys, y, loss_student, loss_teacher, loss_course)  
                # rocket launching loss 
                # return F.cross_entropy(y, targets) + F.cross_entropy(ys, targets) + opt.gamma * ((y_course - ys) * (y_course - ys)).sum() / opt.batchSize, (ys, y, loss_student, loss_teacher, loss_course)  
            else:
                # print("-----------------------------------------------------------------------")
                # print("WRN_"+ str(opt.depth) +"_" + str(opt.width))
                y = data_parallel(f, inputs, params, stats, sample[2], np.arange(opt.ngpu))[0] # 并行计算输出结果，[0]为输出的第一项。
                return F.cross_entropy(y, targets), y # loss 为 teacher，student 单一网络的训练损失，即为交叉熵损失。

    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in params.items()},
                        stats=stats,
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   open(os.path.join(opt.save, 'model.pt7'), 'wb'))
        z = vars(opt).copy()
        z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    if opt.gamma:
        def on_forward(state):
            classacc_s.add(state['output'][0].data, torch.LongTensor(state['sample'][1]))
            classacc_t.add(state['output'][1].data, torch.LongTensor(state['sample'][1]))
            meter_loss.add(state['loss'].item())
            meter_loss_s.add(state['output'][2].item())
            meter_loss_t.add(state['output'][3].item())
            meter_loss_c.add(state['output'][4].item()) 
            classacc_s_top1.add(state['output'][0].data, torch.LongTensor(state['sample'][1])) # 额外参数
        
        def on_start_epoch(state):
            classacc_s.reset()
            classacc_t.reset()
            classacc_s_top1.reset() # 额外参数
            meter_loss.reset()
            meter_loss_s.reset()
            meter_loss_t.reset()
            meter_loss_c.reset()
            timer_train.reset()
            [meter.reset() for meter in meters_at]
            # state['iterator'] = tqdm(train_loader)

            epoch = state['epoch'] + 1
            if epoch in sigma_refine_step:
                opt.running_sigma += opt.gamma
            if epoch in epoch_step:
                lr = state['optimizer'].param_groups[0]['lr']
                state['optimizer'] = create_optimizer(
                    opt, lr * opt.lr_decay_ratio)
        
        def on_end_epoch(state):
            train_loss = meter_loss.value()
            train_loss_s = meter_loss_s.value()
            train_loss_t = meter_loss_t.value()
            train_loss_c = meter_loss_c.value()
            train_acc_s = classacc_s.value()
            train_acc_t = classacc_t.value()
            train_time = timer_train.value()
            meter_loss.reset()
            meter_loss_s.reset()
            meter_loss_t.reset()
            meter_loss_c.reset()
            classacc_s.reset()
            classacc_t.reset()
            classacc_s_top1.reset() # 额外参数
            timer_test.reset()

            engine.test(h, test_loader)

            test_acc_s = classacc_s.value()[0]
            test_acc_t = classacc_t.value()[0]
            test_acc_s_top1 = classacc_s_top1.value()[0] # 额外参数
            print(log({
                "train_loss": train_loss[0],
                "train_acc_student": train_acc_s[0],
                "train_acc_teacher": train_acc_t[0],
                "train_loss_student": train_loss_s[0],
                "train_loss_teacher": train_loss_t[0],
                "test_loss": meter_loss.value()[0],
                "test_loss_student": meter_loss_s.value()[0],
                "test_loss_teacher": meter_loss_t.value()[0],
                "test_loss_course": meter_loss_c.value()[0],
                "test_acc_student": test_acc_s,
                "test_acc_teacher": test_acc_t,
                "test_acc_student_top1": test_acc_s_top1,
                "epoch": state['epoch'],
                "num_classes": num_classes,
                "n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
                "at_losses": [m.value() for m in meters_at],
            }, state))
            print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                (opt.save, state['epoch'], opt.epochs, test_acc_s))
    else:
        def on_forward(state):
            classacc.add(state['output'].data,
                         torch.LongTensor(state['sample'][1]))
            meter_loss.add(state['loss'].item())
        
        def on_start_epoch(state):
            classacc.reset()
            meter_loss.reset()
            timer_train.reset()
            [meter.reset() for meter in meters_at]
            # state['iterator'] = tqdm(train_loader)

            epoch = state['epoch'] + 1
            if epoch in epoch_step:
                lr = state['optimizer'].param_groups[0]['lr']
                state['optimizer'] = create_optimizer(
                    opt, lr * opt.lr_decay_ratio)

        def on_end_epoch(state):
            train_loss = meter_loss.value()
            train_acc = classacc.value()
            train_time = timer_train.value()
            meter_loss.reset()
            classacc.reset()
            timer_test.reset()

            engine.test(h, test_loader)

            test_acc = classacc.value()[0]
            print(log({
                "train_loss": train_loss[0],
                "train_acc": train_acc[0],
                "test_loss": meter_loss.value()[0],
                "test_acc": test_acc,
                "epoch": state['epoch'],
                "num_classes": num_classes,
                "n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
                "at_losses": [m.value() for m in meters_at],
            }, state))
            print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                (opt.save, state['epoch'], opt.epochs, test_acc))

    def on_start(state):
        state['epoch'] = epoch

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)

if __name__ == '__main__':
    main()
