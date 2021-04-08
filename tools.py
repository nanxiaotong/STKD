import torch
from functools import partial
import torch.nn.functional as F
from nested_dict import nested_dict
from torch.nn.init import kaiming_normal_
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather

# 蒸馏损失
def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1) # 学生网络的输出结果概率化，并进行log
    q = F.softmax(teacher_scores/T, dim=1) # 教师网络的输出结果概率化
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0] # KL散度损失
    l_ce = F.cross_entropy(y, labels) # 学生网络交叉熵损失
    return l_kl * alpha + l_ce * (1. - alpha) # 如果alpha为0，则整个蒸馏损失是学生交叉熵损失，即无蒸馏操作。

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    print(x.shape)
    print(y.shape)
    return (at(x) - at(y)).pow(2).mean()

def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()

def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))

def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}

def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}

def data_parallel(f, input, params, mode, device_ids, output_device=None):
    device_ids = list(device_ids)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)

def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}

def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)

def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)

def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True


import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_training_dataloader(mean, std, batch_size=128, num_workers=4, shuffle=True):
    # 进行数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        #transforms.Pad(4),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    svhn_training = torchvision.datasets.SVHN(root='svhn',split='train', download=False, transform=transform_train)
    svhn_training_loader = DataLoader(
        svhn_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return svhn_training_loader

def get_test_dataloader(mean, std, batch_size=128, num_workers=4, shuffle=True):
    # 进行数据增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    svhn_test = torchvision.datasets.SVHN(root='svhn',split="test",download=False,transform=transform_test)
    svhn_test_loader = DataLoader(
        svhn_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return svhn_test_loader
