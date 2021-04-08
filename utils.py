import torch
import torchvision
from functools import partial
import torch.nn.functional as F
import torch.cuda.comm as comm
from torch.autograd import Variable
from nested_dict import nested_dict
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.nn.init import kaiming_normal
import torchvision.transforms as transforms
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather

def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * (T * T * 2. * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

def rocket_distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * (T * T * 2. * alpha)

def mutual_distillation(ys, yt):
    # print(ys.shape)
    # print(yt.shape)
    return F.kl_div(F.log_softmax(ys, dim = 1), F.softmax(Variable(yt), dim=1)) + F.kl_div(F.log_softmax(yt, dim = 1), F.softmax(Variable(ys), dim=1))

def normalize(input, p=2, dim=1, eps=1e-12):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation
        dim (int): the dimension to reduce
        eps (float): small value to avoid division by zero
    """
    return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input) 

def at(x):
    return normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

def cd_loss(x, y):
    """Channel Distillation Loss"""
    loss = 0.
    for s, t in zip(x, y):
        s = s.mean(dim=(2, 3), keepdim=False)
        t = t.mean(dim=(2, 3), keepdim=False)
        loss += torch.mean(torch.pow(s - t, 2))
    return loss

def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda(), dtype)()

# 卷积层参数初始化
def conv_params(ni, no, k=1):
    return cast(kaiming_normal(torch.Tensor(no, ni, k, k)))

# 全连接层参数初始化
def linear_params(ni, no):
    return cast({'weight': kaiming_normal(torch.Tensor(no, ni)), 'bias': torch.zeros(no)})

def bnparams(n):
    return cast({'weight': torch.rand(n), 'bias': torch.zeros(n)})

def bnstats(n):
    return cast({'running_mean': torch.zeros(n), 'running_var': torch.ones(n)})

def data_parallel2(f, input, params, mode, device_ids, output_device=None):
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

def data_parallel(f, input, params, stats, mode, device_ids, output_device=None):
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, stats, mode)

    def replicate(param_dict, g):
        replicas = [{} for d in device_ids]
        for k, v in param_dict.items():
            for i, u in enumerate(g(v)):
                replicas[i][k] = u
        return replicas

    params_replicas = replicate(params, lambda x: Broadcast(device_ids)(x))
    stats_replicas = replicate(stats, lambda x: comm.broadcast(x, device_ids))

    replicas = [partial(f, params=p, stats=s, mode=mode)
                for p, s in zip(params_replicas, stats_replicas)]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)

def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}

def flatten_params(params):
    return OrderedDict(('.'.join(k), Variable(v, requires_grad=True))
                       for k, v in nested_dict(params).iteritems_flat() if v is not None)

def flatten_stats(stats):
    return OrderedDict(('.'.join(k), v)
                       for k, v in nested_dict(stats).iteritems_flat())

def batch_norm(x, params, stats, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=stats[base + '.running_mean'],
                        running_var=stats[base + '.running_var'],
                        training=mode)

def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)

def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True

def get_training_dataloader(mean, std, batch_size=128, num_workers=4, shuffle=True):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        #transforms.Pad(4),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    svhn_training = torchvision.datasets.SVHN(root='svhn',split='train', download=False, transform=transform_train)
    svhn_training_loader = DataLoader(svhn_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return svhn_training_loader

def get_test_dataloader(mean, std, batch_size=128, num_workers=4, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    svhn_test = torchvision.datasets.SVHN(root='svhn',split="test",download=False,transform=transform_test)
    svhn_test_loader = DataLoader(svhn_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return svhn_test_loader
