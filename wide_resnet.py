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