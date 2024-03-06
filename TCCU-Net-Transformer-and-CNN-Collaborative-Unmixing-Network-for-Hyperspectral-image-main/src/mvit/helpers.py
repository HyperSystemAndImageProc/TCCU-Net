from torch import nn


def conv_1x1_bn(c_in, c_out, activation=True):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU() if activation else nn.Identity(),
    )


def conv_nxn_bn(c_in, c_out, kernel_size, *, stride=1, padding=None, groups=1):
    if padding is None:
        padding = kernel_size // 2

    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size, stride, padding,
                  groups=groups, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(),
    )
