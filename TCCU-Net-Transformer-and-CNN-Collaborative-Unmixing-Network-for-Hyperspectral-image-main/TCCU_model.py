import os
import pickle
import time

import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.tensorboard import writer
from torchsummary import summary

import datasets
import plots
import transformer
import utils


import matplotlib.pyplot as plt
#%matplotlib inline
# from numpy import linalg as LA
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
#from models import *
#import math
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as FN

from functools import partial
from collections import OrderedDict

from src.mvit import CrossViT




def crossvit(image_size, num_classes, kind):
    kind = kind.lower()
    if kind.startswith('xxs'):
        channels = [16, 16, 24, 24, 48, 64, 80, 320]
        dims = [64, 80, 96]
        expansion = 2
    elif kind.startswith('xs'):
        channels = [16, 32, 48, 48, 64, 80, 96, 384]
        dims = [96, 120, 144]
        expansion = 4
    elif kind.startswith('s'):
        channels = [16, 32, 64, 64, 96, 128, 160, 640]
        dims = [144, 192, 240]
        expansion = 4
    else:
        raise ValueError("`kind` must be in ('xxs', 'xs', 's')")

    return CrossViT(
        image_size=image_size,
        num_classes=num_classes,
        chs=channels,
        dims=dims,
        depths=[2, 4, 3],
        expansion=expansion,
        # kernel_size=3,
        # patch_size=(2, 2),
    )

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(FN.softplus(x)))
        return x



"""
class CFPE_module(nn.Module):

    def __init__(self, neighbor_band, num_patches,):   # 7 15
        super().__init__()
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=num_patches, out_channels=num_patches, kernel_size=1, padding=0),
            nn.BatchNorm1d(num_patches),
            Mish()
        )

        # 光谱、空间卷积

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(neighbor_band, 5, 5),
                      padding=(int((neighbor_band - 1) / 2), 2, 2), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish()

        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(neighbor_band, 3, 3),
                      padding=(int((neighbor_band - 1) / 2), 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish()
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish()
        )

        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(1, 1), stride=1, padding=0,
                      groups=num_patches),
            Mish(),
            nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(5, 5), stride=1, padding=2,
                      groups=num_patches),
            Mish()
        )

        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(1, 1), stride=1, padding=0,
                      groups=num_patches),
            Mish(),
            nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(3, 3), stride=1, padding=1,
                      groups=num_patches),
            Mish()
        )

    def forward(self, x,):
        x_3d = torch.unsqueeze(x, 1)
        x_3d_1 = self.conv3d_1(x_3d)
        x_3d_2 = self.conv3d_2(x_3d)
        x_3d = x_3d_1 + x_3d_2
        x_3d = self.conv3d_3(x_3d)
        x_3d = torch.squeeze(x_3d, 1)

        # 空间通道
        x_spa_1 = self.depth_conv1(x)
        x_spa_2 = self.depth_conv2(x)
        x_spa = x_spa_1 + x_spa_2


        # x= (x_spa + x_3d) * 0.7 + x
        x = x_spa +x_3d
        return x
"""

num_patches = 256
num_patches2 = 512
neighbor_band = 9

class CA_Block(nn.Module):
    def __init__(self, L, h, w, reduction=4):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=L, out_channels=L // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(L // reduction)

        self.F_h = nn.Conv2d(in_channels=L // reduction, out_channels=L, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=L // reduction, out_channels=L, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = s_h.expand_as(x) * s_w.expand_as(x)

        return out


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, hws,channels=64, r=4):
        super(AFF, self).__init__()


        # self.local_att = nn.Sequential(
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )
        #
        # self.global_att = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )
        self.hws = hws
        self.ca = CA_Block(channels, h=hws, w=hws)


        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        # xl = self.local_att(xa)
        # xg = self.global_att(xa)
        # xlg = xl + xg
        # wei = self.sigmoid(xlg)
        wei = self.ca(xa)
        x1 = 2 * x * wei
        x2 = 2 * residual * (1 - wei)
        xi = x1+x2
        wei2 = self.ca(xi)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

# class PSAModule(nn.Module):
#
#     def __init__(self, inplans, planes, hws, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 4, 16]):
#         super(PSAModule, self).__init__()
#         self.conv_1 = conv(inplans, planes, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
#                             stride=stride, groups=conv_groups[0])
#         self.conv_2 = conv(inplans, planes, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
#                             stride=stride, groups=conv_groups[1])
#         self.conv_3 = conv(inplans, planes, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
#                             stride=stride, groups=conv_groups[2])
#         self.conv_4 = conv(inplans, planes, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
#                             stride=stride, groups=conv_groups[3])
#         self.hws = hws
#         self.fusion = AFF(hws, planes)
#
#     def forward(self, x):
#         x1 = self.conv_1(x)
#         x2 = self.conv_2(x)
#         x3 = self.conv_3(x)
#         x4 = self.conv_4(x)
#         out = self.fusion(x1, x2)
#         out = self.fusion(out, x3)
#         out = self.fusion(out, x4)
#
#
#         return out



class AutoEncoder(nn.Module):
    def __init__(self,  A, P, L, size, patch, dim, inplans, planes, hws, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 4, 16]):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim
        # midplanes = planes
        # norm_layer = nn.BatchNorm2d
        # self.conv1 = nn.Conv2d(inplans, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        # self.bn1 = norm_layer(midplanes)
        # self.conv2 = nn.Conv2d(inplans, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        # self.bn2 = norm_layer(midplanes)
        # self.conv3 = nn.Conv2d(midplanes, planes, kernel_size=1, bias=True)
        # self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        # self.relu = nn.ReLU(inplace=False)

        self.conv_1 = conv(inplans, planes, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])


        # self.ca = CA_Block(planes // 4, h=hws, w=hws)

        # self.se = SEWeightModule(planes // 4, h=hws, w=hws)
        # self.split_channel = planes // 4
        # self.softmax = nn.Softmax(dim=1)
        # 11.30
        # self.hws = hws
        # self.fusion = AFF(hws, planes)

        # cross_vit加深高级特征，便于全局处理
        self.encoder = nn.Sequential(
            nn.Conv2d(L,  128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),  # 256->L
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim*P)//patch**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((dim*P)//patch**2, momentum=0.5),
        )

        #self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=(dim * P), depth=2,
        #                              heads=8, mlp_dim=12, pool='cls')
        # cross_vit处理
        self.model = CrossViT(
            image_size=[(dim*P)//patch**2, size, size], # (dim*P)//patch**2
            # samson:95*95*156 # apex:110*110*285 #24/32 samson/apex
            num_classes=A,
            #num_classes=200,
            # samson:150 apex:200 sim:150
            chs=[24, 32, 64, 64, 96, 128, 160, 640],
            # chs=[156,160,192,224,256,288,320,640],
            #chs=[16, 24, 32, 48, 64, 96, 128, 256],
            # 8，16，24，48，64，128，156，320
            dims=[144, 192, 240],  # 144,192,240
            depths=[2, 4, 3],  # [1,1,1] [2,4,3]
            expansion=2,
            # kernel_size=3,
            # patch_size=(2, 2),
            # depths=[2,4,3]
        )

        # cross_vit结果线性化
        self.upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),
        )

        # cross_vit上采样部分
        self.upsample = nn.Sequential(
            nn.Conv2d(P, planes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(planes, momentum=0.1),
            nn.LeakyReLU(),
        )


        # 下采样环节
        self.Decoder = nn.Sequential(
            nn.Conv2d(planes, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),  # 256->L
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim * P) // patch ** 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((dim * P) // patch ** 2, momentum=0.5),
            nn.LeakyReLU(),
            nn.Conv2d((dim * P) // patch ** 2, P, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(P, momentum=0.9),
        )
        # softmax->丰度
        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )
        # 解码部分->端元
        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

        # 加深低级特征
        self.new1 = nn.Sequential(
            nn.Conv2d(L, planes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(planes, momentum=0.1),
            nn.LeakyReLU(),

        )
        # 双支路下采样模块
        # self.downsample= nn.Sequential(
        #     nn.Conv2d(planes, 156, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d(156, momentum=0.1),
        #     nn.LeakyReLU(),
        # )
        # self.downsample2 = nn.Sequential(
        #     nn.Conv2d(planes, 156, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d(156, momentum=0.1),
        #     nn.LeakyReLU(),
        # )

        # spatrial
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(neighbor_band, 5, 5),
                      padding=(int((neighbor_band - 1) / 2), 2, 2), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish()

        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(neighbor_band, 3, 3),
                      padding=(int((neighbor_band - 1) / 2), 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish()
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish()
        )

        # MISIC-NET
        # self.conv1 = nn.Sequential(
        #     conv(L, 256, 3, 1),
        #     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(0.1, inplace=True),
        # )
        # self.conv2 = nn.Sequential(
        #     conv(256, 256, 3, 1),
        #     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(0.1, inplace=True),
        # )
        # self.conv3 = nn.Sequential(
        #     conv(L, 4, 3, 1),
        #     nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(0.1, inplace=True),
        # )
        # self.dconv2 = nn.Sequential(
        #     nn.Upsample(scale_factor=1),
        #     conv(260, 256, 3, 1),
        #     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(0.1, inplace=True),
        # )
        #
        # self.dconv3 = nn.Sequential(
        #     nn.Upsample(scale_factor=1),
        #     conv(256, P, 3, 1),
        #     nn.BatchNorm2d(P, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Softmax(),
        # )
        # self.dconv4 = nn.Sequential(
        #     nn.Linear(P, L, bias=False),
        # )

        # # space
        # self.depth_conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(1, 1), stride=1, padding=0,
        #               groups=num_patches),
        #     Mish(),
        #     nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(5, 5), stride=1, padding=2,
        #               groups=num_patches),
        #     Mish()
        # )
        # self.depth_conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(1, 1), stride=1, padding=0,
        #               groups=num_patches),
        #     Mish(),
        #     nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(3, 3), stride=1, padding=1,
        #               groups=num_patches),
        #     Mish()
        # )

        # # CONV
        # self.conv = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),  # 256->L
        #     nn.BatchNorm2d(512, momentum=0.9),
        #     nn.Dropout(0.25),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(512, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d(384, momentum=0.9),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d(256, momentum=0.1),
        #
        # )













    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)






    def forward(self, x):
        # x = torch.squeeze(x,4)
        # _, _, h, w = x.size()
        # x1 = self.pool1(x)
        # x1 = self.conv1(x1)
        # x1 = self.bn1(x1)
        # x1 = x1.expand(-1, -1, h, w)
        # # x1 = F.interpolate(x1, (h, w))
        #
        # x2 = self.pool2(x)
        # x2 = self.conv2(x2)
        # x2 = self.bn2(x2)
        # x2 = x2.expand(-1, -1, h, w)
        # # x2 = F.interpolate(x2, (h, w))
        #
        # x = self.relu(x1 + x2)
        # x = self.conv3(x).sigmoid()

        #第一轮
        x3 = self.new1(x)

        #光谱通道spectral
        x_3d = torch.unsqueeze(x3, 1)#x3->x
        x_3d_1 = self.conv3d_1(x_3d)
        x_3d_2 = self.conv3d_2(x_3d)
        x_3d = x_3d_1 + x_3d_2
        x_3d = self.conv3d_3(x_3d)
        x_3d = torch.squeeze(x_3d, 1)
        # x_3d = self.downsample2(x_3d)

        # 空间通道space
        # x_spa_1 = self.depth_conv1(x3)#x3->x
        # x_spa_2 = self.depth_conv2(x3)#x3->x
        # x_spa = x_spa_1 + x_spa_2

        x_spa1 = self.conv_1(x3)
        x_spa2 = self.conv_2(x3)
        x_spa3 = self.conv_3(x3)
        x_spa4 = self.conv_4(x3)
        out = torch.add(x_spa1, x_spa2) # self.fusion
        out = torch.add(out, x_spa3)
        out = torch.add(out, x_spa4)
        # out = self.downsample(out)

        # CONV
        # Encode_conv = self.conv(x3)
        # Encode_conv = self.downsample(Encode_conv)


        abuest2= self.encoder(x) #squeeze
       # abuest = self.encoder(abuest2)  # encoder cls_emb = self.model(x)
        # cross VIT
        cls_emb = self.model(abuest2)  # vtrans  abu_est
        cls_emb = cls_emb.view(1, self.P, -1)  # view 改变 tensor 形状 #当view(-1)时，其作用是将一个矩阵展开为一个向量
        abuest = self.upscale(cls_emb).view(1, self.P, self.size, self.size)
        abuest = self.upsample(abuest)
        # abuest = abuest + abuest2
        # abuest = abuest + x
        # abuest = self.downsample2(abuest)


        # x= (x_spa + x_3d) * 0.7 + x
        # x_f1 = self.fusion(x_3d, out)
        # x_f2 = self.fusion(x_f1,  Encode_conv)
        # x = self.fusion(x_f1, abuest)
        # x = x_3d + out +  abuest
        #x = x_3d + x_spa + Encode_conv + abuest
        #x_m = torch.add(abuest, x)
        #x = torch.cat((x_3d, out, abuest), dim=1)
        xa = torch.add(x_3d, abuest)
        xb = torch.add(x_3d, out)
        xc = torch.add(abuest, out)

        x = xa+xb+xc+out+abuest+x_3d
        #x = xa+xb+xc
        # x =  out

        # x1 = self.conv3(x)
        # x = self.conv1(x)
        # x = torch.cat([x, x1], 1)
        # x = self.dconv2(x)


        abu_est = self.Decoder(x)  #  encoder cls_emb = self.model(x)
        # cls_emb = self.vtrans(abu_est) # vtrans  abu_est
        # cls_emb = cls_emb.view(1, self.P, -1)    # view 改变 tensor 形状 #当view(-1)时，其作用是将一个矩阵展开为一个向量
        # abu_est = self.upscale(cls_emb).view(1, self.P, self.size, self.size)

        abu_est = self.smooth(abu_est)  #  对python数据进行平滑处理   #  求出了丰度  abu_est
        re_result = self.decoder(abu_est)
        return abu_est, re_result





    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)



class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)


class Train_test:
    def __init__(self, dataset, device, skip_train=False, save=True):# save=False
        super(Train_test, self).__init__()
        self.skip_train = skip_train
        self.device = device
        self.dataset = dataset
        self.save = save
        self.save_dir = "trans_mod_" + dataset + "/"
        os.makedirs(self.save_dir, exist_ok=True)
        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95
            self.h, self.w, self.hws, self.inplans, self.planes, self.A = 95, 95, 95, 272, 272, 150 # 156->272
            self.LR, self.EPOCH = 1e-3, 100 #6e-3
            self.patch, self.dim = 5, 200  #5 200
            self.beta, self.gamma = 5e3, 3e-3
            self.weight_decay_param = 2e-5 #4e-5
            self.order_abd, self.order_endmem = (0, 1, 2), (0, 1, 2)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        elif dataset == 'apex':
            self.P, self.L, self.col = 4, 285, 110
            self.h, self.w, self.hws, self.inplans, self.planes, self.A = 110, 110, 110, 432, 432, 200
            self.LR, self.EPOCH = 1e-4, 100  # 9e-3
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 1e5, 1e-4
            self.weight_decay_param =4e-5  # 4e-5
            self.order_abd, self.order_endmem = (0, 1, 2, 3), (0, 1, 2, 3) # 3 1 2 0
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        elif dataset == 'sim':#sim效果不好
            self.P, self.L, self.col = 6, 224, 105
            self.h, self.w, self.hws, self.inplans, self.planes, self.A = 105, 105, 105, 400, 400, 150
            self.LR, self.EPOCH = 1e-3, 40
            self.patch, self.dim = 5, 100 # 200
            self.beta, self.gamma = 1e3, 1e-5
            self.weight_decay_param = 3e-5
            self.order_abd, self.order_endmem = (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5) #(0,2,1,4,5,3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        elif dataset == 'jasper2':#EGU带不动
            self.P, self.L, self.col = 4, 198, 100
            self.h, self.w, self.hws, self.inplans, self.planes, self.A = 200, 200, 200, 400, 400, 200
            self.LR, self.EPOCH = 1e-3, 100
            self.patch, self.dim = 5, 200 # 200
            self.beta, self.gamma = 3e2, 1e-2
            self.weight_decay_param = 3e-5
            self.order_abd, self.order_endmem = (0, 1, 2, 3), (0, 1, 2, 3) #(0,2,1,4,5,3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        elif dataset == 'urban1': # RE:20尺寸太小，无法下采样迭代;# urban:数据类型不对
            self.P, self.L, self.col = 5, 162, 307
            self.h, self.w, self.hws, self.inplans, self.planes, self.A = 307, 307, 307, 400, 400, 450
            self.LR, self.EPOCH = 1e-6, 200
            self.patch, self.dim = 1, 100 # 200
            self.beta, self.gamma = 5e3, 1e-2
            self.weight_decay_param = 3e-5
            self.order_abd, self.order_endmem = (0, 1, 2,3,4), (0, 1, 2,3,4) #(0,2,1,4,5,3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        elif dataset == 'dc':
            self.P, self.L, self.col = 4, 224, 104
            self.h, self.w, self.hws, self.inplans, self.planes, self.A = 104, 104, 104, 256, 256, 100
            self.LR, self.EPOCH = 1e-2, 30  #4e-5稳定
            self.patch, self.dim = 4, 100 # 200
            self.beta, self.gamma = 1e5, 1e-4 #  1e-4(rmse/sad)
            self.weight_decay_param = 3e-5
            self.order_abd, self.order_endmem = (0, 1, 2, 3), (0, 1, 2, 3) #(0,2,1,4,5,3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        else:
            raise ValueError("Unknown dataset")

    def run(self, smry):
        net = AutoEncoder(P=self.P, L=self.L, size=self.col,
                          patch=self.patch, dim=self.dim, inplans=self.inplans, planes=self.planes, hws=self.hws, A=self.A).to(self.device)

        if smry:
            summary(net, (1, self.L, self.col, self.col), batch_dim=None)
            return

        net.apply(net.weights_init)

        model_dict = net.state_dict()
        model_dict['decoder.0.weight'] = self.init_weight
        net.load_state_dict(model_dict)

        loss_func = nn.MSELoss(reduction='mean')
        loss_func2 = utils.SAD(self.L)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
        apply_clamp_inst1 = NonZeroClipper()
        
        if not self.skip_train:
            time_start = time.time()
            net.train()
            epo_vs_los = []
            for epoch in range(self.EPOCH):
                for i, (x, _) in enumerate(self.loader):

                    x = x.transpose(1, 0).view(1, -1, self.col, self.col)
                    abu_est, re_result = net(x)

                    loss_re = self.beta * loss_func(re_result, x)
                    loss_sad = loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                          x.view(1, self.L, -1).transpose(1, 2))
                    loss_sad = self.gamma * torch.sum(loss_sad).float()

                    total_loss = loss_re + loss_sad

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()

                    net.decoder.apply(apply_clamp_inst1)
                    
                    if epoch % 10 == 0:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data)

                    epo_vs_los.append(float(total_loss.data))

                scheduler.step()
            time_end = time.time()
            
            if self.save:
                with open(self.save_dir + 'weights_new.pickle', 'wb') as handle:
                    pickle.dump(net.state_dict(), handle)
                sio.savemat(self.save_dir + f"{self.dataset}_losses.mat", {"losses": epo_vs_los})
            
            print('Total computational cost:', time_end - time_start)

        else:
            with open(self.save_dir + 'weights.pickle', 'rb') as handle:
                net.load_state_dict(pickle.load(handle))

        # Testing ================

        net.eval()
        x = self.data.get("hs_img").transpose(1, 0).view(1, -1, self.col, self.col)
        abu_est, re_result = net(x)
        abu_est = abu_est / (torch.sum(abu_est, dim=1))
        abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
        true_endmem = self.data.get("end_mem").numpy()
        est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
        est_endmem = est_endmem.reshape((self.L, self.P))

        abu_est = abu_est[:, :, self.order_abd]
        est_endmem = est_endmem[:, self.order_endmem]

        sio.savemat(self.save_dir + f"{self.dataset}_abd_map.mat", {"A_est": abu_est})
        sio.savemat(self.save_dir + f"{self.dataset}_endmem.mat", {"E_est": est_endmem})

        x = x.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
        re_result = re_result.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
        re = utils.compute_re(x, re_result)
        print("RE:", re)

        rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
        print("class-wise RMSE value:")
        for i in range(self.P):
            print("class", i + 1, ":", rmse_cls[i])
        print("Mean RMSE:", mean_rmse)

        sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
        print("class-wise SAD value:")
        for i in range(self.P):
            print("class", i + 1, ":", sad_cls[i])
        print("Mean SAD:", mean_sad)

        with open(self.save_dir + "log1.csv", 'a') as file:
            file.write(f"LR: {self.LR}, ")
            file.write(f"WD: {self.weight_decay_param}, ")
            file.write(f"RE: {re:.4f}, ")
            file.write(f"SAD: {mean_sad:.4f}, ")
            file.write(f"RMSE: {mean_rmse:.4f}\n")

        plots.plot_abundance(target, abu_est, self.P, self.save_dir)
        plots.plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir)


# =================================================================

if __name__ == '__main__':
     pass
