from collections import OrderedDict

import torch
import torch.nn as nn

from nets.CSPdarknet import darknet53
from nets.attention import CSCAv2


def conv_bn_LR(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

class SPP(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SPP, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv_bn_LR(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

def three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv_bn_LR(in_filters, filters_list[0], 1),
        conv_bn_LR(filters_list[0], filters_list[1], 3),
        conv_bn_LR(filters_list[1], filters_list[0], 1),
    )
    return m

def five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv_bn_LR(in_filters, filters_list[0], 1),
        conv_bn_LR(filters_list[0], filters_list[1], 3),
        conv_bn_LR(filters_list[1], filters_list[0], 1),
        conv_bn_LR(filters_list[0], filters_list[1], 3),
        conv_bn_LR(filters_list[1], filters_list[0], 1),
    )
    return m

def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_bn_LR(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

class APAN(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(APAN, self).__init__()

        self.backbone = darknet53(None)

        self.conv1 = three_conv([512,1024],1024)
        self.SPP = SPP()
        self.conv2 = three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = conv_bn_LR(512,256,1)
        self.make_five_conv1 = five_conv([256, 512],512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv_bn_LR(256,128,1)
        self.make_five_conv2 = five_conv([128, 256],256)

        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)

        self.down_sample1 = conv_bn_LR(128,256,3,stride=2)
        self.make_five_conv3 = five_conv([256, 512],512)

        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)

        self.down_sample2 = conv_bn_LR(256,512,3,stride=2)
        self.make_five_conv4 = five_conv([512, 1024],1024)

        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)

        self.make_five_conv0 = five_conv([64,128],128)
        self.upsample0 = Upsample(128,64)

        self.conv_for_P2 = conv_bn_LR(128,64,1)
        self.down_sample0 = conv_bn_LR(64,128,3,stride=2)
        final_out_filter4 = num_anchors * (5 + num_classes)
        self.yolo_head4 = yolo_head([128,final_out_filter4],64)
        self.make_five_conv6 = five_conv([128,256],256)

        self.cscav2_0 = CSCAv2(64,8,104)
        self.cscav2_1 = CSCAv2(128,16,52)
        self.cscav2_2 = CSCAv2(256,16,26)
        self.cscav2_3 = CSCAv2(512,16,13)

    def forward(self, x):

        x3, x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P4 = self.conv_for_P4(x1)

        P4_upsample = self.upsample2(P4)

        P3 = self.conv_for_P3(x2)
        P3 = self.cscav2_1(P3,P4_upsample)
        P3 = self.make_five_conv2(P3)

        P3_upsample = self.upsample0(P3)

        P2 = self.conv_for_P2(x3) 
        P2 = self.cscav2_0(P2,P3_upsample)
        P2 = self.make_five_conv0(P2)

        P2_downsample = self.down_sample0(P2)

        P3 = torch.cat([P2_downsample,P3],axis=1)
        P3 =self.make_five_conv6(P3)

        P3_downsample = self.down_sample1(P3)

        P4 = torch.cat([P3_downsample,P4],axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)

        P5 = torch.cat([P4_downsample,P5],axis=1)
        P5 = self.make_five_conv4(P5)

        out3 = self.yolo_head4(P2)
        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2, out3

