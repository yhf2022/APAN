import torch
import torch.nn as nn

class CA_Block(nn.Module):
    def __init__(self,h,w,channel,reduction):
        super(CA_Block,self).__init__()
        self.h = h
        self.w = w
        self.x_avg_pool = nn.AdaptiveAvgPool2d((h,1))
        self.y_avg_pool = nn.AdaptiveAvgPool2d((1,w))
        self.conv2d_bn_nonlinear = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.BatchNorm2d(channel//reduction),
            nn.LeakyReLU(0.1))
        self.conv2d_sigmod = nn.Sequential(
            nn.Conv2d(channel//reduction,channel,1,bias=False),
            nn.Sigmoid())
    
    def forward(self,x1,x2):
        x1_h = self.x_avg_pool(x1).permute(0, 1, 3, 2)
        x1_w = self.y_avg_pool(x1)
        x2_h = self.x_avg_pool(x2).permute(0, 1, 3, 2)
        x2_w = self.y_avg_pool(x2)

        y1 = torch.cat([x1_h,x1_w],3)
        y1 = self.conv2d_bn_nonlinear(y1)
        y2 = torch.cat([x2_h,x2_w],3)
        y2 = self.conv2d_bn_nonlinear(y2)


        x1_h,x1_w = y1.split([self.h,self.w],3)
        x1_h = self.conv2d_sigmod(x1_h)
        x1_w = self.conv2d_sigmod(x1_w)
        x2_h,x2_w = y2.split([self.h,self.w],3)
        x2_h = self.conv2d_sigmod(x2_h)
        x2_w = self.conv2d_sigmod(x2_w)


        x1 = x1*x1_h.expand_as(x1)*x1_w.expand_as(x1)
        x2 = x2*x1_h.expand_as(x2)*x1_w.expand_as(x2)

        return x1,x2
class Channels_Attention(nn.Module):
    def __init__(self,channels,reduction):
        super(Channels_Attention,self).__init__()
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels*2, channels//reduction),
            nn.LeakyReLU(0.1),
            nn.Linear(channels//reduction, channels*2),
            nn.Sigmoid())
    def forward(self,x1,x2):
        b, c, _, _ = x1.size()
        x1 = self.avepool(x1).view(b,c,)
        x2 = self.avepool(x2).view(b,c,)
        x = torch.cat([x1,x2],dim=1)
        x = self.fc(x)
        x1,x2 = x.split([c,c],1)
        x1 = x1.view(b,c,1,1)
        x2 = x2.view(b,c,1,1)
        
        return x1,x2
class Spatial_Attention(nn.Module):
    def __init__(self,channels,length):
        super(Spatial_Attention, self).__init__()
        self.conv_3x3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding = 3//2)
        self.resize_bilinear = nn.Upsample([length,length],mode ='bilinear')
        self.sigmoid = nn.Sigmoid()
        self.conv_1x1 = nn.Conv2d(2,2,1,1,0)

    def forward(self,x1,x2):
        avgout1 = torch.mean(x1, dim=1, keepdim=True)
        avgout2 = torch.mean(x2, dim=1, keepdim=True)
        x = torch.cat([avgout1,avgout2], dim=1)
        x = self.conv_3x3(x)
        x = self.resize_bilinear(x)
        x = self.conv_1x1(x)
        x1,x2 = x.split([1,1],1)

        
        return x1,x2


class CSCAv2(nn.Module):
    def __init__(self,channels,reduction,length):
        super(CSCAv2,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.CA = CA_Block(length,length,channels,reduction)
        self.SCA = Spatial_Attention(channels,length)
        self.conv1 = nn.Conv2d(channels,channels,1,1,0)
        self.conv2 = nn.Conv2d(1,channels,1,1,0)
    def forward(self,x1,x2):
        out1,out2 =x1,x2
        c1,c2 = self.CA(x1,x2)
        s1,s2 = self.SCA(x1,x2)
        s1 = self.conv2(s1)
        s2 = self.conv2(s2)
        a1 = c1.expand_as(s1)*s1
        a1 = self.sigmoid(self.conv1(a1))
        a2 = c2.expand_as(s2)*s2
        a2 = self.sigmoid(self.conv1(a2))
        out1 = out1*a1
        out2 = out2*a2
        out = torch.cat([out1,out2],dim=1)


        return out