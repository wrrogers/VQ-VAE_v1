# Adapted from zhulf0804 at https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch

import numpy as np
import torch
import torch.nn as nn

from dbcr import DenseBlockChannelReduction as DBCR


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels, n):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv2d(in_channels, n, 3, stride=1, padding=1, bias=False), # 149 x 149 x 32
            Conv2d(n, n,   3, stride=1, padding=1, bias=False), # 147 x 147 x 32
            Conv2d(n, n*2, 3, stride=1, padding=1, bias=False), # 147 x 147 x 64
            #nn.MaxPool2d(3, stride=1, padding=1), # 73 x 73 x 64
            Conv2d(n*2, n*3, 1, stride=1, padding=0, bias=False), # 73 x 73 x 80
            Conv2d(n*3, n*6, 3, stride=1, padding=1, bias=False), # 71 x 71 x 192
            #nn.MaxPool2d(3, stride=2, padding=0), # 35 x 35 x 192
        )
        self.branch_0 = Conv2d(n*6, n*3, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(n*6, n*2, 1, stride=1, padding=0, bias=False),
            Conv2d(n*2, n*2, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(n*6, n*2, 1, stride=1, padding=0, bias=False),
            Conv2d(n*2, n*3, 3, stride=1, padding=1, bias=False),
            Conv2d(n*3, n*3, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(n*6, n*2, 1, stride=1, padding=0, bias=False)
        )
        
    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x = torch.cat((x0, x1, x2, x3), dim=1)
        return x


class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, n, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, n, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, n, 1, stride=1, padding=0, bias=False),
            Conv2d(n, n, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, n, 1, stride=1, padding=0, bias=False),
            Conv2d(n,   n*2, 3, stride=1, padding=1, bias=False),
            Conv2d(n*2, n*2, 3, stride=1, padding=1, bias=False)
        )
        self.conv = nn.Conv2d(n*4, n*10, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        x = self.relu(x + self.scale * x_res)
        return x


class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=2, padding=1, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=1, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=1)
        
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x = torch.cat((x0, x1, x2), dim=1)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024


class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, n, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, n*6, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, n*4, 1, stride=1, padding=0, bias=False),
            Conv2d(n*4, n*5, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(n*5, n*6, (7, 1), stride=1, padding=(3, 0), bias=False)
        )
        self.conv = nn.Conv2d(n*12, n*34, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        x = self.relu(x + self.scale * x_res)
        return x


class Reduction_B(nn.Module):
    def __init__(self, in_channels, n):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, n*8, 1, stride=1, padding=0, bias=False),
            Conv2d(n*8, n*12, 3, stride=2, padding=1, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, n*8, 1, stride=1, padding=0, bias=False),
            Conv2d(n*8, n*9, 3, stride=2, padding=1, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, n*8, 1, stride=1, padding=0, bias=False),
            Conv2d(n*8, n*9, 3, stride=1, padding=1, bias=False),
            Conv2d(n*9, n*9, 3, stride=2, padding=1, bias=False)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=1)
    
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x = torch.cat((x0, x1, x2, x3), dim=1)
        return x    
    
    
class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, n, scale=1.0):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, n*6, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, n*6, 1, stride=1, padding=0, bias=False),
            Conv2d(n*6, n*7, (1, 3), stride=1, padding=(0, 1), bias=False),
            Conv2d(n*7, n*8, (3, 1), stride=1, padding=(1, 0), bias=False)
        )
        self.conv = nn.Conv2d(n*14, n*64, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class DenseLayer(nn.Module):
    def __init__(self, n, m, growth_rate):
        super(DenseLayer, self).__init__()
        #print(n, m, growth_rate)
        self.conv1 = nn.Conv2d(in_channels=growth_rate//n, out_channels=growth_rate//m, kernel_size=1, stride=1)
        #self.conv2 = nn.Conv2d(in_channels=growth_rate*n, out_channels=growth_rate, kernel_size=3, stride=1, padding=1)
        
        #self.bn1 = nn.BatchNorm2d(growth_rate*n)
        #self.bn2 = nn.BatchNorm2d(growth_rate*4)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class Dense_Channel_Reduction(nn.Module):
    def __init__(self, channels):
        super(Dense_Channel_Reduction, self).__init__()
        num_layers = np.log2(channels).astype(np.int32) + 1 - 4
        self.steps = [2**i for i in range(num_layers)]
        self.steps[0] = 1
        self.channels = channels
        self.layers = nn.ModuleList([DenseLayer(self.steps[n], 
                                                self.steps[n+2], channels) for n in range(num_layers-2)])
        self.conv_out = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1).cuda()
        
    def forward(self, x):       
        for n, layer in enumerate(self.layers):
            conv = nn.Conv2d(in_channels =self.channels//self.steps[n], 
                             out_channels=self.channels//self.steps[n+2], 
                             kernel_size=1, stride=1).cuda()
            n = n + 1
            x_in = conv(x)
            l = layer(x)
            x = torch.cat((x_in, l), 1)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base=32, k=256, l=256, m=384, n=384):
        super(Encoder, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels, base))
        for i in range(1):
            blocks.append(Inception_ResNet_A(base*10, base, 0.17))
        blocks.append(Reduction_A(base*10, k, l, m, n))
        for i in range(1):
            blocks.append(Inception_ResNet_B(base*34, base, 0.10))
        blocks.append(Reduction_B(base*34, base))
        for i in range(1):
            blocks.append(Inception_ResNet_C(base*64, base, 0.20))
        #blocks.append(Inception_ResNet_C(base*64, base, activation=False))
        blocks.append(Dense_Channel_Reduction(base*64))
        #blocks.append(Inception_Channel_Reduction(base*64))
        self.features = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((4, 3, 128, 128))
    x = torch.tensor(x).float().cuda()

    # test encoder
    model = Encoder().cuda()
    out = model(x)
    print()
    print('Out shape:', out.shape)






