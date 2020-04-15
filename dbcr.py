# Code from Vincent 6606, Pulled from https://github.com/vincent6606/DenseNet/blob/master/DenseNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from model.residual import ResidualStack

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
        #x = self.bn1(x)
                
        #x = self.conv2(x)
        #x = self.relu(x)
        #x = self.bn2(x)
        
        return x
        
class DenseBlockChannelReduction(nn.Module):
    def __init__(self, channels):
        super(DenseBlockChannelReduction, self).__init__()
        num_layers = np.log2(channels).astype(np.int32) + 1 - 4
        self.steps = [2**i for i in range(num_layers)]
        self.steps[0] = 1
        #print(self.steps)
        self.channels = channels
        self.layers = nn.ModuleList([DenseLayer(self.steps[n], 
                                                self.steps[n+2], channels) for n in range(num_layers-2)])
        self.conv_out = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1).cuda()
        #print(self.layers)
        
    def forward(self, x):        
        for n, layer in enumerate(self.layers):
            #print('--------------------------------------------------')
            #print('Before:', x.size())
            #print(self.channels, (n*2))
            conv = nn.Conv2d(in_channels =self.channels//self.steps[n], 
                             out_channels=self.channels//self.steps[n+2], 
                             kernel_size=1, stride=1).cuda()
            #print(conv)
            n = n + 1
            x_in = conv(x)
            #print('X_in:  ', x.size())
            #print('New_in:', x_in.size())
            l = layer(x)
            #print('Layer: ', l.size())
            x = torch.cat((x_in, l), 1)
            #print('After: ', x.size())
        #out = self.conv_out(x)
        #print('--------------------------------------------------')
        return x


if __name__ == "__main__":
    # random data
    channels = 2048
    x = np.random.random_sample((32, channels, 32, 32))
    x = torch.tensor(x).float().cuda()
    
    # test decoder
    dense = DenseBlockChannelReduction(channels).cuda()
    out = dense(x)
    print('Dense out shape:', out.shape)
    

