import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset
class VGGBlock(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 5,stride=(1,1), padding=2)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 5,stride=(1,1), padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if out_channels>8:
            self.SE = Squeeze_Excite(out_channels,8)
        else:
            self.SE = None
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.SE is not None:
            out = self.SE(out)
        
        return(out)
class Squeeze_Excite(nn.Module):
    
    def __init__(self,channel,reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class UNet(nn.Module):
    
    def __init__(self, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 256]

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[4]*2, nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[3]*2, nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[2]*2, nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[1]*2, nb_filter[0], nb_filter[0])
        self.convf = VGGBlock(nb_filter[0]*2, input_channels, input_channels)
        
        self.sizes = []
        self.tensors = []
        self.indices = []
        #self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, m):
        # print(m.shape)
        x0_0 = self.conv0_0(m)
        self.tensors.append(x0_0)
        self.sizes.append(x0_0.size())
        x0_0, ind = self.pool(x0_0)
        self.indices.append(ind)
        
        # print(x0_0.shape)
        x1_0 = self.conv1_0(x0_0)
        self.tensors.append(x1_0)
        self.sizes.append(x1_0.size())
        x1_0, ind = self.pool(x1_0)
        self.indices.append(ind)
        
        # print(x1_0.shape)
        x2_0 = self.conv2_0(x1_0)
        self.tensors.append(x2_0)
        self.sizes.append(x2_0.size())
        x2_0, ind = self.pool(x2_0)
        self.indices.append(ind)
        
        # print(x2_0.shape)
        x3_0 = self.conv3_0(x2_0)
        self.tensors.append(x3_0)
        self.sizes.append(x3_0.size())
        x3_0, ind = self.pool(x3_0)
        self.indices.append(ind)
        
        # print("x3_0:",x3_0.shape)
        x4_0 = self.conv4_0(x3_0)
        self.tensors.append(x4_0)
        self.sizes.append(x4_0.size())
        x4_0, ind = self.pool(x4_0)
        self.indices.append(ind)
        # print("x4_0:",x4_0.shape)           
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        # print("ind:",ind.shape)
        
        x = F.max_unpool2d(x4_0, ind, 2, 2, output_size=size)
        x3_1 = torch.cat([tensor, x], dim=1)
        x3_1 = self.conv3_1(x3_1)        
        # print(x3_1.shape)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x3_1, ind, 2, 2, output_size=size)
        x2_2 = torch.cat([tensor, x], dim=1)
        x2_2 = self.conv2_2(x2_2)        
        # print(x2_2.shape)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x2_2, ind, 2, 2, output_size=size)
        x1_3 = torch.cat([tensor, x], dim=1)
        x1_3 = self.conv1_3(x1_3)        
        # print(x1_3.shape)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x1_3, ind, 2, 2, output_size=size)
        x0_4 = torch.cat([tensor, x], dim=1)
        x0_4 = self.conv0_4(x0_4)        
        # print(x0_4.shape)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x0_4, ind, 2, 2, output_size=size)
        output = torch.cat([tensor, x], dim=1)
        output = self.convf(output)        
        # print(output.shape)
                
        return output
class NestedUNet(nn.Module):
    
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
class DUNet(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        
        nb_filter = [32, 64, 128, 256, 256]

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv3_1 = VGGBlock(nb_filter[4]*2, nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[3]*2, nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[2]*2, nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[1]*2, nb_filter[0], nb_filter[0])
        self.convf = VGGBlock(nb_filter[0]*2, input_channels, input_channels)                                     
        
        self.sizes = []
        self.tensors = []
        self.indices = []        
               
        self.conv01 = VGGBlock(6,nb_filter[0],nb_filter[0])
        self.conv11 = VGGBlock(nb_filter[0],nb_filter[1],nb_filter[1])
        self.conv21 = VGGBlock(nb_filter[1],nb_filter[2],nb_filter[2])
        self.conv31 = VGGBlock(nb_filter[2],nb_filter[3],nb_filter[3])
        self.conv41 = VGGBlock(nb_filter[3],nb_filter[4],nb_filter[4])
        
        self.conv_31 = VGGBlock(nb_filter[4]*2+nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv_22 = VGGBlock(nb_filter[3]*2+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv_13 = VGGBlock(nb_filter[2]*2+nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv_04 = VGGBlock(nb_filter[1]*2+nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_05 = VGGBlock(nb_filter[0]*2, 3, 3)
        
        self.final = VGGBlock(3*2, 3, 3)
    
    def forward(self,m):
        
        x0_0 = self.conv0_0(m)
        self.tensors.append(x0_0)
        self.sizes.append(x0_0.size())
        x0_0, ind = self.pool(x0_0)
        self.indices.append(ind)
        
        #print("x0_0:",x0_0.shape)
        x1_0 = self.conv1_0(x0_0)
        self.tensors.append(x1_0)
        self.sizes.append(x1_0.size())
        x1_0, ind = self.pool(x1_0)
        self.indices.append(ind)
        
        #print("x1_0:",x1_0.shape)
        x2_0 = self.conv2_0(x1_0)
        self.tensors.append(x2_0)
        self.sizes.append(x2_0.size())
        x2_0, ind = self.pool(x2_0)
        self.indices.append(ind)
        
        #print("x2_0:",x2_0.shape)
        x3_0 = self.conv3_0(x2_0)
        self.tensors.append(x3_0)
        self.sizes.append(x3_0.size())
        x3_0, ind = self.pool(x3_0)
        self.indices.append(ind)
        
        #print("x3_0:",x3_0.shape)
        x4_0 = self.conv4_0(x3_0)
        self.tensors.append(x4_0)
        self.sizes.append(x4_0.size())
        x4_0, ind = self.pool(x4_0)
        self.indices.append(ind)
        #print("x4_0:",x4_0.shape)           
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        # print("ind:",ind.shape)
        
        x = F.max_unpool2d(x4_0, ind, 2, 2, output_size=size)
        x3_1 = torch.cat([tensor, x], dim=1)
        x3_1 = self.conv3_1(x3_1)        
        # print(x3_1.shape)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x3_1, ind, 2, 2, output_size=size)
        x2_2 = torch.cat([tensor, x], dim=1)
        x2_2 = self.conv2_2(x2_2)        
        # print(x2_2.shape)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x2_2, ind, 2, 2, output_size=size)
        x1_3 = torch.cat([tensor, x], dim=1)
        x1_3 = self.conv1_3(x1_3)        
        # print(x1_3.shape)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x1_3, ind, 2, 2, output_size=size)
        x0_4 = torch.cat([tensor, x], dim=1)
        x0_4 = self.conv0_4(x0_4)        
        #print(x0_4.shape)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x0_4, ind, 2, 2, output_size=size)
        output = torch.cat([tensor, x], dim=1)
        output = self.convf(output)        
        #print("output:",output.shape)
        
        x = torch.cat([m,output],axis=1)
        #print(x.shape)
        
        x1_1 = self.conv01(x)
        self.tensors.append(x1_1)
        self.sizes.append(x1_1.size())
        x1_1, ind = self.pool(x1_1)
        self.indices.append(ind)        
        #print(x1_1.shape)
        
        x2_1 = self.conv11(x1_1)
        self.tensors.append(x2_1)
        self.sizes.append(x2_1.size())
        x2_1, ind = self.pool(x2_1)
        self.indices.append(ind)
        #print(x2_1.shape)
        
        x3_1 = self.conv21(x2_1)
        self.tensors.append(x3_1)
        self.sizes.append(x3_1.size())
        x3_1, ind = self.pool(x3_1)
        self.indices.append(ind)
        #print("x3_1:",x3_1.shape)
        
        x4_1 = self.conv31(x3_1)
        self.tensors.append(x4_1)
        self.sizes.append(x4_1.size())
        x4_1, ind = self.pool(x4_1)
        self.indices.append(ind)
        #print("x4_1:",x4_1.shape)
        
        x5_1 = self.conv41(x4_1)
        self.tensors.append(x5_1)
        self.sizes.append(x5_1.size())
        x5_1, ind = self.pool(x5_1)
        self.indices.append(ind)
        #print(x5_1.shape)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x5_1, ind, 2, 2, output_size=size)
        x = torch.cat([tensor, x, x3_0], dim=1)
        x = self.conv_31(x)
                
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
        x = torch.cat([tensor, x, x2_0], dim=1)
        x = self.conv_22(x)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
        x = torch.cat([tensor, x, x1_0], dim=1)
        x = self.conv_13(x)
              
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
        x = torch.cat([tensor, x, x0_0], dim=1)
        x = self.conv_04(x)
        
        tensor = self.tensors.pop()
        size = self.sizes.pop()
        ind = self.indices.pop()
        
        x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
        x = torch.cat([tensor, x], dim=1)
        x = self.conv_05(x)
        #print(output.shape)
        
        x = torch.cat([output, x], dim=1)
        output_final = self.final(x)
        # print(output_final.shape)
        return output_final
