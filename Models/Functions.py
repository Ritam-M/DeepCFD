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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from Models.AutoEncoder import create_layer

def create_encoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, layers=2):
    encoder = []
    for i in range(layers):
        _in = out_channels
        _out = out_channels
        if i == 0:
            _in = in_channels
        encoder.append(create_layer(_in, _out, kernel_size, wn, bn, activation, nn.Conv2d))
    return nn.Sequential(*encoder)

def create_decoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, layers=2, final_layer=False, concat_layer = 2):
    decoder = []
    for i in range(layers):
        _in = in_channels
        _out = in_channels
        _bn = bn
        _activation = activation
        if i == 0:
            _in = in_channels * concat_layer
        if i == layers - 1:
            _out = out_channels
            if final_layer:
                _bn = False
                _activation = None
        decoder.append(create_layer(_in, _out, kernel_size, wn, _bn, _activation, nn.ConvTranspose2d))
    return nn.Sequential(*decoder)

def create_encoder(in_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2):
    encoder = []
    for i in range(len(filters)):
        if i == 0:
            encoder_layer = create_encoder_block(in_channels, filters[i], kernel_size, wn, bn, activation, layers)
        else:
            encoder_layer = create_encoder_block(filters[i-1], filters[i], kernel_size, wn, bn, activation, layers)
        encoder = encoder + [encoder_layer]
    return nn.Sequential(*encoder)

def create_decoder(out_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2, concat_layer=2):
    decoder = []
    for i in range(len(filters)):
        if i == 0:
            decoder_layer = create_decoder_block(filters[i], out_channels, kernel_size, wn, bn, activation, layers, final_layer=True, concat_layer=concat_layer)
        else:
            decoder_layer = create_decoder_block(filters[i], filters[i-1], kernel_size, wn, bn, activation, layers, final_layer=False, concat_layer=concat_layer)
        decoder = [decoder_layer] + decoder
    return nn.Sequential(*decoder)

def create_attention(out_channels, filters, kernel_size, wn=True, bn=True, device="gpu",activation=nn.ReLU, layers=2, concat_layer=2):
    Att_layers = []
    for i in range(len(filters)):
        if i == 0:
            att_layer = Attention_block(filters[i], filters[i], out_channels)
            if device=="gpu":att_layer.cuda()
        else:
            att_layer = Attention_block(filters[i], filters[i], filters[i-1])
            if device=="gpu":att_layer.cuda()
        Att_layers = [att_layer] + Att_layers
    return Att_layers

def create_dense(in_channels, out_channels):
    layers = []
    layers.append(F.relu(nn.Linear(in_channels, out_channels)))
    layers.append(nn.Dropout(0.2))
    #layers.append(F.relu(nn.Linear(out_channels,out_channels)))
    #layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(out_channels,out_channels))
    
    return nn.Sequential(*layers)

  class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm2d(1),
            # nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
