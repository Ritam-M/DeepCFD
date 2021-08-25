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

class Att_UNetEx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=3,
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None):
        super().__init__()
        assert len(filters) > 0
        self.final_activation = final_activation
        self.encoder = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)
        decoders = []
        attentions = []
        for i in range(out_channels):
            decoders.append(create_decoder(1, filters, kernel_size, weight_norm, batch_norm, activation, layers))
            attentions.append(create_attention(1, filters, kernel_size, weight_norm, batch_norm, activation, layers))
        self.decoders = nn.Sequential(*decoders)
        self.attentions = attentions

    def encode(self, x):
        tensors = []
        indices = []
        sizes = []
        for encoder in self.encoder:
            x = encoder(x)
            sizes.append(x.size())
            tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)
        return x, tensors, indices, sizes

    def decode(self, _x, _tensors, _indices, _sizes):
        y = []
        for _decoder,_attention in zip(self.decoders,self.attentions):
            x = _x
            tensors = _tensors[:]
            indices = _indices[:]
            sizes = _sizes[:]
            for decoder,attention in zip(_decoder,_attention):
                tensor = tensors.pop()
                size = sizes.pop()
                ind = indices.pop()
                x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
                x = attention(tensor,x)
                x = torch.cat([tensor, x], dim=1)
                x = decoder(x)
            y.append(x)
        return torch.cat(y, dim=1)

    def forward(self, x):
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, tensors, indices, sizes)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
