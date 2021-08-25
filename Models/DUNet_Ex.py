
class DUNetEx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=3,
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None):
        super().__init__()
        assert len(filters) > 0
        self.final_activation = final_activation
        self.encoder1 = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)
        self.encoder2 = create_encoder(in_channels*2, filters, kernel_size, weight_norm, batch_norm, activation, layers)
        decoders1 = []
        decoders2 = []
        for i in range(out_channels):
            decoders1.append(create_decoder(1, filters, kernel_size, weight_norm, batch_norm, activation, layers, concat_layer=2))
            decoders2.append(create_decoder(1, filters, kernel_size, weight_norm, batch_norm, activation, layers, concat_layer=3))
        self.decoders1 = nn.Sequential(*decoders1)
        self.decoders2 = nn.Sequential(*decoders2)
        
    def encode(self, x, switch):        
        if switch==0:
            self.encoder = self.encoder1
        elif switch==1:
            self.encoder = self.encoder2
            
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

    def decode(self, _x, _tensors, _indices, _sizes, switch):        
        if switch==0:
            self.decoders = self.decoders1
        elif switch==1:
            self.decoders = self.decoders2
            
        y = []
        for _decoder in self.decoders:
            x = _x
            tensors = _tensors[:]
            indices = _indices[:]
            sizes = _sizes[:]
            for decoder in _decoder:
                tensor = tensors.pop()
                size = sizes.pop()
                ind = indices.pop()
                x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
                x = torch.cat([tensor, x], dim=1)
                x = decoder(x)
            y.append(x)
        return torch.cat(y, dim=1)

    def decode_(self, _tensors1, _x2, _tensors2, _indices2, _sizes2, switch):        
        if switch==0:
            self.decoders = self.decoders1
        elif switch==1:
            self.decoders = self.decoders2
            
        y = []
        for _decoder in self.decoders:
            x = _x2
            tensors1 = _tensors1[:]
            tensors2 = _tensors2[:]
            indices2 = _indices2[:]
            sizes2 = _sizes2[:]
            
            for decoder in _decoder:
                tensor1 = tensors1.pop()
                tensor2 = tensors2.pop()
                size2 = sizes2.pop()
                ind2 = indices2.pop()
                x = F.max_unpool2d(x, ind2, 2, 2, output_size=size2)
                x = torch.cat([tensor1, tensor2, x], dim=1)
                x = decoder(x)
            y.append(x)
        return torch.cat(y, dim=1)
    
    def forward(self, x):
        x1, tensors1, indices1, sizes1 = self.encode(x,0)
        y1 = self.decode(x1, tensors1, indices1, sizes1,0)
        y1 = torch.cat([x, y1], dim=1)
        # print(torch.add(x,y1).shape)
        x2, tensors2, indices2, sizes2 = self.encode(y1,1)
        y2 = self.decode_(tensors1, x2, tensors2, indices2, sizes2, 1)
        if self.final_activation is not None:
            y2 = self.final_activation(y2)
        return y2
