class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1, t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        filters = [16,32,64,128, 256]
        
        self.Conv1 = RRCNN_block(ch_in=img_ch,ch_out=filters[0], t=t)
        self.Conv2 = RRCNN_block(ch_in=filters[0],ch_out=filters[1], t=t)
        self.Conv3 = RRCNN_block(ch_in=filters[1],ch_out=filters[2], t=t)
        self.Conv4 = RRCNN_block(ch_in=filters[2],ch_out=filters[3], t=t)
        
        self.Up5 = up_conv(ch_in=filters[3],ch_out=filters[3])
        self.Up_conv5 = RRCNN_block(ch_in=filters[4], ch_out=filters[2], t=t)

        self.Up4 = up_conv(ch_in=filters[2],ch_out=filters[2])
        self.Up_conv4 = RRCNN_block(ch_in=filters[3], ch_out=filters[1], t=t)
        
        self.Up3 = up_conv(ch_in=filters[1],ch_out=filters[1])
        self.Up_conv3 = RRCNN_block(ch_in=filters[2], ch_out=filters[0], t=t)
        
        self.Up2 = up_conv(ch_in=filters[0],ch_out=filters[0])
        self.Up_conv2 = RRCNN_block(ch_in=filters[1], ch_out=filters[0], t=t)

        self.Conv_1x1 = nn.Conv2d(filters[0],output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        indices = []
        sizes = []
        
        x1 = self.Conv1(x)
        sizes.append(x1.size())
        x2,ind = self.Maxpool(x1)
        indices.append(ind)
        
        x2 = self.Conv2(x2)
        sizes.append(x2.size())
        x3,ind = self.Maxpool(x2)
        indices.append(ind)
        
        x3 = self.Conv3(x3)
        sizes.append(x3.size())
        x4,ind = self.Maxpool(x3)
        indices.append(ind)        
        
        x4 = self.Conv4(x4)
        sizes.append(x4.size())
        x5,ind = self.Maxpool(x4)
        indices.append(ind)
                        
        # decoding + concat path
        
        size = sizes.pop()
        ind = indices.pop()           
        d5 = self.Up5(x5,ind,size)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
                
        size = sizes.pop()
        ind = indices.pop()
        d4 = self.Up4(d5, ind, size)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
                
        size = sizes.pop()
        ind = indices.pop()
        d3 = self.Up3(d4, ind, size)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        size = sizes.pop()
        ind = indices.pop()
        d2 = self.Up2(d3, ind, size)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)       
        
        d1 = self.Conv_1x1(d2)

        return d1
