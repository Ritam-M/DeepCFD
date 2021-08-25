class Att_R2UNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(Att_R2UNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        filters = [64,128, 256, 512, 1024]
        
        self.Conv1 = RRCNN_block(ch_in=img_ch,ch_out=filters[0])
        self.Conv2 = RRCNN_block(ch_in=filters[0],ch_out=filters[1])
        self.Conv3 = RRCNN_block(ch_in=filters[1],ch_out=filters[2])
        self.Conv4 = RRCNN_block(ch_in=filters[2],ch_out=filters[3])
        
        self.Up5 = up_conv(ch_in=filters[3],ch_out=filters[3])
        self.Att5 = Attention_block(F_g=filters[3],F_l=filters[3],F_int=filters[2])
        self.Up_conv5 = RRCNN_block(ch_in=filters[4], ch_out=filters[2])

        self.Up4 = up_conv(ch_in=filters[2],ch_out=filters[2])
        self.Att4 = Attention_block(F_g=filters[2],F_l=filters[2],F_int=filters[1])
        self.Up_conv4 = RRCNN_block(ch_in=filters[3], ch_out=filters[1])
        
        self.Up3 = up_conv(ch_in=filters[1],ch_out=filters[1])
        self.Att3 = Attention_block(F_g=filters[1],F_l=filters[1],F_int=filters[0])
        self.Up_conv3 = RRCNN_block(ch_in=filters[2], ch_out=filters[0])
        
        self.Up2 = up_conv(ch_in=filters[0],ch_out=filters[0])
        self.Att2 = Attention_block(F_g=filters[0],F_l=filters[0],F_int=filters[0]//2)
        self.Up_conv2 = RRCNN_block(ch_in=filters[1], ch_out=filters[0])

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
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
                
        size = sizes.pop()
        ind = indices.pop()
        d4 = self.Up4(d5, ind, size)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
                
        size = sizes.pop()
        ind = indices.pop()
        d3 = self.Up3(d4, ind, size)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        size = sizes.pop()
        ind = indices.pop()
        d2 = self.Up2(d3, ind, size)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)       
        
        d1 = self.Conv_1x1(d2)

        return d1
