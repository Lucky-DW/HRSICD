'''
The model is mainly for low-resolution remote sensing images,
so the size of the image input is defaulted to 64*64
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

class PCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PCM,self).__init__()
        in_channels_4 = int(in_channels / 4)
        self.Conv1x1_1 =nn.Conv2d(in_channels,in_channels_4,kernel_size=1)

        self.Conv1x1_3 = nn.Conv2d(in_channels, in_channels_4, kernel_size=1)
        self.Conv3x3 = nn.Conv2d(in_channels_4,in_channels_4,kernel_size=3,padding=1)

        self.Conv1x1_5 = nn.Conv2d(in_channels, in_channels_4, kernel_size=1)
        self.Conv5x5 = nn.Conv2d(in_channels_4,in_channels_4, kernel_size=5,padding=2)

        self.Conv1x1_a = nn.Conv2d(in_channels, in_channels_4, kernel_size=1)

        self.bn=nn.BatchNorm2d(out_channels,eps=0.001)

    def forward(self, x):
        branch1=self.Conv1x1_1(x)

        branch2_1=self.Conv1x1_3(x)
        branch2_2=self.Conv3x3(branch2_1)

        branch3_1=self.Conv1x1_5(x)
        branch3_2=self.Conv5x5(branch3_1)

        branch4_1=self.Conv1x1_a(x)
        branch4_2=F.avg_pool2d(branch4_1,kernel_size=3,stride=1,padding=1)

        outputs=[branch1,branch2_2,branch3_2,branch4_2]
        x= torch.cat(outputs,1)
        x=self.bn(x)
        return F.relu(x,inplace=True)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(

            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SP_Block(nn.Module):
    def __init__(self,channel,h,w,reduction = 4):
        super(SP_Block,self).__init__()
        self.h = h
        self.w = w
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
        self.conv_1x1 = nn.Conv2d(
            in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
    def forward(self,x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out + x

class CrossTransformer(nn.Module):
    def __init__(self, dropout, d_model=1024, n_head=4):
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2):
        # dif_as_kv
        dif = input2 - input1
        output_1 = self.cross(input1, dif)  # (Q,K,V)
        output_2 = self.cross(input2, dif)  # (Q,K,V)

        return output_1,output_2
    def cross(self, input,dif):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input, dif, dif)  # (Q,K,V)

        output = input + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output

class HRSICD(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=64, bilinear=True):
        super(HRSICD, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.bilinear = bilinear
        self.inc = DoubleConv(self.n_channels, 64)
        self.PCM1 = PCM(64, 64)
        self.FE1 = DoubleConv(64, 128)
        self.PCM2 = PCM(128,128)
        self.FE2 = DoubleConv(128, 256)
        self.PCM3 = PCM(256, 256)
        self.FE3 = DoubleConv(256, 512)
        self.PCM4 = PCM(512,512)
        self.SPM1 = SP_Block(64, self.img_size, self.img_size)
        self.SPM2 = SP_Block(128, self.img_size, self.img_size)
        self.SPM3 = SP_Block(256, self.img_size, self.img_size)
        self.SPM4 = SP_Block(512, self.img_size, self.img_size)
        self.SFM = CrossTransformer(dropout=0.1, d_model=512)
        self.fusion1 = DoubleConv(128,64)
        self.fusion2 = DoubleConv(256,128)
        self.fusion3 = DoubleConv(512,256)
        self.fusion4 = DoubleConv(1024,512)

        self.double2single1 = SingleConv(512,256)
        self.sce1 = DoubleConv(512, 256)
        self.double2single2 = SingleConv(256, 128)
        self.sce2 = DoubleConv(256, 128)
        self.double2single3 = SingleConv(128, 64)
        self.sce3 = DoubleConv(128, 64)
        self.out = OutConv(64, n_classes)

    def forward(self,x1,x2):
        B = x1.shape[0]
        opt_en_out = self.encoder(x1)
        sar_en_out = self.encoder(x2)

        opt_out, sar_out = self.SFM(opt_en_out[-1].view(B, 512, self.img_size* self.img_size).permute(0, 2, 1),sar_en_out[-1].view(B, 512, self.img_size* self.img_size).permute(0, 2, 1))

        opt_en_out[-1] = opt_out.permute(0, 2, 1).view(B, 512,self.img_size, self.img_size)
        sar_en_out[-1] = sar_out.permute(0, 2, 1).view(B, 512,self.img_size, self.img_size)

        en_out = [torch.cat((opt_en_out[i], sar_en_out[i]), dim=1) for i in range(len(opt_en_out))]

        en_out = [self.fusion1(en_out[0]), self.fusion2(en_out[1]), self.fusion3(en_out[2]), self.fusion4(en_out[3])]

        opt_de_out = self.decoder(en_out)

        out = torch.sigmoid(self.out(opt_de_out))

        return out

    def encoder(self, x):
        x1 = self.inc(x)
        x1 = self.PCM1(x1)
        x2 = self.FE1(x1)
        x2= self.PCM2(x2)
        x3 = self.FE2(x2)
        x3 = self.PCM3(x3)
        x4 = self.FE3(x3)
        x4 =self.PCM4(x4)
        en_out = [self.SPM1(x1), self.SPM2(x2), self.SPM3(x3), self.SPM4(x4)]
        return en_out

    def decoder(self, x):
        out = self.double2single1(x[-1])
        out = self.double2single2(self.sce1(torch.cat((out, x[-2]), dim=1)))
        out = self.double2single3(self.sce2(torch.cat((out, x[1]), dim=1)))
        out = self.sce3(torch.cat((out,x[0]), dim=1))

        return out
