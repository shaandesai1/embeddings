"""
This implementation is based on following code:
https://github.com/milesial/Pytorch-UNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class triple_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(triple_conv, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 3, padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_ch, out_ch, 3, padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_ch, out_ch, 3, padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True)

          )

    def forward(self, x):
        x = self.conv(x)
        return x




class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class down2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down2, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            triple_conv(in_ch, out_ch)
            )
    
    def forward(self, x):
        x = self.mpconv(x)
        return x



class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.bilin = bilinear
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.functional.interpolate
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilin:
            x1 = self.up(x1,scale_factor=2,mode='bilinear')
        else:
            x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up2(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up2, self).__init__()
        self.bilin = bilinear
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.functional.interpolate
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        
        self.conv = triple_conv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        if self.bilin:
            x1 = self.up(x1,scale_factor=2,mode='bilinear')
        else:
            x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x








class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = nn.Sequential(
#            nn.Conv2d(in_ch, in_ch//2, 1),
#            nn.BatchNorm2d(in_ch//2),
#            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down2(128, 256)
        self.down3 = down2(256, 512)
        self.down4 = down2(512, 512)
        self.up1 = up2(1024, 256)
        self.up2 = up2(512, 128)
        self.up3 = up2(256, 64)
        self.up4 = up(128, 64)
        self.sem_out = outconv(64, 41)
        self.ins_out = outconv(64, 32)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        sems = self.sem_out(x)
        ins = self.ins_out(x)
        return  ins,sems