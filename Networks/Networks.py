import torch
import torch.nn as nn


# 6 Level 3D UNet as proposed by Ronneberger with Isensee adaptions
class Adriana(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        #Layer 1
        self.conv1 = self.ConvBlock(in_channels, 32, kernel_size = 3, stride = 1, padding = 1)
        self.pool1 = self.MaxPool(kernel_size = 2, stride = 2)    
        
        #Layer 2
        self.conv2 = self.ConvBlock(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.pool2 = self.MaxPool(kernel_size = 2, stride = 2) 
        
        #Layer 3
        self.conv3 = self.ConvBlock(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.pool3 = self.MaxPool(kernel_size = 2, stride = 2)    
        
        #Layer 4
        self.conv4 = self.ConvBlock(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.pool4 = self.MaxPool(kernel_size = 2, stride = 2)

        #Layer 5
        self.conv5 = self.ConvBlock(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.pool5 = self.MaxPool(kernel_size = 1, stride = 1) 
        
        #Layer 6
        self.conv6 = self.ConvBlock(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.upconv5 = self.Upconv(256, 256, kernel_size = 1, stride = 1)
        
        #Layer 5 
        self.conv7 = self.ConvBlock(2*256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.upconv4 = self.Upconv(256, 256, kernel_size = 2 , stride = 2)
        
        #Layer 4
        self.conv8 = self.ConvBlock(2*256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.upconv3 = self.Upconv(256, 128, kernel_size = 2, stride = 2)
        
        #Layer 3
        self.conv9 = self.ConvBlock(2*128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.upconv2 = self.Upconv(128, 64, kernel_size = 2, stride = 2)
        
        #Layer 2
        self.conv10 = self.ConvBlock(2*64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.upconv1 = self.Upconv(64, 32, kernel_size = 2, stride = 2)
        
        #Layer 1
        self.conv11 = self.ConvBlock(2*32, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv12 = self.Conv(32, out_channels, kernel_size = 3 , stride = 1, padding = 1)
        
    
    def __call__(self, x):
        #Layer 1
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        #Layer 2
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)       
        
        #Layer 3
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        #Layer 4
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        #Layer 5
        conv5 = self.conv5(pool4)
        pool5 = self.pool5(conv5)
        
        #Layer 6
        conv6 = self.conv6(pool5)
        upconv5 = self.upconv5(conv6)
        
        #Layer 5
        conv7 = self.conv7(torch.cat([upconv5, conv5], 1))
        upconv4 = self.upconv4(conv7)
        
        #Layer 4
        conv8 = self.conv8(torch.cat([upconv4, conv4], 1))
        upconv3 = self.upconv3(conv8)
        
        #Layer 3
        conv9 = self.conv9(torch.cat([upconv3, conv3], 1))
        upconv2 = self.upconv2(conv9)
        
        #Layer 2
        conv10 = self.conv10(torch.cat([upconv2, conv2], 1))
        upconv1 = self.upconv1(conv10)
        
        #Layer 1
        conv11 = self.conv11(torch.cat([upconv1, conv1], 1))
        conv12 = self.conv12(conv11)
        
        return conv12
        

    def ConvBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                             nn.InstanceNorm2d(out_channels, affine=True),
                             nn.LeakyReLU(negative_slope=0.01, inplace=True),
                             nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                             nn.InstanceNorm2d(out_channels, affine=True),
                             nn.LeakyReLU(negative_slope=0.01, inplace=True)                             
                             ) 
        return conv
    
    
    def MaxPool(self, kernel_size, stride):
        maxPool = nn.MaxPool2d(kernel_size, stride)
        
        return maxPool
    
    
    def Upconv(self, in_channels, out_channels, kernel_size, stride):
        upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        
        return upconv
    
    
    def Conv(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        return conv     
    
    
    
     
class Candice(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = self.InitialBlock(in_channels, 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = self.ConvBlock(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv3 = self.ConvBlock(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.conv4 = self.ConvBlock(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.conv5 = self.ConvBlock(512, 512, kernel_size = 4, stride = 2, padding = 1)
        self.conv6 = self.FinalBlock(512, out_channels, kernel_size = 4, stride = 2, padding = 1)
        
        
    def __call__(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        
        return conv6
        
    
    def InitialBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode = 'reflect'),
                              nn.LeakyReLU(negative_slope=0.2)
                              ) 
        
        return conv
    
    
    def ConvBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode='reflect'),
                             nn.InstanceNorm2d(out_channels, affine=True),
                             nn.LeakyReLU(0.2)
                             )
        
        return conv
        
    def FinalBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode = 'reflect'))
        
        return conv
      
    
    
    
    
    
    
    
    
    
    
    
    