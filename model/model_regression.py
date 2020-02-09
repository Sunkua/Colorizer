import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

from torch.nn import Tanh, ReLU, Sequential, BatchNorm2d, AdaptiveAvgPool2d, Conv2d, MaxPool2d, AvgPool2d, Dropout,ConvTranspose2d
class ColorCNN_REGRESSION(torch.nn.Module):

    def __init__(self):
        super(ColorCNN_REGRESSION, self).__init__()
        self.enc_conv1 = Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = BatchNorm2d(128)
        self.enc_conv3 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.enc_conv5 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.bn2 = BatchNorm2d(256)
        self.enc_conv6 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_conv7 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.enc_conv8 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.bn3 = BatchNorm2d(512)
        self.enc_conv9 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_conv10 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)        

        self.bn4 = BatchNorm2d(512)
        self.dec_conv1 = Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2)
        self.dec_conv1_1 = Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2)
        self.dec_conv1_2 = Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2)

        self.bn4_1 = torch.nn.BatchNorm2d(512)
        self.dec_conv1_3 = Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2)
        self.dec_conv2 = Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.dec_conv3 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.dec_conv4 = Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        
        self.bn6 = BatchNorm2d(128)
        self.dec_conv5 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.dec_conv6 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.bn7 = BatchNorm2d(64)
        self.dec_conv7 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dec_conv8 = Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        
        self.bn8 = BatchNorm2d(32)
        self.dec_conv9 = Conv2d(32, 2, kernel_size=3, stride=1, padding=1)


        
        
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.constant(m.bias, 0.01)
        

        torch.nn.init.xavier_uniform_(self.dec_conv9.weight)
        torch.nn.init.constant_(self.dec_conv9.bias,0.01)
        

    def forward(self, frames):
        
        # encoding 
        x = frames
        ds0_dims = frames.size()
        x = torch.relu(self.enc_conv1(x))
        x = torch.relu(self.enc_conv2(x))

        ds1_dims = x.size()
        x = self.bn1(torch.relu(self.enc_conv3(x)))
        x = torch.relu(self.enc_conv4(x))
        x = torch.relu(self.enc_conv5(x))

        ds2_dims = x.size()
        x = self.bn2(torch.relu(self.enc_conv6(x)))
        x = torch.relu(self.enc_conv7(x))
        x = torch.relu(self.enc_conv8(x))

        ds3_dims = x.size()
        x = self.bn3(torch.relu(self.enc_conv9(x)))
        x = torch.relu(self.enc_conv10(x))


        x = self.bn4(torch.relu(self.dec_conv1(x)))
        x = torch.relu(self.dec_conv1_1(x))
        x = torch.relu(self.dec_conv1_2(x))

        x = self.bn4_1(torch.relu(self.dec_conv1_3(x)))

        x = torch.nn.functional.interpolate(x, (ds3_dims[2], ds3_dims[3]))
        x = torch.relu(self.dec_conv2(x))

        x = self.bn5(torch.relu(self.dec_conv3(x)))
        x = torch.nn.functional.interpolate(x, (ds2_dims[2], ds2_dims[3]))
        x = torch.relu(self.dec_conv4(x))

        x = self.bn6(torch.relu(self.dec_conv5(x)))
        x = torch.nn.functional.interpolate(x, (ds1_dims[2], ds1_dims[3]))
        x = torch.relu(self.dec_conv6(x))

        x = self.bn7(torch.relu(self.dec_conv7(x)))
        x = torch.nn.functional.interpolate(x, (ds0_dims[2], ds0_dims[3]))
        x = torch.relu(self.dec_conv8(x))

        x = torch.tanh(self.dec_conv9(x))
        #a_out = torch.softmax(torch.relu(self.dec_conv_a(x)), dim=1)
        #b_out = torch.softmax(torch.relu(self.dec_conv_b(x)), dim=1)

        return x