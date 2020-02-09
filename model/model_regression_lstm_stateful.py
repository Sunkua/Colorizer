import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

from torch.nn import Tanh, ReLU, Sequential, BatchNorm2d, AdaptiveAvgPool2d, Conv2d, MaxPool2d, AvgPool2d, Dropout,ConvTranspose2d
class ColorCNN_REGRESSION_LSTM_STATEFUL(torch.nn.Module):

    def __init__(self):
        super(ColorCNN_REGRESSION_LSTM_STATEFUL, self).__init__()
        self.enc_conv1 = Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = torch.nn.Identity()
        self.enc_conv3 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.enc_conv5 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.bn2 = torch.nn.Identity()
        self.enc_conv6 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_conv7 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.enc_conv8 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.bn3 = torch.nn.Identity()
        self.enc_conv9 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_conv10 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)        

        self.bn4 = torch.nn.Identity()
        self.dec_conv1 = Conv2d(1024, 512, kernel_size=3, stride=1,dilation=1, padding=1)
        self.dec_conv1_1 = Conv2d(512, 512, kernel_size=3, stride=1,dilation=2, padding=2)
        self.dec_conv1_2 = Conv2d(512, 512, kernel_size=3, stride=1,dilation=2, padding=2)

        self.bn4_1 = torch.nn.Identity()
        self.dec_conv1_3 = Conv2d(512, 512, kernel_size=3, stride=1,dilation=2, padding=2)
        self.dec_conv2 = Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        
        self.bn5 = torch.nn.Identity()
        self.dec_conv3 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.dec_conv4 = Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        
        self.bn6 = torch.nn.Identity()
        self.dec_conv5 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.dec_conv6 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.bn7 = torch.nn.Identity()
        self.dec_conv7 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dec_conv8 = Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        
        self.bn8 = torch.nn.Identity()
        self.dec_conv9 = Conv2d(32,2, kernel_size=3, stride=1, padding=1)

        self.dec_conv_b = Conv2d(32, 15, kernel_size=3, stride=1, padding=1)
        self.dec_conv_a = Conv2d(32, 15, kernel_size=3, stride=1, padding=1)
        
        self.lstm = torch.nn.LSTM(512, 512, 2, batch_first=True)
        self.h_n = None
        self.c_n = None
        
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.constant(m.bias, 0.01)
        

        torch.nn.init.xavier_uniform_(self.dec_conv9.weight)
        torch.nn.init.constant_(self.dec_conv9.bias,0.01)

    def reset_hidden_states(self):
        self.h_n = None
        self.c_n = None    
    

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

        lstm_in = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        lstm_in = lstm_in.permute(2,0,1)
        if self.h_n is None:
            lstm_out, (hidden_states, cell_states) = self.lstm(lstm_in)
            self.h_n = hidden_states.detach()
            self.c_n = cell_states.detach()
        else:
            lstm_out, (hidden_states, cell_states) = self.lstm(lstm_in, (self.h_n, self.c_n))
            self.h_n = hidden_states.detach()
            self.c_n = cell_states.detach()
        lstm_out = lstm_out.permute(1,2,0)
        lstm_out = lstm_out.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        x = torch.cat((x, lstm_out), dim=1)

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


        return x