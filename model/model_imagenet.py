import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

from torch.nn import Tanh, ReLU, Sequential, BatchNorm2d, AdaptiveAvgPool2d, Conv2d, MaxPool2d, AvgPool2d, Dropout,ConvTranspose2d
class ColorCNN(torch.nn.Module):

    def __init__(self):
        super(ColorCNN, self).__init__()
        self.enc_conv1 = Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.enc_conv3 = Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.enc_conv5 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.enc_conv6 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_conv7 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.enc_conv8 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        #self.enc_pool = AdaptiveAvgPool2d((1,1))

        #self.lstm = torch.nn.LSTM(512, 512, num_layers=2,batch_first=True)
        #self.linear = torch.nn.Linear(512,512)

        self.dec_conv1 = Conv2d(2304, 256, kernel_size=1, stride=1, padding=0)
        self.dec_conv2 = Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.dec_conv3 = Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.dec_conv4 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dec_conv5 = Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dec_conv6 = Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        
        
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.constant(m.bias, 0.01)
        
        torch.nn.init.xavier_uniform_(self.dec_conv6.weight)
        torch.nn.init.constant_(self.dec_conv6.bias,0.01)

        model = models.resnext50_32x4d(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1]) 
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        

    def forward(self, frames, features):
        ds0_dims = frames.size()
        # encoding 
        x = frames
        x = torch.relu(self.enc_conv1(x))
        ds1_dims = x.size()
        x = torch.relu(self.enc_conv2(x))
        x = torch.relu(self.enc_conv3(x))
        ds2_dims = x.size()
        x = torch.relu(self.enc_conv4(x))
        x = torch.relu(self.enc_conv5(x))
        x = torch.relu(self.enc_conv6(x))
        ds3_dims = x.size()
        x = torch.relu(self.enc_conv7(x))
        x = torch.relu(self.enc_conv8(x))

        
        # Pooling and Lstm
        #pooled_out = self.enc_pool(x)
        '''
        lstm_in = x.reshape(x.shape[0], x.shape[1], (x.shape[2]*x.shape[3]))
        lstm_in = lstm_in.permute(2,0,1)

        #pooled_out = pooled_out.squeeze()
        if len(lstm_in.shape) == 2:
            lstm_in = lstm_in.unsqueeze(dim=0)
        lstm_out = self.lstm(lstm_in)[0].squeeze()
        lstm_out = torch.relu(self.linear(lstm_out))
        lstm_out = lstm_out.reshape((x.shape[2], x.shape[3], lstm_out.shape[2], x.shape[0]))
        lstm_out = lstm_out.permute(3,2,0,1)
        #lstm_out = lstm_out.reshape((x.shape[0], x.shape[1], 1, 1))
        #lstm_out = torch.nn.functional.interpolate(lstm_out,(x.shape[2], x.shape[3]))
        '''

        # resnet
        '''
        scale_factor = 224 / resnet_in.shape[2] 
        resnet_in = torch.nn.functional.interpolate(resnet_in, scale_factor=scale_factor)
        resnet_in = resnet_in.repeat((1,3,1,1))
        deeplab_out = self.model(resnet_in)
        deeplab_out = torch.nn.functional.interpolate(deeplab_out,(x.shape[2], x.shape[3]))
        '''
        deeplab_out = torch.nn.functional.interpolate(features,(x.shape[2], x.shape[3]))
        t = torch.cat((x,deeplab_out),dim=1)      

        # decoding
        #t = torch.cat((x, lstm_out), dim=1)
        t = torch.relu(self.dec_conv1(t))
        t = torch.relu(self.dec_conv2(t))
        t = torch.nn.functional.interpolate(t, (ds2_dims[2], ds2_dims[3]))

        t = torch.relu(self.dec_conv3(t))
        t = torch.relu(self.dec_conv4(t))
        t = torch.nn.functional.interpolate(t, (ds1_dims[2], ds1_dims[3]))

        t = torch.relu(self.dec_conv5(t))
        t = torch.sigmoid(self.dec_conv6(t))
        t = torch.nn.functional.interpolate(t, (ds0_dims[2], ds0_dims[3]))

        return t
