import torch
import torch.nn.functional as F
import pretrainedmodels
from torch.nn import Tanh, ReLU, Sequential, Conv2d, MaxPool2d, Dropout,ConvTranspose2d
class ColorCNN(torch.nn.Module):

    def __init__(self):
        super(ColorCNN, self).__init__()

        self.enc_conv1 = Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.enc_conv3 = Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.enc_conv5 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.enc_conv6 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_conv7 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.enc_conv8 = Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.dec_conv1 = Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.dec_conv2 = Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.dec_conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.dec_conv4 = ConvTranspose2d(128,128, 3, stride=2, padding=1,output_padding=1)
        self.dec_conv5 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dec_conv6 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dec_conv7 = ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv8 = Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dec_conv9 = Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.dec_conv10 = Conv2d(32,2, kernel_size=3, stride=1, padding=1)
        self.dec_conv11 = ConvTranspose2d(2, 2, 3, stride=2, padding=1, output_padding=1)

        #self.lstm = torch.nn.LSTM(256, 256, num_layers=2,batch_first=True)
        #self.gru = torch.nn.GRU(256,256,num_layers=2)
        self.linear = torch.nn.Linear(256,256)
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
        self.h_n = None
        self.c_n = None

    def forward(self, x, first_it):
        '''
        if self.h_n is not None:
            for i in range(first_it.shape[0]):
                if first_it[i] > 0:
                    self.h_n[:,i] = 0
                    #self.c_n[:,i] = 0
        '''
        bs = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2], x.shape[3], x.shape[4])
        x = x.permute(0,3,1,2)
        ds0_dims = x.size()

        # encoding 
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

       
        # lstm
        a = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        a = a.reshape(seq_len, a.shape[1], a.shape[2] * bs)
        a = a.permute(0,2,1)
        '''
        if self.h_n is None:
            #a, (h_n, c_n) = self.lstm(a)
            #self.h_n = h_n.detach()
            #self.c_n = c_n.detach()
            a, h_n = self.gru(a)
            self.h_n = h_n.detach()
        else:
            #a, (h_n, c_n) = self.lstm(a, (self.h_n, self.c_n))
            #self.h_n = h_n.detach()
            #self.c_n = c_n.detach()
            a, h_n = self.gru(a,self.h_n)
            self.h_n = h_n.detach()
        a = torch.relu(self.linear(a))
        '''
        a = self.transformer_encoder(a)
        a = a.permute(0,2,1)
        a = a.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        t = torch.cat((a,x), dim=1)
        
        # decoding
        t = torch.relu(self.dec_conv1(t))
        t = torch.relu(self.dec_conv2(t))
        t = torch.relu(self.dec_conv3(t))
        t = torch.relu(self.dec_conv4(t, output_size=ds2_dims))
        t = torch.relu(self.dec_conv5(t))
        t = torch.relu(self.dec_conv6(t))
        t = torch.relu(self.dec_conv7(t, output_size=ds1_dims))
        t = torch.relu(self.dec_conv8(t))
        t = torch.relu(self.dec_conv9(t))
        t = torch.relu(self.dec_conv10(t))
        t = torch.relu(self.dec_conv11(t, output_size=ds0_dims))

        t = t.reshape(bs, t.shape[0] // bs, t.shape[1], t.shape[2], t.shape[3])

        return t
