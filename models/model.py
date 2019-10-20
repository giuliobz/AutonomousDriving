import torch
import torch.nn             as nn
import torchvision.models   as models
from torch.nn.utils.rnn     import pack_padded_sequence
from torch.autograd         import Variable
import numpy                as np


########################################################################################################################
# LAMBDA LAYER
########################################################################################################################

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


########################################################################################################################
# NETWORK
########################################################################################################################

class NET(nn.Module):
    """ Define our Network """

    def __init__(self, len_seq, batch_size, hidden_dimension, num_layers, in_channels):
        super(NET, self).__init__()

        self.hidden_dimension = hidden_dimension
        self.batch_size = batch_size
        self.len_seq = len_seq
        self.num_layers = num_layers

        # DEFINE THE ENCODER

        self.Lambda = LambdaLayer(lambda x: x / 127.5 - 1.0)

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 24, kernel_size=(5, 5), padding=5, stride=(2, 2)), nn.ELU())
                                    
        self.layer2 = nn.Sequential(nn.Conv2d(24, 36, kernel_size=(5, 5), padding=5, stride=(2, 2)), nn.ELU())
        self.max1 = nn.MaxPool2d(5, stride=2)

        self.layer3 = nn.Sequential(nn.Conv2d(36, 48, kernel_size=(5, 5), padding=5, stride=(2, 2)), nn.ELU())        
        self.max2 = nn.MaxPool2d(5, stride=2)

        self.layer4 = nn.Sequential(nn.Conv2d(48, 64, kernel_size=(3, 3), padding=3), nn.ELU())
        self.max3 = nn.MaxPool2d(3, stride=2)

        self.layer5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3, 3), padding=3), nn.ELU())

        self.Dropout_1 = nn.Dropout(0.4)

        self.layer6 = nn.Linear(in_features=3584, out_features=self.hidden_dimension)

        self.LSTM = nn.LSTM(input_size=self.hidden_dimension, hidden_size=self.hidden_dimension, num_layers=self.num_layers, batch_first=True, dropout=0.4) # 

        self.Dropout_2 = nn.Dropout(0.4)
        
        self.Linear = nn.Linear(in_features=self.hidden_dimension, out_features=2)
    
    def forward(self, image, device):
                                 
        """Encoding phase"""
        out = self.Lambda(image)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.max1(out)
        out = self.layer3(out)
        out = self.max2(out)
        out = self.layer4(out)
        out = self.max3(out)
        out = self.layer5(out)
        out = self.Dropout_1(out)
        features = out.reshape(out.size(0), -1)
        features = self.layer6(features)

        """Dencoding phase"""
        output_seq = torch.empty(   (self.len_seq, self.batch_size, 2)  )
        h, c = self.init_hidden(features, device)

        for t in range(self.len_seq):
            if t == 0:
                out, (hidden, cell) = self.LSTM(features.unsqueeze(1), (h, c))
            else:
                out, (hidden, cell) = self.LSTM(out, (hidden, cell))
            
            out = self.Dropout_2(out)
            tmp = self.Linear(out)
            
            output_seq[t] = tmp.view(self.batch_size, -1)
        
        output_seq = output_seq.transpose(1, 0)
           
        return output_seq

    def init_hidden(self, features, device):
        """ Initialize hidden state """

        h = torch.zeros((self.num_layers, self.batch_size, self.hidden_dimension)).to(device)
        c = torch.zeros((self.num_layers, self.batch_size, self.hidden_dimension)).to(device)

        # init hidden and cell state with fc7 vector
        for it in range(self.batch_size):
            h[:, it, :] = c[:, it, :] = features[it, :]

        return h, c

