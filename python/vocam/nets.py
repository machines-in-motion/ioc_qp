## This file contains the Neural Network architectures that are used in the demos
## Author : Avadesh Meduri and Huaijiang Zhu
## Date : 3/05/2022

import torch
import torch.nn as nn
from torch.nn import functional as F



class Net(torch.nn.Module):

    def __init__(self, inp_size, out_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(inp_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.out = torch.nn.Linear(256, out_size)

    def forward(self, x):
       
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.out(x)
        return x


class C_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(4, 64, 3)
        self.conv12 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv21 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)

        self.conv31 = nn.Conv2d(128, 256, 3)
        self.conv32 = nn.Conv2d(256, 256, 3)
        self.conv33 = nn.Conv2d(256, 256, 3)

        
        self.conv41 = nn.Conv2d(256, 512, 3)
        self.conv42 = nn.Conv2d(512, 512, 3)
        
        self.fc1 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.pool(F.relu(self.conv12(x)))
        
        x = F.relu(self.conv21(x))
        x = self.pool(F.relu(self.conv22(x)))
        
        x = self.pool(F.relu(self.conv31(x)))
        x = self.pool(F.relu(self.conv32(x)))
        
        x = self.pool(F.relu(self.conv41(x)))
        x = F.relu(self.conv42(x))
            
        x = torch.flatten(x, 1) # flatten all dimensions except batch
#         print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


class C_Net_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(4, 64, 3)
        self.conv12 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv21 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)

        self.conv31 = nn.Conv2d(128, 256, 3)
        self.conv32 = nn.Conv2d(256, 256, 3)
        self.conv33 = nn.Conv2d(256, 256, 3)

        
        self.conv41 = nn.Conv2d(256, 512, 3)
        self.conv42 = nn.Conv2d(512, 512, 3)
        
        self.fc1 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.pool(F.relu(self.conv12(x)))
        
        x = F.relu(self.conv21(x))
        x = self.pool(F.relu(self.conv22(x)))
        
        x = self.pool(F.relu(self.conv31(x)))
        x = self.pool(F.relu(self.conv32(x)))
        
        x = self.pool(F.relu(self.conv41(x)))
        x = F.relu(self.conv42(x))
            
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        enc = x
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x, enc