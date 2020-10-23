import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class encoder32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, num_rp_per_cls=8, gap=False):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.num_rp_per_cls = num_rp_per_cls
        self.latent_size = latent_size
        self.gap = gap
        
        if self.gap:
            print("model using global average pooling layer on top of encoder")
        else:
            print("model using fc layer on top of encoder")
        
        if self.gap:
            assert self.latent_size == 128
        
        self.reciprocal_points = nn.Parameter(torch.zeros((self.num_classes * self.num_rp_per_cls, self.latent_size)))
        nn.init.normal_(self.reciprocal_points)
        self.R = nn.Parameter(torch.zeros((self.num_classes, )))

        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv10 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        if not self.gap:
            self.fc1 = nn.Linear(128*2*2, self.latent_size)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.dr4 = nn.Dropout2d(0.2)


    def forward(self, x):
        batch_size = x.shape[0]

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr4(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)
        
        if self.gap:
            embeddings = self.forward_gap(x)  
        else:
            embeddings =  self.forward_linear(x)
            
        return embeddings
             
        
    def forward_gap(self, x):
        """ Call this method if encoder has global average pooling on top. """
        # output of avg pool is expected to be 128-d
        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(x.shape[0], -1)
        assert x.shape[1] == 128
        return x
    
    def forward_linear(self, x):
        """ Call this method if encoder has linear layer on top. """
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
    
