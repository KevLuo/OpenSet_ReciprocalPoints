import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class encoder(nn.Module):
    def __init__(self, backbone, backbone_output_size, latent_size=100, num_classes=2, num_rp_per_cls=8, gap=False):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.num_rp_per_cls = num_rp_per_cls
        self.latent_size = latent_size
        self.backbone_output_size = backbone_output_size
        self.gap = gap
        
        if self.gap:
            print("model using global average pooling layer on top of encoder")
            assert self.latent_size == 2048
        else:
            print("model using fc layer on top of encoder")
        
        self.reciprocal_points = nn.Parameter(torch.zeros((self.num_classes * self.num_rp_per_cls, self.latent_size)))
        nn.init.normal_(self.reciprocal_points)
        self.R = nn.Parameter(torch.zeros((self.num_classes, )))

        # Resnet Backbone [includes avg pooling layer, takes off last FC layer]
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # define a linear layer if not using GAP
        if not self.gap:
            self.out = nn.Linear(self.backbone_output_size, self.latent_size)
        
    def forward(self, inputs):
        assert type(resnet_features) is str
        
        # put images through ResNet backbone
        img_features = self.features(inputs)
        img_features = torch.flatten(img_features, start_dim=1)
        
        if self.gap:
            embeddings = img_features         
        else:
            embeddings = self.out(img_features)
    
        return embeddings
        

    