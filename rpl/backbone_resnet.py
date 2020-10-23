import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class encoder(nn.Module):
    def __init__(self, backbone, backbone_output_size, latent_size=100, num_classes=2, num_rp_per_cls=8, dropout_rate=0, gap=False):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.num_rp_per_cls = num_rp_per_cls
        self.latent_size = latent_size
        self.backbone_output_size = backbone_output_size
        self.dropout_rate = dropout_rate
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
        
        if self.dropout_rate > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout_rate)
            print("model using dropout of " + str(self.dropout_rate))
        
    def forward(self, inputs, resnet_features='None'):
        assert type(resnet_features) is str
        
        # put images through ResNet backbone
        img_features = self.features(inputs)
        img_features = torch.flatten(img_features, start_dim=1)
        
        if self.dropout_rate > 0:
            ready_img_features = self.dropout_layer(img_features)
        else:
            ready_img_features = img_features
        
        if self.gap:
            embeddings = ready_img_features         
        else:
            embeddings = self.out(ready_img_features)
    
        if resnet_features == 'None':
            return embeddings
        else:
            if resnet_features == 'last':
                features = embeddings
            elif resnet_features == '2_to_last':
                if self.gap:
                    raise ValueError("Using GAP, can't get second to last yet.")
                else:
                    features = img_features
            else:
                raise ValueError(resnet_features + ' is not supported.')
            
            # return the L2-normalized features
            small_eps = torch.full((features.shape[0], 1), 0.000001).cuda()
            features = features / (torch.norm(features, p=2, dim=1, keepdim=True) + small_eps)
            return embeddings, features

    def produce_embeddings(self, inputs, features_type='last'):
        if features_type == 'last':
            return self.forward(inputs)
        elif features_type == '2_to_last':
            # put images through ResNet backbone
            img_features = self.features(inputs)
            flattened_img_features = torch.flatten(img_features, start_dim=1)

            if self.normalize:
                small_eps = torch.full((inputs.shape[0], 1), 0.000001).cuda()
                flattened_img_features = flattened_img_features / (torch.norm(flattened_img_features, p=2, dim=1, keepdim=True) + small_eps)

            return flattened_img_features
        else:
            raise ValueError(features_type + ' is not supported.')

    