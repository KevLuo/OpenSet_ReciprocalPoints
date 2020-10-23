import math
import numpy as np
import torch


def compute_rpl_logits(model, outputs, gamma):
    
    # Calculate L2 distance to reciprocal points
    raw_dist_to_rp = torch.cdist(outputs, model.reciprocal_points, p=2)

    # expects each distance to be squared (squared euclidean distance)
    dist_to_rp = raw_dist_to_rp ** 2
    dist_to_rp = torch.reshape(dist_to_rp, (dist_to_rp.shape[0], model.num_classes, model.num_rp_per_cls))
    # output should be batch_size x num_classes
    logits = gamma * torch.mean(dist_to_rp, dim=2)
    
    return logits, dist_to_rp
    

def compute_rpl_loss(model, outputs, labels, criterion, lamb, gamma, divide):
    assert type(divide) is bool    

    open_loss = torch.tensor(0.).cuda()
    ### BEGIN: Compute closed loss and open loss ###

    logits, dist_to_rp = compute_rpl_logits(model, outputs, gamma)
    
    for i in range(0, labels.shape[0]):
        curr_label = labels[i].item()
        if divide:
            dist_to_cls_rp_vector = dist_to_rp[i, curr_label]/model.latent_size
        else:
            dist_to_cls_rp_vector = dist_to_rp[i, curr_label]
        open_loss += torch.mean((dist_to_cls_rp_vector - model.R[curr_label]) ** 2)

    # this criterion is just cross entropy
    closed_loss = torch.sum(criterion(logits, labels)) 
    
    open_loss = lamb * open_loss
    ### END: Compute closed loss and open loss ###
    loss = closed_loss + open_loss
    
    return loss, open_loss, closed_loss, logits