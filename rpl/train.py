import argparse
import logging
import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import optim
import random
import pickle
from sklearn.preprocessing import normalize
from PIL import Image

from datasets.cifar_dataset import CIFARDataset
from datasets.dataset import StandardDataset
from evaluate import evaluate_val
from models.backbone import encoder32
from models.backbone_resnet import encoder
from models.backbone_wide_resnet import wide_encoder
from penalties import compute_rpl_loss
from utils import count_parameters, setup_logger


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("img_base_path", type=str,
                    help="path to folder containing image data, i.e. /data/ or /share/nikola/export/image_datasets/")
    parser.add_argument("dataset_folder_path", type=str,
                    help="path to dataset folder.")
    parser.add_argument("dataset_folder", type=str,
                    help="name of folder where dataset details live.")
    parser.add_argument("checkpoint_folder_path", type=str,
                    help="folder where checkpoints will be saved")
    parser.add_argument("logging_folder_path", type=str,
                    help="folder where logfile will be saved")
    
    parser.add_argument("--n_epochs", type=int,
                    default=150, help="number of epochs to train")
    
    parser.add_argument("--gap", type=str,
                    default='TRUE', help="TRUE iff use global average pooling layer. Otherwise, use linear layer.")
    parser.add_argument("--lr_scheduler", type=str,
                    default='patience', help="patience, step.")
    
    parser.add_argument("--latent_size", type=int,
                    default=128, help="Dimension of embeddings.")
    parser.add_argument("--num_rp_per_cls", type=int,
                    default=1, help="Number of reciprocal points per class.")
    parser.add_argument("--lamb", type=float,
                    default=0.1, help="how much to weight the open-set regularization term in objective.")
    parser.add_argument("--gamma", type=float,
                    default=1, help="how much to weight the probability assignment.")

    parser.add_argument("--divide", type=str,
                    default='TRUE', help="TRUE or FALSE, as to whether or not to divide loss by latent_size for convergence.")
    
    parser.add_argument("--dataset", type=str,
                        default='TINY', help="CIFAR_PLUS, TINY, or IMAGENET, or LT")
    
    
    parser.add_argument("--batch_size", type=int,
                    default=64, help="size of a batch during training")
    parser.add_argument("--lr", type=float,
                    default=0.01, help="initial learning rate during training")
    parser.add_argument("--patience", type=int,
                    default=5, help="patience of lr scheduler")
    parser.add_argument("--img_size", type=int,
                    default=32, help="desired square image size.")
    parser.add_argument("--num_workers", type=int,
                    default=3, help="number of workers during training")
    parser.add_argument("--backbone_type", type=str,
                    default='OSCRI_encoder', help="architecture of backbone")
    
    parser.add_argument("--debug", type=str,
                    default='NO_DEBUG', help="this input is 'DEBUG' when experiment is for debugging")
    parser.add_argument("--msg", type=str,
                    default='NONE', help="if none, put NONE. else, place message.")

    args = parser.parse_args()

        
    if args.msg == 'NONE':
        CKPT_BASE_NAME = 'pat_' + str(args.patience) + '_div_' + args.divide + '_gap_' + args.gap + '_sched_' + args.lr_scheduler + '_latsize_' + str(args.latent_size) + '_numrp_' + str(args.num_rp_per_cls) + '_lambda_' + str(args.lamb) + '_gamma_' + str(args.gamma) + '_dataset_' + args.dataset_folder + '_' + str(args.lr).replace('0.','') + '_' + str(args.batch_size) + '_' + str(args.img_size) + '_' + args.backbone_type 
        LOGFILE_NAME = CKPT_BASE_NAME + '_logfile'
    
    else:
        CKPT_BASE_NAME = args.msg + '_pat_' + str(args.patience) + '_div_' + args.divide + '_gap_' + args.gap + '_sched_' + args.lr_scheduler + '_latsize_' + str(args.latent_size) + '_numrp_' + str(args.num_rp_per_cls) + '_lambda_' + str(args.lamb) + '_gamma_' + str(args.gamma) + '_dataset_' + args.dataset_folder + '_' + str(args.lr).replace('0.','') + '_' + str(args.batch_size) + '_' + str(args.img_size) + '_' + args.backbone_type  
        LOGFILE_NAME = CKPT_BASE_NAME + '_logfile'
    
    if args.debug == 'DEBUG':
        CKPT_BASE_NAME = 'debug_' + CKPT_BASE_NAME
        LOGFILE_NAME = 'debug_' + LOGFILE_NAME     
        
    os.mkdir(args.checkpoint_folder_path + CKPT_BASE_NAME)
    os.mkdir(args.checkpoint_folder_path + CKPT_BASE_NAME + '/' + 'backups')
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = setup_logger('logger', formatter, args.logging_folder_path + LOGFILE_NAME)
    

    if args.dataset == 'CIFAR_PLUS':
        with open(args.dataset_folder_path + args.dataset_folder + '/train_obj.pkl', 'rb') as f:
            train_obj = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/test_obj.pkl', 'rb') as f:
            test_obj = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/meta.pkl', 'rb') as f:
            meta_dict = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/img_to_idx.pkl', 'rb') as f:
            img_to_idx = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/test_img_to_idx.pkl', 'rb') as f:
            test_img_to_idx = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/label_to_idx.pkl', 'rb') as f:
            label_to_idx = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/idx_to_label.pkl', 'rb') as f:
            idx_to_label = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/closed_classes.pkl', 'rb') as f:
            closed_classes = pickle.load(f)

        with open(args.dataset_folder_path + args.dataset_folder + '/closed_train_img_list.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/closed_val_img_list.pkl', 'rb') as f:
            val_data = pickle.load(f)

        with open(args.dataset_folder_path + args.dataset_folder + '/closed_train_mean.pkl', 'rb') as f:
            train_mean = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/closed_train_std.pkl', 'rb') as f:
            train_std = pickle.load(f)
        img_normalize = transforms.Normalize(mean=train_mean,
                                                 std=train_std)
        num_classes = len(closed_classes)

        logging.info("Number of seen classes: " + str(num_classes))
        logging.info("Number of training images is: " + str(len(train_data)))
        logging.info("Number of validation images is: " + str(len(val_data)))

        dataset = CIFARDataset(train_data, train_obj, meta_dict, label_to_idx,
                              transforms.Compose([
                                            transforms.RandomResizedCrop(args.img_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            img_normalize,
                                            ]))
        val_dataset = CIFARDataset(val_data, train_obj, meta_dict, label_to_idx,
                              transforms.Compose([
                                            transforms.Resize((args.img_size,args.img_size)),
                                            transforms.ToTensor(),
                                            img_normalize,
                                            ]))
        
    else:
        with open(args.dataset_folder_path + args.dataset_folder + '/folder_to_name_map.pkl', 'rb') as f:
            folder_to_name = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/name_to_folder_map.pkl', 'rb') as f:
            name_to_folder = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/folder_to_idx.pkl', 'rb') as f:
            folder_to_idx = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/closed_classes.pkl', 'rb') as f:
            closed_classes = pickle.load(f)

        with open(args.dataset_folder_path + args.dataset_folder + '/closed_train_img_list.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/closed_val_img_list.pkl', 'rb') as f:
            val_data = pickle.load(f)

        with open(args.dataset_folder_path + args.dataset_folder + '/closed_train_mean.pkl', 'rb') as f:
            train_mean = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/closed_train_std.pkl', 'rb') as f:
            train_std = pickle.load(f)
        img_normalize = transforms.Normalize(mean=train_mean,
                                                 std=train_std)
        num_classes = len(closed_classes)

        logger.info("Number of seen classes: " + str(num_classes))
        logger.info("Number of training images is: " + str(len(train_data)))
        logger.info("Number of validation images is: " + str(len(val_data)))

        dataset = StandardDataset(train_data, folder_to_idx, folder_to_name,      
                                            transforms.Compose([
                                            transforms.RandomResizedCrop(args.img_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            img_normalize,
                                            ]), args.img_base_path)

        val_dataset = StandardDataset(val_data, folder_to_idx, folder_to_name,     
                                            transforms.Compose([
                                            transforms.Resize((args.img_size,args.img_size)),
                                            transforms.ToTensor(),
                                            img_normalize,
                                            ]), args.img_base_path)

    if args.backbone_type == 'OSCRI_encoder':
        model = encoder32(latent_size=args.latent_size, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls, gap=args.gap == 'TRUE')
        
    elif args.backbone_type == 'wide_resnet':
        model = wide_encoder(args.latent_size, 40, 4, 0, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls)
        
    elif args.backbone_type == 'resnet_50':
        backbone = models.resnet50(pretrained=False)
        VISUAL_FEATURES_DIM = 2048
        model = encoder(backbone, VISUAL_FEATURES_DIM, latent_size=args.latent_size, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls, gap=args.gap == 'TRUE')
    
    else:
        raise ValueError(args.backbone_type + ' is not supported.')

    model.cuda()
    
    num_params = count_parameters(model)
    logger.info("Number of model parameters: " + str(num_params))

    criterion = nn.CrossEntropyLoss(reduction='none')   
    
       
    if args.lr_scheduler == 'step':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
    elif args.lr_scheduler == 'patience':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True)
        
    else:
        raise ValueError(args.lr_scheduler + ' is not supported.')
        

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_n = len(dataset)
    best_used_running_loss = 100000000
    best_val_acc = 0.
    

    last_lr = False
    last_patience_counter = 0
    for epoch in range(0, args.n_epochs):

        logger.info("EPOCH " + str(epoch))
        running_loss = 0.0
        train_rpl_loss = 0.
        train_std_loss = 0.0
        train_correct = 0.0
        
        actual_lr = None
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
            if actual_lr is None:
                actual_lr = curr_lr
            else:
                if curr_lr != actual_lr:
                    raise ValueError("some param groups have different lr")
        logger.info("Learning rate: " + str(actual_lr))
        if actual_lr < 10 ** (-7):
            last_lr = True

        for i, data in enumerate(train_loader, 0):
            
            if args.debug == 'DEBUG':
                print('\nbatch ' + str(i))
            
            # get the inputs & combine positive and negatives together
            img = data['image']
            img = img.cuda()
            
            labels = data['label']
            labels = labels.cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model.forward(img)

            
            # Compute RPL loss
            loss, open_loss, closed_loss, logits = compute_rpl_loss(model, outputs, labels, criterion, args.lamb, args.gamma, args.divide == 'TRUE')
            train_rpl_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()

            # update loss for this epoch
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            max_probs, max_indices = torch.max(probs, 1)
                      
            train_correct += torch.sum(max_indices == labels).item()
        
            if args.debug == 'DEBUG':
                print("batch loss: " + str(loss.item()))
                print("rpl loss: " + str((closed_loss + open_loss).item()))
                print("number correct: " + str(torch.sum(max_indices == labels).item()))

        train_acc = train_correct/train_n
        logger.info("Training Accuracy is: " + str(train_acc))
        logger.info("Average overall training loss in epoch is: " + str(running_loss/train_n))

        model.eval()
        used_running_loss, used_val_acc = evaluate_val(model, criterion, val_loader, args.gamma, args.lamb, args.divide, logger)
        
        # Adjust learning rate
        if args.lr_scheduler == 'patience':
            scheduler.step(used_running_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()
        else:
            raise ValueError('scheduler did not update.')

        # case where only acc is top
        if used_val_acc > best_val_acc:
            
            curr_files = os.listdir(args.checkpoint_folder_path + CKPT_BASE_NAME + '/')
            models_to_move = []
            for file in curr_files:
                if file[-3:] == '.pt':
                    models_to_move.append(file)
            for mover in models_to_move:
                os.replace(args.checkpoint_folder_path + CKPT_BASE_NAME + '/' + mover, args.checkpoint_folder_path + CKPT_BASE_NAME + '/backups/' + mover)
            
            torch.save(model.state_dict(), args.checkpoint_folder_path + CKPT_BASE_NAME + '/' + str(epoch) + '.pt')
            
        elif args.lr_scheduler == 'patience' and last_lr:
            last_patience_counter += 1
            if last_patience_counter == 5:
                break
            
        if used_running_loss < best_used_running_loss:
            best_used_running_loss = used_running_loss
        if used_val_acc > best_val_acc:
            best_val_acc = used_val_acc
            if args.lr_scheduler == 'patience' and last_lr:
                last_patience_counter = 0
        model.train()
        
