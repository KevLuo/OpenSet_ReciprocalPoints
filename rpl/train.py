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
from nltk.corpus import wordnet as wn
import nltk
from PIL import Image
from zsh.baseline.cifar_dataset import CIFARDataset
from zsh.baseline.dataset import StandardDataset
from zsh.rpl.backbone import encoder32
from zsh.rpl.backbone_resnet import encoder
from penalties import compute_rpl_loss
from zsh.rpl.backbone_wide_resnet import wide_encoder
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def evaluate_val(model, val_loader, gamma, lamb, desired_features, divide):

    with torch.no_grad():
        running_loss = 0.0
        normal_correct = 0.
        used_correct = 0.
        normal_total = 0.0
        used_total = 0.0 
        normal_running_loss = 0.
        used_running_loss = 0.
        val_rpl_loss = 0.
        
        logging.info("beginning validation")

        
        for i, data in enumerate(val_loader, 0):
            
            # get the inputs & combine positive and negatives together
            img = data['image']
            img = img.cuda()
            
            labels = data['label']
            labels = labels.cuda()
            
            if desired_features == 'None':
                outputs = model.forward(img)
            else:
                outputs, features = model.forward(img, desired_features)
            
            # Compute RPL loss
            loss, open_loss, closed_loss, logits = compute_rpl_loss(model, outputs, labels, criterion, lamb, gamma, divide == 'TRUE')
            val_rpl_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            max_probs, max_indices = torch.max(probs, 1)
            
            used_correct += torch.sum(max_indices == labels).item()
            used_total += probs.shape[0]
            used_running_loss += loss.item()

        used_val_acc = used_correct/(used_total)
        logging.info("Used Validation Accuracy is : " + str(used_val_acc))
        logging.info("Used Average validation loss is: " + str(used_running_loss/used_total))
        logging.info("finished validation")
    
        return used_running_loss, used_val_acc
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("n_epochs", type=int,
                    help="number of epochs to train")
    
    parser.add_argument("gap", type=str,
                    help="TRUE iff use global average pooling layer. Otherwise, use linear layer.")
    parser.add_argument("desired_features", type=str,
                    help="None means no features desired. Other examples include last, 2_to_last.")
    parser.add_argument("lr_scheduler", type=str,
                    help="patience, warmup, step.")
    
    parser.add_argument("latent_size", type=int,
                    help="Dimension of embeddings.")
    parser.add_argument("num_rp_per_cls", type=int,
                    help="Number of reciprocal points per class.")
    parser.add_argument("lamb", type=float,
                    help="how much to weight the open-set regularization term in objective.")
    parser.add_argument("gamma", type=float,
                    help="how much to weight the probability assignment.")

    parser.add_argument("divide", type=str,
                    help="TRUE or FALSE, as to whether or not to divide loss by latent_size for convergence.")
    
    
    parser.add_argument("dataset", type=str,
                        help="CIFAR_PLUS, TINY, or IMAGENET, or LT")
    parser.add_argument("img_base_path", type=str,
                    help="path to folder containing image data, i.e. /data/ or /share/nikola/export/image_datasets/")
    parser.add_argument("dataset_folder_path", type=str,
                    help="path to dataset folder.")
    parser.add_argument("dataset_folder", type=str,
                    help="name of folder where dataset details live.")
    
    parser.add_argument("batch_size", type=int,
                    help="size of a batch during training")
    parser.add_argument("lr", type=float,
                    help="initial learning rate during training")
    parser.add_argument("patience", type=int,
                    help="patience of lr scheduler")
    parser.add_argument("img_size", type=int,
                    help="desired square image size.")
    parser.add_argument("num_workers", type=int,
                    help="number of workers during training")
    parser.add_argument("backbone_type", type=str,
                    help="architecture of backbone")
    
    parser.add_argument("checkpoint_folder_path", type=str,
                    help="folder where checkpoints will be saved")
    parser.add_argument("logging_folder_path", type=str,
                    help="folder where logfile will be saved")
    parser.add_argument("debug", type=str,
                    help="this input is 'DEBUG' when experiment is for debugging")
    parser.add_argument("msg", type=str,
                    help="if none, put NONE. else, place message.")

    args = parser.parse_args()

        
    if args.msg == 'NONE':
        CKPT_BASE_NAME = 'pat_' + str(args.patience) + '_div_' + args.divide + '_gap_' + args.gap + '_sched_' + args.lr_scheduler + '_latsize_' + str(args.latent_size) + '_numrp_' + str(args.num_rp_per_cls) + '_lambda_' + str(args.lamb) + '_gamma_' + str(args.gamma) + '_df_' + args.desired_features + '_dataset_' + args.dataset_folder + '_' + str(args.lr).replace('0.','') + '_' + str(args.batch_size) + '_' + str(args.img_size) + '_' + args.backbone_type 
        LOGFILE_NAME = CKPT_BASE_NAME + '_logfile'
    
    else:
        CKPT_BASE_NAME = args.msg + '_pat_' + str(args.patience) + '_div_' + args.divide + '_gap_' + args.gap + '_sched_' + args.lr_scheduler + '_latsize_' + str(args.latent_size) + '_numrp_' + str(args.num_rp_per_cls) + '_lambda_' + str(args.lamb) + '_gamma_' + str(args.gamma) + '_df_' + args.desired_features + '_dataset_' + args.dataset_folder + '_' + str(args.lr).replace('0.','') + '_' + str(args.batch_size) + '_' + str(args.img_size) + '_' + args.backbone_type  
        LOGFILE_NAME = CKPT_BASE_NAME + '_logfile'
    
    if args.debug == 'DEBUG':
        CKPT_BASE_NAME = 'debug_' + CKPT_BASE_NAME
        LOGFILE_NAME = 'debug_' + LOGFILE_NAME     
        
    os.mkdir(args.checkpoint_folder_path + CKPT_BASE_NAME)
    os.mkdir(args.checkpoint_folder_path + CKPT_BASE_NAME + '/' + 'backups')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO, filename=args.logging_folder_path + LOGFILE_NAME, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    

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

        logging.info("Number of seen classes: " + str(num_classes))
        logging.info("Number of training images is: " + str(len(train_data)))
        logging.info("Number of validation images is: " + str(len(val_data)))

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
        model = encoder32(latent_size=args.latent_size, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls, dropout_rate=args.dropout_rate, gap=args.gap == 'TRUE')
        
    elif args.backbone_type == 'wide_resnet':
        model = wide_encoder(args.latent_size, 40, 4, 0, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls, spec_dropout_rate=args.dropout_rate)
        
    elif args.backbone_type == 'resnet_50':
        backbone = models.resnet50(pretrained=False)
        VISUAL_FEATURES_DIM = 2048
        model = encoder(backbone, VISUAL_FEATURES_DIM, latent_size=args.latent_size, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls, dropout_rate=args.dropout_rate, gap=args.gap == 'TRUE')
    
    else:
        raise ValueError(args.backbone_type + ' is not supported.')

    model.cuda()
    
    num_params = count_parameters(model)
    logging.info("Number of model parameters: " + str(num_params))

    criterion = nn.CrossEntropyLoss(reduction='none')   
    
    if args.lr_scheduler == 'warmup':
        warmup_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr * (1 / (10**5)), weight_decay = 2/(10**4))
        warmup_scheduler = torch.optim.lr_scheduler.StepLR(warmup_optimizer, 1, gamma=10)
        final_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay = 2/(10**4))
        final_scheduler = torch.optim.lr_scheduler.MultiStepLR(final_optimizer, [60, 80])
        
    elif args.lr_scheduler == 'step':
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

        logging.info("EPOCH " + str(epoch))
        running_loss = 0.0
        train_frob_loss = 0.0
        train_norm_loss = 0.0
        train_num_norm_comps = 0.0
        train_rpl_loss = 0.
        train_std_loss = 0.0
        train_correct = 0.0
        
        actual_lr = None
        if args.lr_scheduler == 'warmup':
            if epoch <= 4:
                for param_group in warmup_optimizer.param_groups:
                    curr_lr = param_group['lr']
                    if actual_lr is None:
                        actual_lr = curr_lr
                    else:
                        if curr_lr != actual_lr:
                            raise ValueError("some param groups have different lr")
                logging.info("Learning rate: " + str(actual_lr))
            else:
                for param_group in final_optimizer.param_groups:
                    curr_lr = param_group['lr']
                    if actual_lr is None:
                        actual_lr = curr_lr
                    else:
                        if curr_lr != actual_lr:
                            raise ValueError("some param groups have different lr")
                logging.info("Learning rate: " + str(actual_lr))
                
        elif args.lr_scheduler == 'patience':
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                if actual_lr is None:
                    actual_lr = curr_lr
                else:
                    if curr_lr != actual_lr:
                        raise ValueError("some param groups have different lr")
            logging.info("Learning rate: " + str(actual_lr))
            if actual_lr < 10 ** (-7):
                last_lr = True
                
        else:
            logging.info("Learning rate: " + str(scheduler.get_last_lr()))
            

        for i, data in enumerate(train_loader, 0):
            
            if args.debug == 'DEBUG':
                print('\nbatch ' + str(i))
            
            # get the inputs & combine positive and negatives together
            img = data['image']
            img = img.cuda()
            
            labels = data['label']
            labels = labels.cuda()
            
            # zero the parameter gradients
            if args.msg == 'warmup':
                
                if epoch <= 4:
                    warmup_optimizer.zero_grad()
                else:
                    final_optimizer.zero_grad()
                
            else:
                optimizer.zero_grad()

            # forward + backward + optimize
            if args.desired_features == 'None':
                outputs = model.forward(img)
            else:
                outputs, features = model.forward(img, args.desired_features)
            
            # Compute RPL loss
            loss, open_loss, closed_loss, logits = compute_rpl_loss(model, outputs, labels, criterion, args.lamb, args.gamma, args.divide == 'TRUE')
            train_rpl_loss += loss.item()
            
            loss.backward()
            
            if args.lr_scheduler == 'warmup':
               
                if epoch <= 4:
                    warmup_optimizer.step()
                else:
                    final_optimizer.step()
                
            else:
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
        logging.info("Training Accuracy is: " + str(train_acc))
        logging.info("Average overall training loss in epoch is: " + str(running_loss/train_n))

        model.eval()
        used_running_loss, used_val_acc = evaluate_val(model, val_loader, args.gamma, args.lamb args.desired_features, args.divide)
        
        # Adjust learning rate
        if args.lr_scheduler == 'warmup':
            if epoch <= 4:
                warmup_scheduler.step()
            else:
                final_scheduler.step()  
        elif args.lr_scheduler == 'patience':
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
        
