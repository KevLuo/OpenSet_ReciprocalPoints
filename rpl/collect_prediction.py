import argparse
import os
import re
import shutil

import numpy as np
import pickle
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.dataset import StandardDataset
from datasets.open_dataset import OpenDataset
from models.backbone import encoder32
from models.backbone_resnet import encoder
from models.backbone_wide_resnet import wide_encoder
from evaluate import collect_rpl_max, seenval_baseline_thresh, unseenval_baseline_thresh



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("type", type=str,
                      help="e.g. LT or IMAGENET or TINY")
    parser.add_argument("dataset_type", type=str,
                      help="e.g. VAL or TEST")
    parser.add_argument("gap", type=str,
                    help="TRUE iff use global average pooling layer. Otherwise, use linear layer.")
    
    parser.add_argument("latent_size", type=int,
                    help="Dimension of embeddings.")
    parser.add_argument("num_rp_per_cls", type=int,
                    help="Number of reciprocal points per class.")
    parser.add_argument("gamma", type=float,
                    help="")
    
    parser.add_argument("backbone_type", type=str,
                    help="architecture of backbone")
    parser.add_argument("img_size", type=int,
                    help="desired square image size.")
    
    parser.add_argument("img_base_path", type=str,
                    help="path to folder containing image data, i.e. /data/ or /share/nikola/export/image_datasets/")
    parser.add_argument("dataset_folder_path", type=str,
                    help="path to dataset folder.")
    parser.add_argument("dataset_folder", type=str,
                    help="name of folder where dataset details live.")
    
    parser.add_argument("model_folder_path", type=str,
                      help="full file path to folder where the baseline is located")

    args = parser.parse_args()

    with open(args.dataset_folder_path + args.dataset_folder + '/folder_to_name_map.pkl', 'rb') as f:
        folder_to_name = pickle.load(f)
    with open(args.dataset_folder_path + args.dataset_folder + '/name_to_folder_map.pkl', 'rb') as f:
        name_to_folder = pickle.load(f)
    with open(args.dataset_folder_path + args.dataset_folder + '/folder_to_idx.pkl', 'rb') as f:
        folder_to_idx = pickle.load(f)
    with open(args.dataset_folder_path + args.dataset_folder + '/idx_to_folder.pkl', 'rb') as f:
        idx_to_folder = pickle.load(f)
    with open(args.dataset_folder_path + args.dataset_folder + '/closed_classes.pkl', 'rb') as f:
        closed_classes = pickle.load(f)
        
    files = os.listdir(args.model_folder_path)
    model_name = None
    for file in files:
        if file[-3:] == '.pt':
            if model_name is not None:
                raise ValueError('Multiple possible models in ' + args.model_folder_path)
            model_name = file[:-3]         
 
    if args.dataset_type == 'VAL':
        with open(args.dataset_folder_path + args.dataset_folder + '/closed_val_img_list.pkl', 'rb') as f:
            seen_data = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/open_val_img_list.pkl', 'rb') as f:
            unseen_data = pickle.load(f)
    elif args.dataset_type == 'TEST':
        with open(args.dataset_folder_path + args.dataset_folder + '/closed_test_img_list.pkl', 'rb') as f:
            seen_data = pickle.load(f)
        with open(args.dataset_folder_path + args.dataset_folder + '/open_test_img_list.pkl', 'rb') as f:
            unseen_data = pickle.load(f)
        if args.type == 'TINY':
            with open(args.dataset_folder_path + args.dataset_folder + '/closedtest_img_to_folder.pkl', 'rb') as f:
                seen_tiny_img2folder = pickle.load(f)  
            with open(args.dataset_folder_path + args.dataset_folder + '/opentest_img_to_folder.pkl', 'rb') as f:
                unseen_tiny_img2folder = pickle.load(f)  
        
    else:
        raise Exception('Unsupported dataset type')

    with open(args.dataset_folder_path + args.dataset_folder + '/closed_train_mean.pkl', 'rb') as f:
        train_mean = pickle.load(f)
    with open(args.dataset_folder_path + args.dataset_folder + '/closed_train_std.pkl', 'rb') as f:
        train_std = pickle.load(f)
    img_normalize = transforms.Normalize(mean=train_mean,
                                             std=train_std)
    num_classes = len(closed_classes)     
    
    
    if args.type == 'TINY' and args.dataset_type == 'TEST':
        seen_dataset = StandardDataset(seen_data, folder_to_idx, folder_to_name,      
                                                transforms.Compose([
                                                transforms.Resize((args.img_size,args.img_size)),
                                                transforms.ToTensor(),
                                                img_normalize, 
                                                ]), args.img_base_path, test_tiny_img2folder=seen_tiny_img2folder)
        unseen_dataset = OpenDataset(unseen_data, transforms.Compose([
                                    transforms.Resize((args.img_size,args.img_size)),
                                    transforms.ToTensor(),
                                    img_normalize,
                                    ]), args.img_base_path, test_tiny_img2folder=unseen_tiny_img2folder)
              
    else:
        seen_dataset = StandardDataset(seen_data, folder_to_idx, folder_to_name,      
                                                transforms.Compose([
                                                transforms.Resize((args.img_size,args.img_size)),
                                                transforms.ToTensor(),
                                                img_normalize,
                                                ]), args.img_base_path)
        unseen_dataset = OpenDataset(unseen_data, transforms.Compose([
                                    transforms.Resize((args.img_size,args.img_size)),
                                    transforms.ToTensor(),
                                    img_normalize,
                                    ]), args.img_base_path)
    
        
    seen_loader = DataLoader(seen_dataset, batch_size=64, shuffle=False, num_workers=3)
    unseen_loader = DataLoader(unseen_dataset, batch_size=64, shuffle=False, num_workers=3)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
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
   
    model.load_state_dict(torch.load(args.model_folder_path + model_name + '.pt'))
    model.cuda()
    model.eval()

    if args.dataset_type == 'VAL':
        seen_confidence_dict = collect_rpl_max(model, 'seen_val', seen_loader, folder_to_name, args.gamma)
        unseen_confidence_dict = collect_rpl_max(model, 'zeroshot_val', unseen_loader, folder_to_name, args.gamma)
    else:
        seen_confidence_dict = collect_rpl_max(model, 'seen_test', seen_loader, folder_to_name, args.gamma)
        unseen_confidence_dict = collect_rpl_max(model, 'zeroshot_test', unseen_loader, folder_to_name, args.gamma)
        
    recordings_folder = args.model_folder_path + args.dataset_type + '_recordings_' + model_name + '/'
    if os.path.isdir(recordings_folder):
        print(recordings_folder + ' already exists. Removing existing and creating new.')
        shutil.rmtree(recordings_folder)
    os.mkdir(recordings_folder)
    with open(recordings_folder + 'seen_confidence_dict.pkl', 'wb') as f:
        pickle.dump(seen_confidence_dict, f)
    with open(recordings_folder + 'unseen_confidence_dict.pkl', 'wb') as f:
        pickle.dump(unseen_confidence_dict, f)

    metrics_folder = args.model_folder_path + 'metrics_' + args.dataset_type + '_' + str(model_name) + '/'
    if os.path.isdir(metrics_folder):
        print(metrics_folder + ' already exists. Removing existing and creating new.')
        shutil.rmtree(metrics_folder)
    os.mkdir(metrics_folder)
    
    prob_thresh = set()
    dist_thresh = set()
    for leaf_name in seen_confidence_dict.keys():
        for record in seen_confidence_dict[leaf_name]:
            prob_thresh.add(record['prob'])
            dist_thresh.add(record['dist'])
    for leaf_name in unseen_confidence_dict.keys():
        for record in unseen_confidence_dict[leaf_name]:
            prob_thresh.add(record['prob'])
            dist_thresh.add(record['dist'])
    prob_thresh = list(prob_thresh)
    prob_thresh.sort()
    dist_thresh = list(dist_thresh)
    dist_thresh.sort()
    
    print("number of prob thresholds: " + str(len(prob_thresh)))
    print("number of dist thresholds: " + str(len(dist_thresh)))

    prob_seen_info = seenval_baseline_thresh(seen_confidence_dict, prob_thresh, folder_to_idx=folder_to_idx, name_to_folder=name_to_folder, save_path=metrics_folder + 'probseenres.pkl')
    prob_unseen_info = unseenval_baseline_thresh(unseen_confidence_dict, prob_thresh, save_path=metrics_folder + 'probunseenres.pkl')
    
    dist_seen_info = seenval_baseline_thresh(seen_confidence_dict, dist_thresh, folder_to_idx=folder_to_idx, name_to_folder=name_to_folder, save_path=metrics_folder + 'distseenres.pkl', value_key='dist')
    dist_unseen_info = unseenval_baseline_thresh(unseen_confidence_dict, dist_thresh, save_path=metrics_folder + 'distunseenres.pkl', value_key='dist')

    dist_auroc_score = calc_auroc(seen_confidence_dict, unseen_confidence_dict, 'dist')
    prob_auroc_score = calc_auroc(seen_confidence_dict, unseen_confidence_dict, 'prob')
    
    print("Dist-Auroc score: " + str(dist_auroc_score))
    print("Prob-Auroc score: " + str(prob_auroc_score))
    
    metrics = summarize(prob_seen_info, prob_unseen_info, prob_thresh, verbose=False)
    dist_metrics = summarize(dist_seen_info, dist_unseen_info, dist_thresh, verbose=False)
    
    metrics['dist_auroc_lwnealstyle'] = dist_auroc_score
    metrics['prob_auroc_lwnealstyle'] = prob_auroc_score
    metrics['dist_OSR_CSR_AUC'] = dist_metrics['OSR_CSR_AUC']
    
    print("prob-AUC score: " + str(metrics['OSR_CSR_AUC']))
    print("dist-AUC score: " + str(metrics['dist_OSR_CSR_AUC']))
    
    with open(metrics_folder + 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
        
