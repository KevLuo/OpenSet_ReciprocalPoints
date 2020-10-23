import os
import shutil

import argparse
import torch
import pickle

import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_methods.image_dataset_creation import folder_to_name, name_to_folder
from data_methods.create_hypernym_mapping import folder_to_label_idx
from zsh.baseline.cifar_dataset import CIFARDataset
from zsh.baseline.dataset import StandardDataset
from zsh.baseline.open_dataset import OpenDataset
from zsh.rpl.backbone import encoder32
from zsh.robust.stats import *

from zsh.utils import calculate_img_statistics, csv_to_desired_leaves, modified_folder_to_idx, select_subset



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset_type", type=str,
                      help="e.g. VAL or TEST")

    parser.add_argument("gap", type=str,
                    help="TRUE iff use global average pooling layer. Otherwise, use linear layer.")
    parser.add_argument("desired_features", type=str,
                    help="None means no features desired. Other examples include last, 2_to_last.")
    parser.add_argument("latent_size", type=int,
                    help="Dimension of embeddings.")
    parser.add_argument("num_rp_per_cls", type=int,
                    help="Number of reciprocal points per class.")
    parser.add_argument("gamma", type=float,
                    help="")
    
    parser.add_argument("dropout_rate", type=float,
                    help="if dropout=FALSE, place 0.")
    
    parser.add_argument("backbone_type", type=str,
                    help="architecture of backbone")
    
    parser.add_argument("closed_dataset_folder_path", type=str,
                    help="path to closed dataset folder.")
    parser.add_argument("open_dataset_name", type=str,
                    help="name of folder where open dataset lives. if operating on closed dataset, put NONE.")
    parser.add_argument("cifar100_path", type=str,
                    help="path to cifar100.")
    
    parser.add_argument("model_folder_path", type=str,
                      help="full file path to folder where the baseline is located")

    args = parser.parse_args()

        
    if args.dataset_type == 'VAL':
        with open(args.closed_dataset_folder_path + 'closed_val_img_list.pkl', 'rb') as f:
            seen_data = pickle.load(f)
        with open(args.closed_dataset_folder_path + args.open_dataset_name + '/open_val_img_list.pkl', 'rb') as f:
            unseen_data = pickle.load(f)
            
    elif args.dataset_type == 'TEST':
        with open(args.closed_dataset_folder_path + 'closed_test_img_list.pkl', 'rb') as f:
            seen_data = pickle.load(f)
        with open(args.closed_dataset_folder_path + args.open_dataset_name + '/open_test_img_list.pkl', 'rb') as f:
            unseen_data = pickle.load(f)  
    else:
        raise Exception('Unsupported dataset type')

    with open(args.closed_dataset_folder_path + 'closed_classes.pkl', 'rb') as f:
        closed_classes = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'closed_train_mean.pkl', 'rb') as f:
        train_mean = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'closed_train_std.pkl', 'rb') as f:
        train_std = pickle.load(f)
    img_normalize = transforms.Normalize(mean=train_mean,
                                             std=train_std)
    num_classes = len(closed_classes)

        
    with open(args.closed_dataset_folder_path + args.open_dataset_name + '/open_label_to_idx.pkl', 'rb') as f:
        open_label_to_idx = pickle.load(f)
    with open(args.closed_dataset_folder_path + args.open_dataset_name + '/open_idx_to_label.pkl', 'rb') as f:
        open_idx_to_label = pickle.load(f)
    with open(args.cifar100_path + 'train', 'rb') as fo:
        open_train_obj = pickle.load(fo, encoding='bytes')
    with open(args.cifar100_path + 'test', 'rb') as fo:
        open_test_obj = pickle.load(fo, encoding='bytes')
    with open(args.cifar100_path + 'meta', 'rb') as fo:
        open_meta_dict = pickle.load(fo, encoding='bytes')


    with open(args.closed_dataset_folder_path + 'label_to_idx.pkl', 'rb') as f:
        label_to_idx = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'idx_to_label.pkl', 'rb') as f:
        idx_to_label = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'train_obj.pkl', 'rb') as f:
        train_obj = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'test_obj.pkl', 'rb') as f:
        test_obj = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'meta.pkl', 'rb') as f:
        meta_dict = pickle.load(f)
        
    files = os.listdir(args.model_folder_path)
    model_name = None
    for file in files:
        if file[-3:] == '.pt':
            if model_name is not None:
                raise ValueError("Multiple possible models.")
            model_name = file[:-3]

    
    if args.dataset_type == 'VAL':
        seen_dataset = CIFARDataset(seen_data, train_obj, meta_dict, label_to_idx,
                                  transforms.Compose([
                                                transforms.ToTensor(),
                                                img_normalize,
                                                ]), openset=False)
        unseen_dataset = CIFARDataset(unseen_data, open_train_obj, open_meta_dict, open_label_to_idx,
                                  transforms.Compose([
                                                transforms.ToTensor(),
                                                img_normalize,
                                                ]), openset=True)
        
    else:
        seen_dataset = CIFARDataset(seen_data, test_obj, meta_dict, label_to_idx,
                                  transforms.Compose([
                                                transforms.ToTensor(),
                                                img_normalize,
                                                ]), openset=False)
        unseen_dataset = CIFARDataset(unseen_data, open_test_obj, open_meta_dict, open_label_to_idx,
                                  transforms.Compose([
                                                transforms.ToTensor(),
                                                img_normalize,
                                                ]), openset=True)

        
    seen_loader = DataLoader(seen_dataset, batch_size=64, shuffle=False, num_workers=3)
    unseen_loader = DataLoader(unseen_dataset, batch_size=64, shuffle=False, num_workers=3)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    if args.backbone_type == 'OSCRI_encoder':
        model = encoder32(latent_size=args.latent_size, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls, dropout_rate=args.dropout_rate, gap=args.gap == 'TRUE')
    else:
        raise ValueError(args.backbone_type + ' is not supported.')
        
    model.load_state_dict(torch.load(args.model_folder_path + model_name + '.pt'))
    model.cuda()
    model.eval()

    if args.dataset_type == 'VAL':
        seen_confidence_dict, seen_features_dict, seen_avg_norm_loss = collect_rpl_max(model, 'seen_val', seen_loader, folder_to_name, args.gamma, args.desired_features, cifar=True, idx_to_label=idx_to_label)
        unseen_confidence_dict, unseen_features_dict, unseen_avg_norm_loss = collect_rpl_max(model, 'zeroshot_val', unseen_loader, folder_to_name, args.gamma, args.desired_features, cifar=True, idx_to_label=open_idx_to_label)
    else:
        seen_confidence_dict, seen_features_dict, seen_avg_norm_loss = collect_rpl_max(model, 'seen_test', seen_loader, folder_to_name, args.gamma, args.desired_features, cifar=True, idx_to_label=idx_to_label)
        unseen_confidence_dict, unseen_features_dict, unseen_avg_norm_loss = collect_rpl_max(model, 'zeroshot_test', unseen_loader, folder_to_name, args.gamma, args.desired_features, cifar=True, idx_to_label=open_idx_to_label)
    
    recordings_folder = args.model_folder_path + args.open_dataset_name + '_' + args.dataset_type + '_recordings_' + model_name + '/'
    if os.path.isdir(recordings_folder):
        print(recordings_folder + ' already exists. Removing existing and creating new.')
        shutil.rmtree(recordings_folder)
    os.mkdir(recordings_folder)
    with open(recordings_folder + 'seen_confidence_dict.pkl', 'wb') as f:
        pickle.dump(seen_confidence_dict, f)
    with open(recordings_folder + 'unseen_confidence_dict.pkl', 'wb') as f:
        pickle.dump(unseen_confidence_dict, f)

    metrics_folder = args.model_folder_path + args.open_dataset_name + '_metrics_' + args.dataset_type + '_' + str(model_name) + '/'
    if os.path.isdir(metrics_folder):
        print(metrics_folder + ' already exists. Removing existing and creating new.')
        shutil.rmtree(metrics_folder)
    os.mkdir(metrics_folder)
    
    thresh = list(np.linspace(0, 0.999, 1000))
    thresh = thresh + [0.9991, 0.9992, 0.9993, 0.9994, 0.9995, 0.99975, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999, 0.9999995, 0.9999999, 0.99999995, 0.99999999, 0.999999995]

    seen_info = seenval_baseline_thresh(seen_confidence_dict, thresh, cifar=True, label_to_idx=label_to_idx, save_path=metrics_folder + 'seenres.pkl')
    unseen_info = unseenval_baseline_thresh(unseen_confidence_dict, thresh, save_path=metrics_folder + 'unseenres.pkl')

    dist_auroc_score = calc_auroc(seen_confidence_dict, unseen_confidence_dict, 'dist')
    prob_auroc_score = calc_auroc(seen_confidence_dict, unseen_confidence_dict, 'prob')
    
    print("Dist-Auroc score: " + str(dist_auroc_score))
    print("Prob-Auroc score: " + str(prob_auroc_score))
    
    metrics = summarize(seen_info, unseen_info, thresh, verbose=False)
    metrics['dist_auroc_lwnealstyle'] = dist_auroc_score
    metrics['prob_auroc_lwnealstyle'] = prob_auroc_score

    with open(metrics_folder + 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
        