import argparse
import os
import shutil

import torch
import pickle

import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.cifar_dataset import CIFARDataset
from datasets.dataset import StandardDataset
from datasets.open_dataset import OpenDataset
from models.backbone import encoder32
from evaluate import collect_rpl_max, seenval_baseline_thresh, unseenval_baseline_thresh



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
        seen_confidence_dict = collect_rpl_max(model, 'seen_val', seen_loader, folder_to_name, args.gamma, cifar=True, idx_to_label=idx_to_label)
        unseen_confidence_dict = collect_rpl_max(model, 'zeroshot_val', unseen_loader, folder_to_name, args.gamma, cifar=True, idx_to_label=open_idx_to_label)
    else:
        seen_confidence_dict = collect_rpl_max(model, 'seen_test', seen_loader, folder_to_name, args.gamma, cifar=True, idx_to_label=idx_to_label)
        unseen_confidence_dict = collect_rpl_max(model, 'zeroshot_test', unseen_loader, folder_to_name, args.gamma, cifar=True, idx_to_label=open_idx_to_label)
    
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
    
    prob_seen_info = seenval_baseline_thresh(seen_confidence_dict, prob_thresh, cifar=True, label_to_idx=label_to_idx, name_to_folder=name_to_folder, save_path=metrics_folder + 'probseenres.pkl')
    prob_unseen_info = unseenval_baseline_thresh(unseen_confidence_dict, prob_thresh, save_path=metrics_folder + 'probunseenres.pkl')
    
    dist_seen_info = seenval_baseline_thresh(seen_confidence_dict, dist_thresh, cifar=True, label_to_idx=label_to_idx, name_to_folder=name_to_folder, save_path=metrics_folder + 'distseenres.pkl', value_key='dist')
    dist_unseen_info = unseenval_baseline_thresh(unseen_confidence_dict, dist_thresh, save_path=metrics_folder + 'distunseenres.pkl', value_key='dist')

    dist_auroc_score = calc_auroc(seen_confidence_dict, unseen_confidence_dict, 'dist')
    prob_auroc_score = calc_auroc(seen_confidence_dict, unseen_confidence_dict, 'prob')
    
    print("Dist-Auroc score: " + str(dist_auroc_score))
    print("Prob-Auroc score: " + str(prob_auroc_score))
    
    metrics = summarize(seen_info, unseen_info, thresh, verbose=False)
    metrics['dist_auroc_lwnealstyle'] = dist_auroc_score
    metrics['prob_auroc_lwnealstyle'] = prob_auroc_score
    metrics['dist_OSR_CSR_AUC'] = dist_metrics['OSR_CSR_AUC']
    
    print("prob-AUC score: " + str(metrics['OSR_CSR_AUC']))
    print("dist-AUC score: " + str(metrics['dist_OSR_CSR_AUC']))

    with open(metrics_folder + 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
        