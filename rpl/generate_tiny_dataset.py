import argparse
import ast
import os
import pickle
from random import shuffle

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils import build_classes_mappings, calculate_img_statistics, folder_to_name, name_to_folder



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("closed_val_prop", type=float,
                    help="proportion of closed train set to make validation.")
    parser.add_argument("img_size", type=int,
                    help="desired square img size for tiny imagenet.")
    parser.add_argument("closed_split_path", type=str,
                    help="path to lwneal closed split data.")
    parser.add_argument("open_split_path", type=str,
                    help="path to lwneal open split data.")
    parser.add_argument("tiny_path", type=str,
                    help="path to tiny imagenet.")
    parser.add_argument("save_folder", type=str,
                    help="folder to save dataset in. Will be created, so cannot exist previously.")

    args = parser.parse_args()



    # create name to folder
    name_to_folder_map = name_to_folder(args.tiny_path + 'train')
    # create folder to name
    folder_to_name_map = folder_to_name(args.tiny_path + 'train')


    closed_train_img_list = []
    closed_test_img_list = []
    closed_cls_folders = set()
    with open(args.closed_split_path, 'r') as f:
        for line in f:
            curr_data = ast.literal_eval(line)
            curr_img_path = curr_data['filename']
            if curr_data['fold'] == 'train':
                closed_train_img_list.append(curr_img_path)
                folder_begin_idx = curr_img_path.find('/n') + 1 
                closed_cls_folders.add(curr_img_path[folder_begin_idx: folder_begin_idx + 9])
            else:
                closed_test_img_list.append(curr_img_path)
                
    
    open_val_img_list = []
    open_test_img_list = []
    open_cls_folders = set()
    with open(args.open_split_path, 'r') as f:
        for line in f:
            curr_data = ast.literal_eval(line)
            curr_img_path = curr_data['filename']
            if curr_data['fold'] == 'train':
                open_val_img_list.append(curr_img_path)
                folder_begin_idx = curr_img_path.find('/n') + 1 
                open_cls_folders.add(curr_img_path[folder_begin_idx: folder_begin_idx + 9])
            else:
                open_test_img_list.append(curr_img_path)
    
    closed_cls_folders = list(closed_cls_folders)
    open_cls_folders = list(open_cls_folders)
    
    closed_classes, open_classes, folder_to_idx, idx_to_folder = build_classes_mappings(closed_cls_folders, open_cls_folders, folder_to_name_map)
    
    # build a validation set for closed-set classification
    shuffle(closed_train_img_list)
    closed_val_img_list = closed_train_img_list[:int(args.closed_val_prop * len(closed_train_img_list))]
    closed_train_img_list = closed_train_img_list[int(args.closed_val_prop * len(closed_train_img_list)):]
    
    train_mean, train_std = calculate_img_statistics(closed_train_img_list, folder_to_idx, folder_to_name_map, args.img_size)
    
    print(train_mean)
    print(train_std)
    
    os.mkdir(args.save_folder)
    
    with open(args.save_folder + 'folder_to_name_map.pkl', 'wb') as f:
        pickle.dump(folder_to_name_map, f)
    with open(args.save_folder + 'name_to_folder_map.pkl', 'wb') as f:
        pickle.dump(name_to_folder_map, f)
    with open(args.save_folder + 'folder_to_idx.pkl', 'wb') as f:
        pickle.dump(folder_to_idx, f)
    with open(args.save_folder + 'idx_to_folder.pkl', 'wb') as f:
        pickle.dump(idx_to_folder, f)
    with open(args.save_folder + 'closed_classes.pkl', 'wb') as f:
        pickle.dump(closed_classes, f)
    with open(args.save_folder + 'open_classes.pkl', 'wb') as f:
        pickle.dump(open_classes, f)
    with open(args.save_folder + 'closed_train_img_list.pkl', 'wb') as f:
        pickle.dump(closed_train_img_list, f)
    with open(args.save_folder + 'closed_val_img_list.pkl', 'wb') as f:
        pickle.dump(closed_val_img_list, f)
    with open(args.save_folder + 'closed_test_img_list.pkl', 'wb') as f:
        pickle.dump(closed_test_img_list, f)
    with open(args.save_folder + 'open_val_img_list.pkl', 'wb') as f:
        pickle.dump(open_val_img_list, f)
    with open(args.save_folder + 'open_test_img_list.pkl', 'wb') as f:
        pickle.dump(open_test_img_list, f)
    with open(args.save_folder + 'closed_train_mean.pkl', 'wb') as f:
        pickle.dump(train_mean, f)
    with open(args.save_folder + 'closed_train_std.pkl', 'wb') as f:
        pickle.dump(train_std, f) 
