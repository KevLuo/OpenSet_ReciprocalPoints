import logging
import os

from nltk.corpus import wordnet as wn
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from datasets.dataset import StandardDataset



def build_classes_mappings(closed_cls_folders, open_cls_folders, folder_to_name_map):
    folder_to_idx = {}
    idx_to_folder = {}
    closed_classes = []
    for i in range(0, len(closed_cls_folders)):
        folder_to_idx[closed_cls_folders[i]] = i
        idx_to_folder[i] = closed_cls_folders[i]
        closed_classes.append(folder_to_name_map[closed_cls_folders[i]])
    open_classes = []
    for j in range(0, len(open_cls_folders)):
        open_classes.append(folder_to_name_map[open_cls_folders[j]])
        
    return closed_classes, open_classes, folder_to_idx, idx_to_folder


def calculate_img_statistics(img_list, folder_to_idx, folder_to_name, img_size):
    """ Given dataset, calculate the mean pixel per channel and the std per channel.
    """
    
    temp_dataset = StandardDataset(img_list, folder_to_idx, folder_to_name,       
                                        transforms.Compose([
                                        transforms.Resize((img_size,img_size)),
                                        transforms.ToTensor()
                                        ]))
    temp_loader = DataLoader(temp_dataset, batch_size=512, shuffle=False, num_workers=3)
    
    sum_mean = np.array([0.0, 0.0, 0.0])
    num_images = 0
    for i, data in enumerate(temp_loader, 0):
        img = data['image']
        assert img.shape[1] == 3
        assert img.shape[2] == img_size
        assert img.shape[3] == img_size
        num_images += img.shape[0]
        sum_mean[0] += torch.sum(img[:,0,:,:]).item()
        sum_mean[1] += torch.sum(img[:,1,:,:]).item()
        sum_mean[2] += torch.sum(img[:,2,:,:]).item()
    mean = sum_mean/(img_size * img_size * num_images)
    
    sum_variance = np.array([0.0, 0.0, 0.0])
    for i, data in enumerate(temp_loader, 0):
        img = data['image']
        sum_variance[0] += torch.sum((img[:,0,:,:] - mean[0])**2).item()
        sum_variance[1] += torch.sum((img[:,1,:,:] - mean[1])**2).item()
        sum_variance[2] += torch.sum((img[:,2,:,:] - mean[2])**2).item()
    variance = sum_variance/(img_size * img_size * num_images)
    std = np.sqrt(variance)
    
    return mean, std


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


def folder_to_name(path):
    folder_to_name_map = {}
    imagenet_leaves_folders = os.listdir(path)
    for folder in imagenet_leaves_folders:
        # Set the class-specific information
        synset_offset = int(folder[1:])
        class_id = folder
        synset_obj = wn.synset_from_pos_and_offset('n', synset_offset)
        synset_name = synset_obj.name()

        folder_to_name_map[folder] = synset_name
            
    return folder_to_name_map


def name_to_folder(path):
    name_to_folder_map = {}
    imagenet_leaves_folders = os.listdir(path)
    for folder in imagenet_leaves_folders:
        # Set the class-specific information
        synset_offset = int(folder[1:])
        class_id = folder
        synset_obj = wn.synset_from_pos_and_offset('n', synset_offset)
        synset_name = synset_obj.name()

        name_to_folder_map[synset_name] = folder
            
    return name_to_folder_map


def setup_logger(name, formatter, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger