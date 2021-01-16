import torch
from torch.utils.data import Dataset
from PIL import Image


class StandardDataset(Dataset):
    
    def __init__(self, img_list, folder_to_idx, folder_to_name, transform, img_base_path, triplet_option=False, num_neg=None, idx_to_folder=None, per_cls_imgs_dict=None, per_cls_neg_cls_dict=None, test_tiny_img2folder=None):
        self.transform = transform
        self.folder_to_idx = folder_to_idx
        self.idx_to_folder = idx_to_folder
        self.folder_to_name = folder_to_name
        self.img_base_path = img_base_path
        self.triplet_option = triplet_option
        self.num_neg = num_neg
        self.img_list = img_list
        self.per_cls_imgs_dict = per_cls_imgs_dict
        self.per_cls_neg_cls_dict = per_cls_neg_cls_dict
        self.test_tiny_img2folder = test_tiny_img2folder
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_shortcut = self.img_list[idx]
        img_folder_begin_idx = img_shortcut.find('/n') + 1
        img_folder = img_shortcut[img_folder_begin_idx: img_folder_begin_idx + 9]

        img = Image.open(self.img_base_path + img_shortcut).convert('RGB')
        img = self.transform(img)
        
        if self.test_tiny_img2folder is None:
            label = int(self.folder_to_idx[img_folder])
        else:
            img_folder = self.test_tiny_img2folder[img_shortcut]
            label = self.folder_to_idx[img_folder]
        
        sample = {"image": img, "label": label, "folder_name": img_folder[1:]}
        
        return sample
