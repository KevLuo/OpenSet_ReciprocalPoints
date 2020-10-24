import torch
from torch.utils.data import Dataset
from PIL import Image

from zsh.robust.triplet_utils import sample_pos_and_neg


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
            
        
        if self.triplet_option:
            sampled_pos_img_path, sampled_neg_img_paths = sample_pos_and_neg(label, self.num_neg, self.per_cls_imgs_dict, self.per_cls_neg_cls_dict, self.idx_to_folder, self.folder_to_name)
            
            sampled_pos_img = Image.open(sampled_pos_img_path).convert('RGB')
            sampled_pos_img = self.transform(sampled_pos_img)
                
            sampled_neg_imgs = torch.zeros(self.num_neg, 3, 224, 224)
            for neg_idx in range(0, len(sampled_neg_img_paths)):
                sampled_neg_img = Image.open(sampled_neg_img_paths[neg_idx]).convert('RGB')
                sampled_neg_img = self.transform(sampled_neg_img)
                sampled_neg_imgs[neg_idx] = sampled_neg_img

            sample = {"image": img, "label": label, "sampled_pos_img": sampled_pos_img, "sampled_neg_imgs": sampled_neg_imgs}
        
        else:
            sample = {"image": img, "label": label, "folder_name": img_folder[1:]}
        
        return sample
