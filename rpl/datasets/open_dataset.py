import torch
from torch.utils.data import Dataset
from PIL import Image


class OpenDataset(Dataset):
    
    def __init__(self, img_list, transform, img_base_path, test_tiny_img2folder=None):
        self.transform = transform 
        self.img_list = img_list
        self.img_base_path = img_base_path
        self.test_tiny_img2folder = test_tiny_img2folder
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_shortcut = self.img_list[idx]

        img_folder_begin_idx = img_shortcut.find('/n') + 1
        img_folder = img_shortcut[img_folder_begin_idx: img_folder_begin_idx + 9]

        img = Image.open(self.img_base_path + img_shortcut).convert('RGB')
        img = self.transform(img)

        if self.test_tiny_img2folder is not None:
            img_folder = self.test_tiny_img2folder[img_shortcut]
        
        sample = {"image": img, "folder_name": img_folder[1:]}
        
        return sample
