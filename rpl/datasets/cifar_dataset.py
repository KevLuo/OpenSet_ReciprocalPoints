from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
    
    def __init__(self, img_list, cifar_obj, meta_dict, label_to_idx, transform, openset=False):
        self.img_list = img_list
        self.cifar_obj = cifar_obj
        self.meta_dict = meta_dict
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.openset=openset
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_idx = self.img_list[idx]
        if self.openset:
            img = self.cifar_obj[b'data'][img_idx]
        else:
            img = self.cifar_obj['data'][img_idx]
        img = np.reshape(img, (3, 32, 32))
        img = np.transpose(img, [1,2,0])
        img = Image.fromarray(img)
        
        img = self.transform(img)

        if self.openset:
            filename = self.cifar_obj[b'filenames'][img_idx]
            if filename in [b'baby_s_000333.png', b'baby_s_000223.png']:
                label_name = 'baby'
            else:
                cifar_label = self.cifar_obj[b'fine_labels'][img_idx]
                label_name = self.meta_dict[b'fine_label_names'][cifar_label].decode("utf-8")    
        else:
            cifar_label = self.cifar_obj['labels'][img_idx]
            label_name = self.meta_dict[b'label_names'][cifar_label].decode("utf-8") 
        
#         if label_name not in self.label_to_idx:
#             print(filename)
#             print(label_name)
            
        label_idx = self.label_to_idx[label_name]
        
        sample = {"image": img, "label": label_idx}
        
        return sample
