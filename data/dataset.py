import os
import numpy as np
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from config import IMG_SIZE
from PIL import Image
from skimage.color import rgb2lab

class Dataset(Dataset):
    def __init__(self, data_path: str, split: str='train'):
        self.data_path = data_path
        self.image_files = [f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if split == 'train':
            # Horizontal flip for data augmentation while training
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'val':
            self.transform = transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

        self.split = split

    def __getitem__(self, index):
        img_name = os.path.join(self.data_path, self.image_files[index])
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)
        img = np.array(img)
        lab_img = rgb2lab(img).astype('float32')
        lab_img = transforms.ToTensor()(lab_img)
        L = lab_img[[0], ...] / 50. - 1. # Between -1 and 1
        ab = lab_img[[1, 2], ...] / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.image_files)



def create_dataloader(data_path: str, batch_size: int=16, pin_memory: bool=True, num_workers: int=4, split: str='train'):
    dataset = Dataset(data_path, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    return dataloader
