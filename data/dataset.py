import numpy as np
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from config import IMG_SIZE
from PIL import Image
from skimage.color import rgb2lab

class Dataset():
    def __init__(self, data_path: str, split: str='train'):
        if split == 'train':
            # Horizontal flip for data augmentation while training
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'val':
            self.transform = transforms.Resize((IMG_SIZE, IMG_SIZE))

        self.split = split
        self.data_path = data_path
        self.img_size = IMG_SIZE


    def __getitem__(self, index):
        img = Image.open(self.data_path[idx]).convert('RGB')
        img = self.transform(img)
        img = np.array(img)
        lab_img = rgb2lab(img).astype('float32')
        lab_img = transforms.ToTensor()(lab_img)
        L = lab_img[[0], ...] / 50. - 1. # Between -1 and 1
        ab = lab_img[[1, 2], ...] / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.data_path)



def create_dataloader(data_path: str, batch_size: int=16, pin_memory: bool=True, num_workers: int=4, split: str='train'):
    dataset = Dataset(data_path, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    return dataloader

