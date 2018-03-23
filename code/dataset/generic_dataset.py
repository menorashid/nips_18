import os
import torch
import numpy as np
import torch.utils.data
from torchvision import transforms
from helpers import util, visualize
import scipy.misc
from PIL import Image
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text_file, transform=None):
        self.files = util.readLinesFromFile(text_file)
        self.transform = transform
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        train_file_curr,label = train_file_curr.split(' ')
        label = int(label)
        image = Image.open(train_file_curr)
        sample = {'image': image, 'label': label}
        # if self.transform:
        sample['image'] = self.transform(sample['image'])

        return sample
