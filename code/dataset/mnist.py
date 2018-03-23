import os
import torch
import numpy as np
from torchvision import datasets, transforms
from helpers import util, visualize
import scipy.misc
from PIL import Image
import cv2
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_data, train , transform=None):

        raw_folder = 'raw'
        processed_folder = 'processed'
        training_file = 'training.pt'
        test_file = 'test.pt'

        if train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(dir_data, processed_folder, training_file))
        else:
            self.train_data, self.train_labels = torch.load(
                os.path.join(dir_data, processed_folder, test_file))

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        image, label = self.train_data[idx], self.train_labels[idx]
        image = Image.fromarray(image.numpy(), mode='L')
        sample = {'image':image,'label':label}
        sample['image'] = self.transform(sample['image'])
        return sample

        
