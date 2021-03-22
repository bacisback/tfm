#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:00:22 2021

@author: e321075
"""


from matplotlib import pyplot as plt
import os
import pandas as pd
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import numpy as np
import scipy.io as sio
import random
import sys
import argparse
import os
import time
from os.path import join
import csv
from MSS_utils import index2color


means=np.array([86.5628,86.6691,86.7348]) / 255
std=[0.229, 0.224, 0.225]

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class loader(Dataset):

    def __init__(self, csv_file, phase, size=224):
        self.data            = pd.read_csv(csv_file)
        self.phase           = phase
        self.size            = size
        self.transform_input = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, std),
                                ])
        self.transform_mask  = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
        self.img_Denorm = DeNormalize(means, std)

    def __len__(self):
        return len(self.data)

    def transform(self, image, mask):
        # TO FILL:
        # 1st Resize image and mask to 400x400 using nearest neighbor interpolation.
        resize = TF.resize
        image  = resize(image, (400, 400), interpolation=Image.NEAREST)
        mask   = resize(mask, (400, 400), interpolation=Image.NEAREST)
        # TO FILL:
        # 2nd Random crop: 
        # a) get random parameters for obtaining a 224x224 version of the image by cropping (to be used later)
        # This is to ensure that crop parameters are the same for image and mask. If not, the ground-truth mask would not be aligned with its image content.
        i, j, h, w = transforms.RandomCrop(224).get_params(image, [224, 224])
        # b) Crop according to these parameters
        image = transforms.functional.crop(image, i,j,h,w)
        mask  = transforms.functional.crop(mask, i,j,h,w) 
        # TO FILL:
        # 3rd random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        return image, mask

    # Default trasnformations on test data
    def test_transform(self, image, mask):
        # TO FILL:
        # 1st Resize image and mask to 400x400 using nearest neighbor interpolation.
        resize = TF.resize
        image  = resize(image, 400, interpolation=Image.NEAREST)
        mask   = resize(mask, 400, interpolation=Image.NEAREST)

        # TO FILL:
        # 2nd a 224x224 center crop: 
        crop = transforms.CenterCrop(224)
        image = crop(image)
        mask  = crop(mask)

        return image, mask

    def __getitem__(self, idx):
        img_name    = self.data.iloc[idx, 0]
        input_image = Image.open(img_name).convert('RGB')
        label_name  = self.data.iloc[idx, 1]
        label       = np.load(label_name)
        mask        = Image.fromarray(label.astype(np.uint8))
            

        # reduce mean

        if self.phase=="train":
            # TO FILL:trasnform training data
            img, mask = self.transform(input_image, mask)
        else:
            # TO FILL:trasnform test data
            img, mask = self.test_transform(input_image, mask)

        if self.transform_input is not None:
            img = self.transform_input(img)
        if self.transform_mask is not None:
            mask = 255*self.transform_mask(mask)
    
        sample = {'X': img, 'Y': mask.long()}

        return sample

    def label_to_RGB(self, image):
        image = image.squeeze()
        height, weight = image.shape

        rgb = np.zeros((height, weight, 3))
        for h in range(height):
            for w in range(weight):
                rgb[h,w,:] = index2color[image[h,w]]
        return rgb.astype(np.uint8)
    def show_batch(self, batch):
        img_batch = batch['X']
        batch_size = len(img_batch)
        #np_batch = img_batch.cpu().numpy().reshape((4,3,224,224))
        plt.figure()

        for i in range(batch_size):
            image_np = (255*(self.img_Denorm(img_batch[i,...]).data.permute(1,2,0).cpu().numpy())).astype(np.uint8)
            img_pil = Image.fromarray(image_np)
            plt.subplot(2, batch_size/2, i+1)
            plt.imshow(img_pil, interpolation='nearest')
        plt.title('Batch from dataloader')

        plt.figure()
        labels = batch['Y'].cpu().numpy()

        for i in range(batch_size):
            plt.subplot(2, batch_size/2, i+1)
            plt.imshow(self.label_to_RGB(labels[i,...]), interpolation='nearest')
        plt.title('Labels')
        plt.show()


if __name__ == "__main__":
    root_dir   = "./../CityScapes/"
    train_file = os.path.join(root_dir, "train.csv")
    train_data = loader(csv_file=train_file, phase='train')

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
    for i, batch in enumerate(dataloader):
        train_data.show_batch(batch)

