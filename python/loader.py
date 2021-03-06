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
means_K = np.array([0.39, 0.39, 0.38])
std=[0.229, 0.224, 0.225]
std_K = [1.33, 1.43, 1.44]

label_translation= {0:0,
                    1:1,#road
                    2:2, #sidewalk
                    3:3,#building
                    4:3, #wall
                    5:3,#billboard
                    6:6, #pole
                    7:7, #trafic light
                    8:5, #trafic sign
                    9:8, #vegetation
                    10:8,#terrain
                    11:9,#sky
                    12:10,# pedestrian
                    13:10,# rider
                    14:11, #car
                    15:12, #truck
                    16:12, #bus
                    17:12, #train
                    18:11, #moto
                    19:11, #bike
                    20:1, #roadmarks
                    21:0, #unknown
                    }
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
        if isinstance(csv_file, dict):
            self.data = None
            for key in csv_file:
                df = pd.read_csv(key)
                if self.data is None:
                    self.data = df.sample(frac = csv_file[key], random_state=0)
                    
                else:
                    if csv_file[key] == 1:
                        self.data = self.data.append(df)
                    else:
                        self.data = self.data.append(df.sample(frac = csv_file[key], random_state=0))
        else:
            self.data            = pd.read_csv(csv_file)
        #print(self.data.keys())
        #print(self.data.tail())
        
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)
        print(len(self.data))
        self.phase           = phase
        self.size            = size
        self.transform_MSS = transforms.Compose([
                                    transforms.ColorJitter(brightness=0.21, contrast=0.21, saturation=0.2, hue=0.1),
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
        label_name  = self.data.iloc[idx, 1]
        try:
            input_image = Image.open(img_name).convert('RGB')
        except:
            print(img_name, label_name)
        
        if "Synthia" in label_name:
            """
            image = Image.open(label_name)
            labels = np.asarray(image.convert("RGB"))
            labels = labels[:,:,0]
            height, weight = labels.shape
            label = np.zeros((height,weight))
            
            for h in range(height):
                for w in range(weight):
                    try:
                        label[h,w] = label_translation[labels[h,w]]
                    except:
                        label[h,w] = 0
            """
            label       = np.load(label_name+".npy")
        else:
            label       = np.load(label_name)
        if "npz" in label_name:
            label = np.array(label['arr_0'], dtype=np.uint8)
            mask = Image.fromarray(label)
        else:
            mask        = Image.fromarray(label.astype(np.uint8))
            

        # reduce mean

        if self.phase=="train":
            # TO FILL:trasnform training data
            img, mask = self.transform(input_image, mask)
        else:
            # TO FILL:trasnform test data
            img, mask = self.test_transform(input_image, mask)

        img = self.transform_MSS(img)
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
            plt.imshow(labels[i,...].squeeze(), interpolation='nearest')
        plt.title('Labels')
        plt.figure()
        #labels = batch['Y'].cpu().numpy()

        for i in range(batch_size):
            plt.subplot(2, batch_size/2, i+1)
            plt.imshow(self.label_to_RGB(labels[i,...]), interpolation='nearest')
        plt.title('Labels')
        plt.show()



datanames_csvfiles = {"./../CityScapes/train.csv": 0.01,
                      './../MSS_vids/data/images/Coche.csv': 0.01,
                      './../Synthia/train.csv': 0,
                      "./../Kitti/training/train.csv": 0.001,
                      './../semantic_drone_dataset/train_dron.csv':1,
                      './../MSS_vids/data/images/AutoBus.csv':0.01,
                      
                      './../MSS_vids/data/images/Helicoptero.csv':0.01,
                      './../MSS_vids/data/images/Peaton.csv':0.01,
                      './../MSS_vids/data/images/Video.csv':0.01}
if __name__ == "__main__":
    root_dir   = "./../Kitti/training/"
    train_file = os.path.join(root_dir, "train.csv")
    print("_".join(datanames_csvfiles.keys()))
    train_data = loader(csv_file=datanames_csvfiles, phase='train')

    # show a batch
    batch_size = 8


    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    for i, batch in enumerate(dataloader):
        #np_b = batch['X'].numpy()
        #np_b = np_b.reshape((40,3,224,224))
        #print(np.std(np_b[:,0,:,:]),np.std(np_b[:,1,:,:]),np.std(np_b[:,2,:,:]),np_b.shape)
        train_data.show_batch(batch)

