# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

#from Cityscapes_loader import CityscapesDataset
#from CamVid_loader import CamVidDataset
from MSS_loader import MSSDataset
from MSS_utils import index2color
from Kitti_adapted_loader import KittiDataset
from loader import loader
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from torchvision import utils
from PIL import Image

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
n_class    = 13
img_Denorm = DeNormalize(means, std)
batch_size = 1
epochs    = 500
lr        = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma     = 0.5
configs    = "DL3s-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)

model_dir = "models"
model_path = os.path.join(model_dir, configs)


root_dir = './../Mapilliary/'
train_file = os.path.join(root_dir, "train.csv")
train_data = loader(csv_file=train_file, phase='test')
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)


model = torch.load("./models/deeplab/finetune/MSS_complete-1_Synthia-1_Mapilliary-0.25_")
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
model.eval()    


means    = np.array([86.5628,86.6691,86.7348]) / 255 # mean of three channels in the order of BGR


def label_to_RGB(image):

    image = image.squeeze()
    print(image.shape)
    height, weight = image.shape

    rgb = np.zeros((height, weight, 3))
    for h in range(height):
        for w in range(weight):
            rgb[h,w,:] = index2color[image[h,w]]
    return rgb.astype(np.uint8)

def oh_2_idx(image):
    label =  [np.where(r==1)[0][0] for r in image]
    return label

def show_batch(batch):
    img_batch = batch['X']
    batch_size = len(img_batch)
    #np_batch = img_batch.cpu().numpy().reshape((4,3,224,224))
    plt.figure()

    for i in range(batch_size):
         image_np = (255*(img_Denorm(img_batch[i,...]).data.permute(1,2,0).cpu().numpy())).astype(np.uint8)
         img_pil = Image.fromarray(image_np)
         plt.subplot(1, 3, i+1)
         plt.imshow(img_pil, interpolation='nearest')
         plt.subplot(1, 3, i+1).set_title('RGB')
         plt.axis('off')
    #plt.title('Batch from dataloader')

    #plt.figure()
    labels = batch['Y'].cpu().numpy()

    for i in range(batch_size):
         plt.subplot(1, 3, 2)
         plt.imshow(label_to_RGB(labels[i,...]), interpolation='nearest')
         plt.subplot(1, 3, 2).set_title('GT')
         plt.axis('off')
    #plt.title('Labels')

    outputs = model(batch['X'].cuda())['out']
    output = outputs.data.cpu().numpy()

    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
    #plt.figure()
    for i in range(N):
        plt.subplot(1, 3, 3)
        plt.imshow(label_to_RGB(pred[i,...]), interpolation='nearest')
        plt.subplot(1, 3, 3).set_title('Output')
        plt.axis('off')

    



if __name__ == "__main__":

    # show a batch
        
    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size(), batch['Y'].size())

        show_batch(batch)
        plt.axis('off')
        plt.ioff()
        plt.show()

