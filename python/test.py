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

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from torchvision import utils



n_class    = 13

batch_size = 2
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
configs    = "DL3s-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)

model_dir = "models"
model_path = os.path.join(model_dir, configs)


root_dir = "./../../MSS_vid/data/"
train_file = os.path.join(root_dir, "train.csv")
if sys.argv[1] == 'CamVid':
	train_data = CamVidDataset(csv_file=train_file, phase='train')
elif sys.argv[1] == 'CityScapes':
	train_data = CityscapesDataset(csv_file=train_file, phase='train')
else:
	train_data = MSSDataset(csv_file=train_file, phase='train')
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)


model = torch.load("./models/DL3")
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
model.eval()    


means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR


def label_to_RGB(image):
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
	img_batch[:,0,...].add_(means[0])
	img_batch[:,1,...].add_(means[1])
	img_batch[:,2,...].add_(means[2])
	batch_size = len(img_batch)

	grid = utils.make_grid(img_batch)
	plt.figure()
	plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

	plt.title('Batch from dataloader')

	plt.figure()
	labels = batch['l'].cpu().numpy()
	for i in range(len(labels)):
		plt.subplot(2, len(labels)/2, i+1)
		plt.imshow(label_to_RGB(labels[i,:,:]), interpolation='nearest')
	plt.title('Labels')
	outputs = model(batch['X'].cuda())['out']
	output = outputs.data.cpu().numpy()

	N, _, h, w = output.shape
	pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
	plt.figure()
	for i in range(N):
		plt.subplot(2, len(labels)/2, i+1)
		plt.imshow(label_to_RGB(pred[i,:,:]), interpolation='nearest')
	plt.title('Output')
	plt.show()

	



if __name__ == "__main__":

	# show a batch
		
	for i, batch in enumerate(dataloader):
		print(i, batch['X'].size(), batch['Y'].size())

		show_batch(batch)
		plt.axis('off')
		plt.ioff()
		plt.show()

