# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
#from Cityscapes_loader import CityscapesDataset
#from CamVid_loader import CamVidDataset
from MSS_loader import MSSDataset
from Kitti_adapted_loader import KittiDataset
from MSS_utils import index2color

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os


n_class    = 13

batch_size = 2
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
configs    = "DL3s-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)
momentum   = 1e-5
if sys.argv[1] == 'CamVid':
    root_dir   = "CamVid/"
elif sys.argv[1] == 'CityScapes':
    root_dir   = "CityScapes/"
else:
    root_dir = "./../../MSS_vid/data/"
train_file = os.path.join(root_dir, "train.csv")
if sys.argv[1] == 'MSS':
    val_dir    = "./../../Kitti/"
    val_file   = os.path.join(val_dir, "train.csv")
else:
    val_file   = os.path.join(root_dir, "val.csv")

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

if sys.argv[1] == 'CamVid':
    train_data = CamVidDataset(csv_file=train_file, phase='train')
elif sys.argv[1] == 'CityScapes':
    train_data = CityscapesDataset(csv_file=train_file, phase='train')
else:
    train_data = MSSDataset(csv_file=train_file, phase='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

if sys.argv[1] == 'CamVid':
    val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
elif sys.argv[1] == 'CityScapes':
    val_data = CityscapesDataset(csv_file=val_file, phase='val', flip_rate=0)
else:
    val_data = KittiDataset(csv_file=val_file, phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

if len(sys.argv) >= 3:
    print("continue training")
    fcn_model = torch.load(model_path)
    #fcn_model = torch.load("./models/DL3")
    fcn_model = fcn_model.cuda()
else:
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


def train():
    for epoch in range(epochs):


        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 100 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss))
        scheduler.step()
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)

        val(epoch)


def label_to_RGB(image):
    height, weight = image.shape

    rgb = np.zeros((height, weight, 3))
    for h in range(height):
        for w in range(weight):
            rgb[h,w,:] = index2color[image[h,w]]
    return rgb

def oh_2_idx(image):
    label =  [np.where(r==1)[0][0] for r in image]
    return label

def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].add_(means[0])
    img_batch[:,1,...].add_(means[1])
    img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)

    #real images
    grid = utils.make_grid(img_batch)
    plt.figure()
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

    plt.title('Batch from dataloader')

    #ground truth
    plt.figure()
    labels = batch['l'].cpu().numpy()
    for i in range(len(labels)):
        plt.subplot(1, len(labels), i+1)
        plt.imshow(Image.fromarray(label_to_RGB(labels[i,:,:]), 'RGB'), interpolation='nearest')
    plt.title('Labels')

    #output
    outputs = model(batch['X'].cuda())['out']
    output = outputs.data.cpu().numpy()

    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
    plt.figure()
    for i in range(N):
        plt.subplot(1, len(labels), i+1)
        plt.imshow(Image.fromarray(label_to_RGB(pred[i,:,:]), 'RGB'), interpolation='nearest')
    plt.title('Output')
    plt.show()


def val(epoch):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)['out']
        #print(output)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)



# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    train()
