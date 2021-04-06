#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:20:51 2021

@author: e321075
"""

import argparse 
from loader import loader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from MSS_utils import index2color
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os

datanames_csvfiles = {"Cityscapes_test": "./../CityScapes/val.csv",
                      "Kitti": "./../Kitti/training/train.csv"}

total = open("./resultados_comparativos.csv", "w")
total.write("train,test, pix accuracy, meanIoU, unlabeled, road, sidewalk, buildings, complements, billboards, pole, lights,vegetation, sky, person, car, bus\n")

def iou(pred, target):
    ious = []
    for cls in range(13):
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

def val(model, test_data, model_name, test_file):
        model.eval()
        total_ious = []
        pixel_accs = []
        for iter, batch in enumerate(test_data):
            inputs = Variable(batch['X'].cuda())
            
            outputs = model(inputs)
            outputs = outputs["out"]
            output = outputs.data.cpu().numpy()
    
            N, _, h, w = output.shape
            pred = output.transpose(0, 2, 3, 1).reshape(-1, 13).argmax(axis=1).reshape(N, h, w)
    
            target = batch['Y'].cpu().numpy().reshape(N, h, w)
            for p, t in zip(pred, target):
                total_ious.append(iou(p, t))
                pixel_accs.append(pixel_acc(p, t))

        # Calculate average IoU
        total_ious = np.array(total_ious).T  # n_class * val_len
        ious = np.nanmean(total_ious, axis=1)
        pixel_accs = np.array(pixel_accs).mean()
        meanIoU = np.nanmean(ious)
        total.write("{},{},{},{},".format(model_name,test_file,pixel_accs, meanIoU))
        total.write(",".join([str(a) for a in ious]) + "\n")

for model_name in os.listdir("./models"):
    if "imgs" == model_name:
        continue
    model  =  torch.load( os.path.join("./models", model_name))
    model.cuda()
    print(model_name)
    for key in datanames_csvfiles:
         
        data_test = loader(csv_file=datanames_csvfiles[key], phase='test') 
        data_test  = DataLoader(data_test, batch_size=8, shuffle=False, num_workers=4, drop_last=True)
        val(model, data_test, model_name, key)
        
total.close()