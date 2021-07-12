#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:10:57 2021

@author: e321075
"""

from loader import loader
import numpy as np
from torch.utils.data import DataLoader

def num_pixels_per_class(loader):
    per_class = {i:0 for i in range(13)}
    n = len(loader)
    for iter, batch in enumerate(loader):
        
        imgs = batch['Y'].cpu().numpy()
        N, _, h, w = imgs.shape
        imgs = imgs.reshape(N, h, w)
        #print(N,h,w)
        for cls in range(13):
            inds = imgs == cls
            #print(inds.sum(), N*h*w)
            per_class[cls] += inds.sum()/(N*h*w*n)
    print(np.sum(list(per_class.values())))
    total.write(",".join([str(a) for a in per_class.values()]) + "\n")

datanames_csvfiles = {"Coche": './../MSS_vids/data/images/Coche.csv',
                      "Bus": './../MSS_vids/data/images/AutoBus.csv',
                      "Helicoptero": './../MSS_vids/data/images/Helicoptero.csv',
                      "Peaton": './../MSS_vids/data/images/Peaton.csv',
                      "Video": './../MSS_vids/data/images/Video.csv'
                      }
total = open("./results/pixels_per_class_MSS.csv", "w")
total.write("dataset, unlabeled, road, sidewalk, buildings, complements, billboards, pole, lights,vegetation, sky, person, car, bus\n")

"""
for key in datanames_csvfiles:
    total.write("{},".format(key))
    data_test = loader(csv_file=datanames_csvfiles[key], phase='test') 
    data_test  = DataLoader(data_test, batch_size=8, shuffle=False, num_workers=4, drop_last=True)
    num_pixels_per_class(data_test)
"""
data_test = loader(csv_file="./../MSS/data/train.csv", phase="test")#"./../MSS/data/train.csv", phase='test') 
data_test  = DataLoader(data_test, batch_size=8, shuffle=False, num_workers=4, drop_last=True)
num_pixels_per_class(data_test)
total.close()