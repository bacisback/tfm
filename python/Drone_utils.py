#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:40:46 2021

@author: e321075
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import scipy.misc
from skimage.transform import resize
import os
from collections import namedtuple
import re


#############################
	# global variables #
#############################
label_translation= {"unlabeled":0,
                    1:1,#road
                    "paved-area":2, #sidewalk
                    "roof":3,#building
                    "wall":3, #wall
                    "door": 3,
                    "window": 4,
                    "fence": 4,
                    
                    5:3,#billboard
                    "fence-pole":6, #pole
                    7:7, #trafic light
                    8:5, #trafic sign
                    "dirt":8, #vegetation
                    "grass":8,#terrain
                    "water":8,#terrain
                    "rocks":8,#terrain
                    "pool":8,#terrain
                    "vegetation":8,#terrain
                    "tree":8,
                    "bald-tree":8,
                    11:9,#sky
                    "person":10,# pedestrian
                    13:10,# rider
                    "car":11, #car
                    15:12, #truck
                    16:12, #bus
                    17:12, #train
                    18:11, #moto
                    "bicycle":11, #bike
                    20:1, #roadmarks
                    21:0, #unknown
                    }
root_dir          = "./../semantic_drone_dataset/"
training_set      = "training_set/"
training_dir      = os.path.join(root_dir, training_set)
img_dir           = os.path.join(training_dir, "images/")
gt_dir            = os.path.join(training_dir, "gt/semantic/")
class_dict        = pd.read_csv(os.path.join(gt_dir, "class_dict.csv"))
class_label_dict  = {tuple(class_dict.iloc[i,1:].values): class_dict.iloc[i,0]  for i in range(len(class_dict))}
label_dir         = os.path.join(gt_dir, "label_images/")
np_labels         = os.path.join(gt_dir, "np_labels/")
train_label_file  = os.path.join(root_dir, "train.csv") # train file
csv_file = open(train_label_file, "w")
csv_file.write("img,label\n")
for idx, img in enumerate(os.listdir(img_dir)):
    img_name = os.path.join(img_dir, img)
    label_name = os.path.join(label_dir, img[:-3]+"png")
    image = Image.open(label_name)
    labels = np.asarray(image.convert("RGB"))
    height, weight, _ = labels.shape
    label = np.zeros((height,weight))
    for h in range(height):
        for w in range(weight):
            try:
                label[h,w] = label_translation[class_label_dict[tuple(labels[h,w,:])]]
            except:
                label[h,w] = 0
    
    label_name = os.path.join(np_labels, img[:-4]+".npy")
    np.save(label_name, label)
    csv_file.write("{},{}\n".format(img_name, label_name))
            
csv_file.close()
