#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:40:46 2021

@author: e321075
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random	
import cv2
import scipy.misc
from skimage.transform import resize
import os
from collections import namedtuple
import re


#############################
	# global variables #
#############################
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
root_dir          = "./../Synthia/"
train_label_file  = os.path.join(root_dir, "train_np_labels.csv") # train file
csv_file = open(train_label_file, "w")
csv_file.write("img,label\n")
for idx, name in enumerate(os.listdir(root_dir)):
    label_dir = os.path.join(root_dir, name)
    for name in os.listdir(label_dir):
        fine_label = os.path.join(label_dir, name)
        print(fine_label)
        SemSeg_dir = os.path.join(fine_label, "SemSeg")
        labels_dir = os.path.join(fine_label, "labels")
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        image_dir = os.path.join(fine_label, "RGB")
        for img in os.listdir(image_dir): 
            img_name = os.path.join(image_dir, img)
            label_name = os.path.join(SemSeg_dir, img)
            image = Image.open(label_name)
            labels = np.asarray(image.convert("RGB"))
            labels = labels[:,:,0]
            height, weight = labels.shape
            label = np.zeros((height,weight))
            label_name = os.path.join(labels_dir, img[:-4])
            for h in range(height):
                for w in range(weight):
                    try:
                        label[h,w] = label_translation[labels[h,w]]
                    except:
                        label[h,w] = 0
            np.save(label_name, label)
            csv_file.write("{},{}\n".format(img_name, label_name))
            
csv_file.close()
