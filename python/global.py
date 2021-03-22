
import argparse 
from Cityscapes_loader import CityScapesDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os

datanames_csvfiles = {"MSS": './../MSS_vids/data/train.csv',
                      "Kitti": "./../Kitti/train.csv",
                      "Cityscapes": "./../CityScapes/train.csv"}



class training:
    def __init__(self, train, test, model, plot):
        self.train_data = CityScapesDataset(csv_file=datanames_csvfiles[train],
                                  phase='train')
        #self.test_data  = DataSet(csv_file=datanames_csvfiles[test],
        #                          phase='test')
        self.use_gpu    = torch.cuda.is_available()
        self.num_gpu    = list(range(torch.cuda.device_count()))

        self.model_dir  = "./models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_file      = os.path.join(self.model_dir, model)
        self.DL3        = "lab" in model
        if os.path.isfile(model_file):  
            self.model  = torch.load(model_file)
            self.model_path = model_file
        else:
            if self.DL3:
                self.model  =  torch.load("./models/deeplabv3scratch")
            self.model_path = os.path.join("./models/", model)
        if self.use_gpu:
            self.model = self.model.cuda()

        



    def train(self, epochs):
        for epoch in range(epochs):

            ts = time.time()
            for iter, batch in enumerate(self.train_data):
                optimizer.zero_grad()

                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

                outputs = self.model(inputs)
                if self.DL3:
                    outputs = outputs["out"]
                loss = criterion(outputs, labels.cuda())
                loss.backward()
                optimizer.step()
                

                
                if iter % 1000 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss))

            scheduler.step()
            
            print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
            torch.save(self.model, self.model_path)

            self.val(epoch)


    def val(self):
        return



        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', help='Dataset a utilizar para entrenar',
                        default="MSS")
    parser.add_argument('--test_file', help='Dataset a utilizar para test',
                        default="Kitti")
    parser.add_argument('-m', '--model', help='Modelo a usar o archivo guardado\
                        :\n\tFCN\n\tdeeplabv3',
                        default='deeplabv3')
    parser.add_argument('-p', '--plot', help='Flag para generar imagenes resultado\
                        tras validar:\n\t1 para activar\n\t0 para desactivar',
                        default='1')
    args = parser.parse_args()
    trainer = training(args.train_file, args.test_file, args.model, args.plot)
    trainer.train(500)
