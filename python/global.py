
import argparse 
from loader import loader
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

datanames_csvfiles = {"MSS": './../MSS_vid/data/train.csv',
                      "Kitti": "./../Kitti/training/train.csv",
                      "Cityscapes": "./../CityScapes/train.csv"}

batch_size = 8
epochs     = 5
lr         = 1e-4
momentum   = 0
w_decay    = 2.4e-5
step_size  = 50
gamma      = 0.5


class training:
    def __init__(self, train, test, model, plot):
        train_data = loader(csv_file=datanames_csvfiles[train],
                                  phase='train')
        test_data  = loader(csv_file=datanames_csvfiles[test],
                                  phase='test')                      
        self.train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_data  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        self.use_gpu    = torch.cuda.is_available()
        self.num_gpu    = list(range(torch.cuda.device_count()))
      
        weights = torch.Tensor([[0.9999, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9439, 0.1106, 1.0000,
         1.0000, 0.9999, 1.0000, 0.9455]])

        self.criterion = nn.CrossEntropyLoss(weight=weights.cuda())
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
                self.model  =  torch.load("./models/DLscratch")
            self.model_path = os.path.join("./models/", model)
        if self.use_gpu:
            self.model = self.model.cuda()
           

        



    def train(self, epochs):
        optimizer = optim.RMSprop(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a

        for epoch in range(epochs):

            ts = time.time()
            for iter, batch in enumerate(self.train_data):
                optimizer.zero_grad()

                inputs, labels = Variable(batch['X'].cuda()), Variable(batch['Y'].cuda())

                outputs = self.model(inputs)
                if self.DL3:
                    outputs = outputs["out"]
                print(outputs, np.min(labels.cpu().numpy()))
                loss = self.criterion(outputs, labels.squeeze(1))
                
                loss.backward()
                optimizer.step()
                

                
                if iter % 100 == 0:
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
