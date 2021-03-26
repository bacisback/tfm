
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
img_Denorm = DeNormalize(means, std)

def label_to_RGB(image):

    image = image.squeeze()
    height, weight = image.shape

    rgb = np.zeros((height, weight, 3))
    for h in range(height):
        for w in range(weight):
            try:
                rgb[h,w,:] = index2color[image[h,w]]
            except:
                rgb[h,w,:] = (0, 0, 0)
    return rgb.astype(np.uint8)

def oh_2_idx(image):
    label =  [np.where(r==1)[0][0] for r in image]
    return label

def show_batch(batch, name, model, flag):
    img_batch = batch['X']
    batch_size = len(img_batch)
    #np_batch = img_batch.cpu().numpy().reshape((4,3,224,224))
    plt.figure()

    for i in range(batch_size):
         image_np = (255*(img_Denorm(img_batch[i,...]).data.permute(1,2,0).cpu().numpy())).astype(np.uint8)
         img_pil = Image.fromarray(image_np)
         plt.subplot(2, batch_size/2, i+1)
         plt.imshow(img_pil, interpolation='nearest')
    plt.title('Batch from dataloader')
    plt.savefig(name+"original.png")
    plt.figure()
    labels = batch['Y'].cpu().numpy()

    for i in range(batch_size):
         plt.subplot(2, batch_size/2, i+1)
         plt.imshow(label_to_RGB(labels[i,...]), interpolation='nearest')
    plt.title('Labels')
    plt.savefig(name+"labels.png")
    outputs = model(batch['X'].cuda())
    if flag:
        outputs = outputs['out']
    output = outputs.data.cpu().numpy()

    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, 13).argmax(axis=1).reshape(N, h, w)
    plt.figure()
    for i in range(N):
        plt.subplot(2, len(labels)/2, i+1)
        plt.imshow(label_to_RGB(pred[i,...]), interpolation='nearest')
    plt.title('Output')
    plt.savefig(name+"output.png")

datanames_csvfiles = {"Cityscapes": "./../CityScapes/train.csv",
                      "Coche": './../MSS_vids/data/images/Coche.csv',
                      
                      "Kitti": "./../Kitti/training/train.csv",

                      "Bus": './../MSS_vids/data/images/AutoBus.csv',
                      
                      "Helicoptero": './../MSS_vids/data/images/Helicoptero.csv',
                      "Peaton": './../MSS_vids/data/images/Peaton.csv',
                      "Video": './../MSS_vids/data/images/Video.csv'}


batch_size = 8
epochs     = 5
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
total = open("./resultados.csv", "w")
total.write("train,test, pix accuracy, meanIoU, unlabeled, road, sidewalk, buildings, complements, billboards, pole, lights,vegetation, sky, person, car, bus\n")

class training:
    def __init__(self, train, test, model, plot):
        self.train_file = train
        
        self.test_file = test
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
        print(model_file)
        self.DL3        = "lab" in model
        if os.path.isfile(model_file):  
            self.model  = torch.load(model_file)
            self.model_path = model_file
            print("cargando:", model_file)
        elif os.path.isfile(model_file+"_final_"+train):
            self.model  = torch.load(model_file+"_final_"+train)
            self.model_path = model_file+"_final_"+train
            print("cargando:", model_file+"_final_"+train)
        else:
            if self.DL3:
                self.model  =  torch.load("./models/DLscratch")
            self.model_path = os.path.join("./models/", model)
            print("cargando:", self.model_path)
        if self.use_gpu:
            self.model = self.model.cuda()
        
        self.iou = 0
           

        



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
                #print(outputs, np.min(labels.cpu().numpy()))
                loss = self.criterion(outputs, labels.squeeze(1))
                
                loss.backward()
                optimizer.step()
                

                
                if iter % 50 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss))

            scheduler.step()
            
            print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
            self.val(epoch)
        if "final" in self.model_path:
            torch.save(self.model, self.model_path)
        else:
            torch.save(self.model, self.model_path+"_final_"+self.train_file)
        for batch in self.train_data:
            show_batch(batch, "models/imgs/"+model+self.train_file+self.test_file+"_train_", self.model,self.DL3)
            break
        for batch in self.test_data:
            show_batch(batch, "models/imgs/"+model+self.train_file+self.test_file+"_test_", self.model,self.DL3)
            break
            


    def val(self, epoch):
        self.model.eval()
        total_ious = []
        pixel_accs = []
        for iter, batch in enumerate(self.test_data):
            if iter > 62:
                break
            if self.use_gpu:
                inputs = Variable(batch['X'].cuda())
            else:
                inputs = Variable(batch['X'])
    
            outputs = self.model(inputs)
            if self.DL3:
                outputs = outputs["out"]
            #print(output)
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
        print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, meanIoU, ious))
        if meanIoU > self.iou:
            #torch.save(self.model, self.model_path+"_"+self.train_file+"_"+self.test_file)
            total.write("{},{},{},{},".format(self.train_file,self.test_file,pixel_accs, meanIoU))
            total.write(",".join([str(a) for a in ious]) + "\n")


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
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



        
if __name__ == '__main__':
    """
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
    """
    for model in ['deeplabv3', 'FCN']:
        for key in datanames_csvfiles:
            for key_2 in datanames_csvfiles:
                if key == key_2:
                    continue
                trainer = training(key, key_2, model, 1)
                trainer.train(25)
    total.close()
