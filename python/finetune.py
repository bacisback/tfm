
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
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
means=np.array([86.5628,86.6691,86.7348]) / 255
std=[0.229, 0.224, 0.225]

datanames_csvfiles = {"Cityscapes": "./../CityScapes/train.csv",
                      "Cityscapes_test": "./../CityScapes/test.csv",
                      "Coche": './../MSS_vids/data/images/Coche.csv',
                      "Synthia": './../Synthia/train_np_labels.csv',
                      "Kitti": "./../Kitti/training/train.csv",
                      "MSS_complete": "./../MSS/data/train.csv",
                      "MSS": './../MSS_vids/data/train.csv',
                      "Bus": './../MSS_vids/data/images/AutoBus.csv',
                      "Dron": './../semantic_drone_dataset/train.csv',
                      "Helicoptero": './../MSS_vids/data/images/Helicoptero.csv',
                      "Peaton": './../MSS_vids/data/images/Peaton.csv',
                      "Video": './../MSS_vids/data/images/Video.csv',
                      "MSS5": './../MSS_vids/data5/train.csv',
                      "Coche5": './../MSS_vids/data5/images/Coche.csv',
                      "Bus5": './../MSS_vids/data5/images/AutoBus.csv',
                      "Helicoptero5": './../MSS_vids/data5/images/Helicoptero.csv',
                      "Peaton5": './../MSS_vids/data5/images/Peaton.csv',
                      "Video5": './../MSS_vids/data5/images/Video.csv',
                      "Mapilliary": './../Mapilliary/train.csv'}

complete_datasets = ["Kitti",  "Cityscapes","Mapilliary", "Synthia"]
real_datasets_train = [ "Cityscapes","Mapilliary", "Kitti"] #"Dron""Kitti",","Mapilliary"
synthetic_datasets = ["Helicoptero5",  "MSS", "Coche", "Bus", "Peaton", "Video", "Helicoptero", "MSS5", "Coche5", "Bus5", "Peaton5", "Video5" ] #"Synthia",
proportions = [0.05, 0.15, 0.25]#

batch_size = 8
epochs     = 5
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 10
gamma      = 0.5
file_name = "./resultados_synthia.csv"



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


class training:
    def __init__(self, train, test, model, finetune, val=True):
        self.max = 0
        self.train_file = train
        self.progress_file = open("def.csv", "w")
        self.test_file = test
        self.doval=val
        if isinstance(train, dict):
            self.train_file = ""
            train_dat = {}
            for key in train:
                self.train_file += str(key) + "-" + str(train[key]) + "_"
                train_dat[datanames_csvfiles[key]]= train[key]
            train_data = loader(csv_file=train_dat,
                                phase='train')
            if test == "":
                test_data1 = loader(csv_file="./../CityScapes/val.csv",
                                phase='test')
                test_data2 = loader(csv_file='./../Mapilliary/val.csv',
                                phase='test')
                test_data3 = loader(csv_file='./../Kitti/training/train.csv',
                                    phase='test')
            else:
                test_data  = loader(csv_file=test,
                                phase='test') 
        else:
            train_data = loader(csv_file=datanames_csvfiles[train],
                                      phase='train')
            test_data  = loader(csv_file=datanames_csvfiles[test],
                                      phase='test')                      
        self.train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        if test == "":
            self.test_data = [DataLoader(test_data1, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True),
                               DataLoader(test_data2, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True),
                               DataLoader(test_data3, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)]
        else:   
            self.test_data  = [DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)]
        self.use_gpu    = torch.cuda.is_available()
        self.num_gpu    = list(range(torch.cuda.device_count()))
        

        weights = torch.Tensor([1., 0.60295847, 0.9058912 , 0.65019269, 1.,
                   0.98816304, 0.99767979, 0.99643363, 0.93523229, 0.95909425,
                   0.99920763, 0.97340155, 0.99174546])

        self.criterion = nn.CrossEntropyLoss(weight=weights.cuda(), ignore_index=0)
        self.model_dir  = "./models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.DL3        = "lab" in model
        if finetune:
            self.finetunefile = os.path.join(self.model_dir, "deeplab/finetune/")
            self.model_file   = os.path.join(self.model_dir, "deeplab/25epoch/") 
            print("finetuning!")
        else:
        
            if self.DL3:
                model_file      = os.path.join(self.model_dir, "deeplab/25epoch/")    
            else:
                model_file      = os.path.join(self.model_dir, "FCN/")
            model_file      = os.path.join(model_file, self.train_file)
            print(model_file)
            if os.path.exists(model_file):
                self.model  = torch.load(model_file)
                self.model_path = model_file
                print("cargando:", model_file)
            elif os.path.exists(model_file+"_final_"+self.train_file):
                self.model  = torch.load(model_file+"_final_"+self.train_file)
                self.model_path = model_file+"_final_"+self.train_file
                print("cargando:", model_file+"_final_"+self.train_file)
            elif os.path.exists(os.path.join(self.model_dir, "deeplab/")+"deeplabv3"+self.train_file):
                self.model  = torch.load(os.path.join(self.model_dir, "deeplab/")+"deeplabv3"+self.train_file)
                self.model_path = model_file
                print("cargando:", os.path.join(self.model_dir, "deeplab/")+"deeplabv3"+self.train_file)
            else:
                if self.DL3:
                    self.model  =  torch.load("./models/deeplabv3")
                else: 
                    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
                    fcn_model = FCNs(pretrained_net=vgg_model, n_class=13)
                    vgg_model = vgg_model.cuda()
                    fcn_model = fcn_model.cuda()
                    self.model = nn.DataParallel(fcn_model, device_ids=self.num_gpu)
                self.model_path = model_file
                print("cargando from scratch:", self.model_path)
            if self.use_gpu:
                self.model = self.model.cuda()
        
        self.iou = 0
           

        



    def train(self, epochs, model_file=None):
        if model_file is None:
            model_file = self.model_path
        optimizer = optim.RMSprop(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a
        #optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if self.doval:
            self.val(-1, model_file)
        for epoch in range(epochs):
            
            ts = time.time()
            for iter, batch in enumerate(self.train_data):
                optimizer.zero_grad()

                inputs, labels = Variable(batch['X'].cuda()), Variable(batch['Y'].cuda())
                #print(inputs.size())
                outputs = self.model(inputs)
                if self.DL3:
                    outputs = outputs["out"]
                #print(outputs, np.min(labels.cpu().numpy()))
                loss = self.criterion(outputs, labels.squeeze(1))
                
                loss.backward()
                optimizer.step()
                

                
                if iter % 500 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss))

            scheduler.step()
            #optimizer.step()
            #torch.save(self.model, model_file)
            print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
            if self.doval:
                self.val(epoch, model_file)
        #self.val(epoch, model_file)
        """
        for batch in self.train_data:
            show_batch(batch, "models/imgs/train/"+self.train_file, self.model,self.DL3)
            break
        for batch in self.test_data:
            show_batch(batch, "models/imgs/test/"+self.train_file, self.model,self.DL3)
            break
        """ 
    def finetune(self, epochs):
        for model_name in os.listdir(self.model_file):
            if "-" not in model_name  or "Mapilliary" in model_name or "Cityscapes" in model_name:
                continue
            model = os.path.join(self.model_file, model_name)        
            self.model = torch.load(model)
            if self.use_gpu:
                self.model = self.model.cuda()
            model_path = os.path.join(self.finetunefile,model_name + self.train_file)
            self.progress_file = model_path+".csv"
            self.progress_file = open(self.progress_file, "w")
            self.progress_file.write("epoch, loss, pix_acc, meanIoU\n")
            self.train(epochs, model_path)
            self.progress_file.close()

    def val(self, epoch, model_file):
        self.model.eval()
        performance = 0
        for test in self.test_data:
            total_ious = []
            pixel_accs = []
            for iter, batch in enumerate(test):
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
            performance += pixel_accs
            meanIoU = np.nanmean(ious[1:])
            print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, meanIoU, ious))
            #torch.save(self.model, self.model_path+"_"+self.train_file+"_"+self.test_file)
            total.write("{},{},{},{},".format(self.train_file,self.test_file,pixel_accs, meanIoU))
            total.write(",".join([str(a) for a in ious]) + "\n")
            self.progress_file.write(str(epoch)+","+ str(pixel_accs)+","+str(meanIoU)+"\n")
        if performance >= self.max:
            self.max = performance
            torch.save(self.model, model_file)

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
    total   = (target == target).sum() - (target == 0).sum()
    return correct / total



        
if __name__ == '__main__':
    """
    total = open("./results/Mix.csv", "w")
    total.write("train,test, pix accuracy, meanIoU, unlabeled, road, sidewalk, buildings, complements, billboards, pole, lights,vegetation, sky, person, car, bus\n")
    train = {}
    train["MSS_complete"] = 1
    train["Synthia"] = 1
    trainer = training(train,  "", 'deeplabv3', False, True)
    trainer.train(45)
    total.close()
   
    total = open("./results/Synthia_complete_mixed.csv", "w+")
    for dataset in real_datasets_train:
        train = {"Synthia":1}
        for proportion in proportions:
            if dataset == "Cityscapes":
                continue
            train[dataset] = proportion
            total.write(dataset+str(proportion)+"\n")
            train[dataset] = proportion
            trainer = training(train,  "", 'deeplabv3', False, True)
            trainer.train(5)
    total.close()
    """
    total = open("./results/Finetuning.csv", "w+")
    for dataset in real_datasets_train:
        train = {}
        for proportion in proportions:
            train[dataset] = proportion
            trainer = training(train,  "", 'deeplabv3', True, True)
            trainer.finetune(45)  
    total.close()
