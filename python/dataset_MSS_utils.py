from __future__ import print_function


from matplotlib import pyplot as plt
import numpy as np
import random    
import cv2
import scipy.misc
from PIL import Image
import os
from collections import namedtuple
from matplotlib import pyplot as plt
import re
from scipy.ndimage import generic_filter
from scipy import stats


def modal(P):
    """We receive P[0]..P[8] with the pixels in the 3x3 surrounding window"""
    mode = stats.mode(P)
    return mode.mode[0]



#############################
    # global variables #
#############################
root_dir          = "./../MSS/"
video_dir         = os.path.join(root_dir, "MSSdataset/")    # videos
data_dir          = os.path.join(root_dir, "data/")      # Data for train and test
images_dir        = os.path.join(data_dir, "images/")    # train label
label_dir         = os.path.join(data_dir, "labels/")    # train label
train_label_file  = os.path.join(data_dir, "train.csv") # train file


Label = namedtuple('Label', [
                   'name', 
                   'id', 
                   'color'])

labels = [ # name                      id       color
    Label(  'unlabeled'              ,  0  , (  -3340,  -253250,  -2352350)       ),
    Label(  'road2'                  ,  1  , (  109,  95,  144)  ),
    Label(  'road2'                  ,  1  , (  114,  98,  148)  ),
    Label(  'road2'                  ,  1  , (  144,  44,   90)  ),
    Label(  'road2'                  ,  1  , (  51,  22,   78)  ),
    Label(  'road2'                  ,  1  , (  76,  61,   92)  ),       
    Label(  'road'                   ,  1  , (  102,  51,  153)  ),
    Label(  'sidewalk2'              ,  2  , (  144,  20,  229)  ),
    Label(  'sidewalk2'              ,  2  , (  140,  3,  71)  ),
    Label(  'sidewalk'               ,  2  , (  255,  20,  147)  ),
    Label(  'buildings_2'            ,  3  , (  74,    90,  105)  ),
    Label(  'buildings_2'            ,  3  , (  85,    103,  117)  ),
    Label(  'buildings_2'            ,  3  , (  61,    70,  77)  ),
    Label(  'buildings_2'            ,  3  , (  63,    72,  67)  ),
    Label(  'buildings'              ,  3  , (  119,  136,  153) ),
    Label(  'buildings (complements)',  3  , (  112,  128,  144) ),
    Label(  'buildings (complements)',  3  , (  113,  128,  147) ),
    Label(  'buildings2(complements)',  3  , (  140,  166,  187) ),
    Label(  'billboards'             ,  5  , (  188,  143,  143) ),
    Label(  'pole'                   ,  6  , (  198,  159,  179) ),
    Label(  'pole'                   ,  6  , (  169,  169,  176) ),
    Label(  'traffic light'          ,  7  , (  196,  209,  129)   ),
    Label(  'traffic light'          ,  7  , (  255,  255,  0)   ),
    Label(  'vegetation_2'           ,  8  , (  13,  75,    8)   ),
    Label(  'vegetation'             ,  8  , (  34,  139,  34)   ),
    Label(  'sky'                    ,  9  , (  0,  191,  255)   ),
    Label(  'person'                 ,  10 , (  133,  0,  0)     ),
    Label(  'person'                 ,  10 , (  162,  20,  36)     ),
    Label(  'person'                 ,  10 , (  255,  0,  0)     ),
    Label(  'car2'                   ,  11 , (  4,  20,  82)     ),
    Label(  'car2'                   ,  11 , (  0,  1,  65)     ),
    Label(  'car2'                   ,  11 , (  60,  80,  126)     ),
    Label(  'car'                    ,  11 , (  0,  0,  128)     ),
    Label(  'bus_2'                  ,  12 , (  0,    68,  63)   ),
    Label(  'bus'                    ,  12 , (  0,  128,  128)   )
    ]

color2index = {}
index2color = {}
colors = []
id_list = []

for obj in labels:
        idx   = obj.id
        label = obj.name
        color = obj.color
        colors.append(color)
        color2index[color] = idx
        index2color[idx] = color
        id_list.append(idx)
colors = np.array(colors)
for dir in [video_dir, data_dir, label_dir, images_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)
print(index2color)
def label_to_RGB(image):
    height, weight = image.shape

    rgb = np.zeros((height, weight, 3), dtype=np.uint8)
    for h in range(height):
        for w in range(weight):
            rgb[h,w,:] = index2color[image[h,w]]
    return rgb

def color_dist(color, colors):
    #print(color, colors)
    """rmean = (color[0]-colors[:,0])/2
    r = color[0]-colors[:,0]
    g = color[1]-colors[:,1]
    b = color[2]-colors[:,2]
    results = np.sqrt((((512+rmean)*r*r) //18) + 4*g*g + (((767-rmean)*b*b)//16))"""
    results = np.sqrt((color[0]-colors[:,0])**2+(color[2]-colors[:,2])**2+(color[1]-colors[:,1])**2)
    #print(results)
    return id_list[np.argmin(results)]

def parse_label(frame):
    height, weight, _ = frame.shape
    idx_mat = np.zeros((height, weight))

    for h in range(height):
        for w in range(weight):
            color = list(frame[h, w])
            min_ind = color_dist(color, colors)
            try:
                #index = color2index[color]
                idx_mat[h, w] = min_ind
            except:
                # no index, assign to void
                idx_mat[h, w] = 0
    #idx_mat = generic_filter(idx_mat, modal, (3, 3))
    """
    plt.subplot(1,2,1)
    plt.imshow(Image.fromarray(label_to_RGB(idx_mat), 'RGB'), interpolation='nearest')
    plt.subplot(1,2,2)
    plt.imshow(Image.fromarray(frame, 'RGB'), interpolation='nearest')
    plt.show()
    """
    return idx_mat


def trim_video():
    
    
    for category in os.listdir(video_dir):
        
        category_dir = os.path.join(video_dir, category)
        img_category = os.path.join(images_dir, category)
        sem_category = os.path.join(label_dir, category)

        if not os.path.exists(img_category):
            os.makedirs(img_category)
            os.makedirs(sem_category)

        for amount_cars in os.listdir(category_dir):
            curriculum_dir = os.path.join(category_dir, amount_cars)
            img_category_cv = os.path.join(img_category, amount_cars)
            sem_category_cv = os.path.join(sem_category, amount_cars)

            if not os.path.exists(img_category_cv):
                os.makedirs(img_category_cv)
                os.makedirs(sem_category_cv)
                os.makedirs(sem_category_cv+"/unprocessed/")

            for video in os.listdir(curriculum_dir):
                print(video)
                if "RGB" not in video or os.path.exists(os.path.join(sem_category_cv, video[:-8] + "_0.npy")):
            
                    continue
                #print(os.path.join(sem_category_cv, video[:-8] + "_0.jpg"))
                filename_dat = os.path.join(curriculum_dir, video)
                filename_sem = os.path.join(curriculum_dir, video[:-7]+ "Semantica.avi")
                """
                i = 0
                cap= cv2.VideoCapture(filename_sem)

                while(cap.isOpened()):
                    
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    if os.path.exists(os.path.join(sem_category_cv, video[:-8] + "_"+ str(i) +".npy")):# or i%2!=0: #i%11 != 0: or
                        i += 1
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    labels = parse_label(frame)
                    image_name = os.path.join(sem_category_cv, video[:-8] + "_" + str(i))
                    img = Image.fromarray(label_to_RGB(labels), 'RGB')
                   
                    #real_name = os.path.join(img_category_cv, video[:-8] + "_" +str(i))
                    np.save(image_name, labels)
                    img.save(image_name + '.jpg')
                    cv2.imwrite(os.path.join(sem_category_cv,"unprocessed/" + video[:-8] + "_" + str(i)) + '.jpg',cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    #total.write("{},{}\n".format(real_name+ '.jpg', image_name + '.npy'))
                    i+=1
                    
                        
                cap.release()
                cv2.destroyAllWindows()
                """
                i = 0
                cap= cv2.VideoCapture(filename_dat)
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    if os.path.exists(os.path.join(sem_category_cv, video[:-8] + "_"+ str(i) +".jpg")):
                        image_name = os.path.join(img_category_cv, video[:-8] + "_" +str(i))
                        cv2.imwrite(image_name + '.jpg',frame)
                    i+=1
         
                cap.release()
                cv2.destroyAllWindows()
                

def create_csv():
    total = open(train_label_file, "w")
    total.write("img,label\n")
    for category in os.listdir(images_dir):
        
        img_category = os.path.join(images_dir, category)
        sem_category = os.path.join(label_dir, category)

        for amount_cars in os.listdir(img_category):
            img_category_cv = os.path.join(img_category, amount_cars)
            sem_category_cv = os.path.join(sem_category, amount_cars)
            for img in os.listdir(img_category_cv):
                real_name = os.path.join(img_category_cv, img)
                label_name = os.path.join(sem_category_cv, img[:-3]+ "npy")
                if os.path.isfile(real_name) and os.path.isfile(label_name):
                    total.write("{},{}\n".format(real_name, label_name))
    total.close()
                
if __name__ == '__main__':
    trim_video()
    create_csv()
    
    
