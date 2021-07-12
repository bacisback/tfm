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


#############################
	# global variables #
#############################
root_dir          = "./../MSS_vids/"
video_dir         = os.path.join(root_dir, "Videos")    # videos
data_dir          = os.path.join(root_dir, "data5/")      # Data for train and test
images_dir        = os.path.join(data_dir, "images/")    # train label
label_dir         = os.path.join(data_dir, "labels/")    # train label
train_label_file  = os.path.join(data_dir, "train.csv") # train file
frame_rate        = 25

Label = namedtuple('Label', [
				   'name', 
				   'id', 
				   'color'])

labels = [ # name                      id       color
	Label(  'unlabeled'              ,  0  , (  0,  0,  0)       ),
	Label(  'road2'                   ,  1  , (  109,  95,  144)  ),    
	Label(  'road'                   ,  1  , (  102,  51,  153)  ),
	Label(  'sidewalk_2'             ,  2  , (  160,  10,   90)  ),
	Label(  'sidewalk'               ,  2  , (  255,  20,  147)  ),
     Label(  'buildings_2'            ,  3  , (  85,    103,  117)  ),
	Label(  'buildings_2'            ,  3  , (  63,    72,  67)  ),
	Label(  'buildings'              ,  3  , (  119,  136,  153) ),
	Label(  'buildings (complements)',  4  , (  112,  128,  144) ),
     Label(  'buildings2(complements)',  4  , (  140,  166,  187) ),
	Label(  'billboards'             ,  5  , (  188,  143,  143) ),
	Label(  'pole'                   ,  6  , (  169,  169,  169) ),
	Label(  'traffic light'          ,  7  , (  255,  255,  0)   ),
	Label(  'vegetation_2'           ,  8  , (  13,  75,    8)   ),
	Label(  'vegetation'             ,  8  , (  34,  139,  34)   ),
	Label(  'sky'                    ,  9  , (  0,  191,  255)   ),
     Label(  'person'                 ,  10 , (  133,  0,  0)     ),
     Label(  'person'                 ,  10 , (  162,  20,  36)     ),
	Label(  'person'                 ,  10 , (  255,  0,  0)     ),
     Label(  'car2'                   ,  11 , (  0,  1,  65)     ),
	Label(  'car'                    ,  11 , (  0,  0,  128)     ),
	Label(  'bus_2'                  ,  12 , (  0,    68,  63)   ),
	Label(  'bus'                    ,  12 , (  0,  128,  128)   )
	]

color2index = {}
index2color = {19:(  0,  0,  128)}
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
def trim_video():
	
	
	for idx, name in enumerate(os.listdir(video_dir)):
		if 'avi' not in name or "RGB" not in name:
			#print(name)
			continue

		print(name)
		filename_dat = os.path.join(video_dir, name)
		filename_sem = "s" + name[1:-7] + "Semantica.avi" #los videos en RGB vienen con Semantica en mayusculas y acabados en _RGB
		filename_sem = os.path.join(video_dir, filename_sem)

		video_category = name[9:-8] # Suponemos nombres de videos en el modo Sequencia+tipo+_RGB/Semantica.avi
		video_category = re.sub(r'[0-9]+', '', video_category)

		category_dir_images = os.path.join(images_dir, video_category)
		semantic_dir = os.path.join(category_dir_images, 'semantics/')
		if not os.path.exists(category_dir_images):
			os.makedirs(category_dir_images)
			os.makedirs(semantic_dir)
		category_dir_labels = os.path.join(label_dir, video_category)
		if not os.path.exists(category_dir_labels):
			os.makedirs(category_dir_labels)

		i = 0
		
		cap= cv2.VideoCapture(filename_sem)

		while(cap.isOpened()):
			ret, frame = cap.read()
			if i%frame_rate != 0:
				i+=1
				continue
			
			if ret == False:
				break
			
			image_name = os.path.join(semantic_dir, name[9:-8] + str(i))
			cv2.imwrite(image_name + '.jpg',frame)
			i+=1
 
		cap.release()
		cv2.destroyAllWindows()
		
		i = 0
		cap= cv2.VideoCapture(filename_dat)
		while(cap.isOpened()):
			ret, frame = cap.read()
			if i%frame_rate != 0:
				i+=1
				continue
			if ret == False:
				break
			image_name = os.path.join(category_dir_images, name[9:-8] + str(i))
			cv2.imwrite(image_name + '.jpg',frame)
			i+=1
 
		cap.release()
		cv2.destroyAllWindows()
	#t.close()

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
def parse_labels():
	total = open(train_label_file, "w")
	total.write("img,label\n")

	for video_category in os.listdir(images_dir):

		#get category folder and semantic folder.
		category_dir_images = os.path.join(images_dir, video_category)
		semantic_dir = os.path.join(category_dir_images, 'semantics/')

		#get the label folder to be allocated, create if needed.
		category_dir_labels = os.path.join(label_dir, video_category)
		if not os.path.exists(category_dir_labels):
			os.makedirs(category_dir_labels)

		#open class specific csv
		category_file = os.path.join(images_dir, video_category + ".csv")
		category_csv = open(category_file, "w")
		category_csv.write("img,label\n")


		#create label matrix
		for name in os.listdir(category_dir_images):
			if name == "semantics":
				continue
			filename_img = os.path.join(category_dir_images, name)
			filename_sem = os.path.join(semantic_dir, name)
			try:
				frame = np.asarray(Image.open(filename_sem).convert('RGB'))
			except:
				continue

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
			"""
			plt.subplot(1,2,1)
			plt.imshow(corrected, interpolation='nearest')
			plt.subplot(1,2,2)
			plt.imshow(frame, interpolation='nearest')
			plt.show()
			)"""
			corrected = Image.fromarray(label_to_RGB(idx_mat), 'RGB')

			cv2.imwrite(filename_sem,np.asarray(corrected))
			idx_mat = idx_mat.astype(np.uint8)
			label_name = os.path.join(category_dir_labels, name)
			np.save(label_name, idx_mat)
			total.write("{},{}\n".format(filename_img, label_name + '.npy'))
			category_csv.write("{},{}\n".format(filename_img, label_name + '.npy'))
		category_csv.close()
	total.close()
if __name__ == '__main__':
	trim_video()
	parse_labels()
