from csv import reader
from PIL import Image
import os.path, sys
from matplotlib.patches import Rectangle

import time
start = time.time()                       # Time measurement
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from pathlib import Path
import tensorflow_hub as hub
import math 
import sklearn.metrics as mtc
from sklearn.metrics import classification_report



# totalimages = 0
pathMetadaFile = '../../data/metadata.csv'
images2Cut = dict()                                 # Load all information in dictionary
with open(pathMetadaFile, 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    if header != None:
        for row in csv_reader:
                # totalimages = totalimages + 1 
                shields = row[0].split(sep=';')
                if(shields[5] != '' and shields[6] != '' and shields[7] != '' and shields[8] != '' and shields[9] != '' and shields[10] != '' and shields[11] != '' and shields[12] != ''):
                    images2Cut[shields[0]] = {'x1':shields[5] ,'y1':shields[6],'x2':shields[7], 'y2':shields[8],'x3':shields[9],'y3':shields[10],'x4':shields[11],'y4':shields[12]}
# print ("Total Images: " + str(totalimages))
# print("Images 2 Cut: "+ str(len(images2Cut)))
# print(images2Cut)


pathRootFolder2Cut = '../../data/OriginalDataset/'
ListofFolders = os.listdir(pathRootFolder2Cut)
data_dir = Path(os.getcwd() + "/"+pathRootFolder2Cut)
images = data_dir.glob('**/*.jpg')
print("Num images: "+ str(len(list(images))))
# Count Num of elements by Class
initial_count = 0
for specificClass in data_dir.iterdir():
   print(str(specificClass))
   initial_count = 0
   for elementFile in specificClass.iterdir():
       if elementFile.is_file():
           initial_count += 1
   print("Elements: " + str(initial_count))

batch_size_param = 1
img_height = 336            #Less resolution --> No classification
img_width = 336
test_percent = 0.2

train_ds  = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  validation_split=test_percent,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size_param)


plt.figure()
count = 0
found = 0
stopFlag = False
file_paths = train_ds.file_paths
class_names = train_ds.class_names
for image, label in train_ds.take(-1):
    head, tail = os.path.split(file_paths[count])
    if tail in images2Cut:
        if((label[0] == 6)):                            #Foreign Body
            if(found == 23):
                # image = image.crop((0,0,image.size[0],image.size[1]-10))
                plt.imshow(image[0].numpy().astype("uint8"))
                plt.title(tail)
                plt.axis("off")
                print(str(images2Cut[tail]['x1'])+"_"+str(images2Cut[tail]['y1'])+"_"+str(images2Cut[tail]['x2'])+"_"+str(images2Cut[tail]['y2'])+"_"+str(images2Cut[tail]['x3'])+"_"+str(images2Cut[tail]['y3'])+"_"+str(images2Cut[tail]['x4'])+"_"+str(images2Cut[tail]['y4']))
                print(class_names[label[0]])
                plt.gca().add_patch(Rectangle((int(images2Cut[tail]['x1']), int(images2Cut[tail]['y1'])), int(images2Cut[tail]['x2'])-int(images2Cut[tail]['x1']), int(images2Cut[tail]['y3'])-int(images2Cut[tail]['y1']), linewidth=1, edgecolor='r', facecolor='none'))
                stopFlag = True
            found = found + 1
    count = count + 1
    if stopFlag:
        break
plt.show()





# count = 0
# for itemFolder in ListofFolders:                                        #Loop all folders
#     fullpath2Folder2Cut = os.path.join(pathRootFolder2Cut,itemFolder)
#     print("Folder:" + itemFolder)
#     imagesFiles = os.listdir(fullpath2Folder2Cut)
#     for imageFile in imagesFiles:
#         fullpath2Image2Cut = os.path.join(fullpath2Folder2Cut,imageFile)
#         if os.path.isfile(fullpath2Image2Cut):
#             if imageFile in images2Cut:
#                 del images2Cut[imageFile]                               #Apply Crop test_image =     test_image.crop((0,0,test_image.size[0],test_image.size[1]-10))
            # im = Image.open(fullpath2Image2Cut)
            # f, e = os.path.splitext(fullpath2Image2Cut)
            # imCrop = im.crop((30, 10, 1024, 1004)) #corrected
            # imCrop.save(f + 'Cropped.bmp', "BMP", quality=100)
# print ("Images found for cutting: "+str(count))

# print("Images NO Cutted: "+ str(len(images2Cut)))
# print(images2Cut)

