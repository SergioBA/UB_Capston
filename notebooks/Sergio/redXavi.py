import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

plt.style.use('ggplot')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
import os

dir = os.getcwd() + "/../../data/reduced_example_split/"

print ("Data dir: " + dir )
def load_images_from_folder(folder):
    images = []
    
    folder1 = folder + '/Ileo-cecal valve'
    for filename in os.listdir(folder1)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder1,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)
            
    folder2 = folder + '/Foreign Bodies'
    for filename in os.listdir(folder2)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder2,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)
    
    folder3 = folder + '/Pylorus'
    for filename in os.listdir(folder3)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder3,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)
            
    folder4 = folder + '/Erythematous'
    for filename in os.listdir(folder4)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder4,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)
            
    folder5 = folder + '/Reduced Mucosal View'
    for filename in os.listdir(folder5)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder5,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)
    
    folder6 = folder + '/Blood'
    for filename in os.listdir(folder6)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder6,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)
    
    folder7 = folder + '/Normal'
    for filename in os.listdir(folder7)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder7,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)
    
    folder8 = folder + '/Angiectasia'
    for filename in os.listdir(folder8)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder8,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)

    folder9 = folder + '/Lymphangiectasia'
    for filename in os.listdir(folder9)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder9,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)
            
    folder10 = folder + '/Ulcer'
    for filename in os.listdir(folder10)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder10,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)

    folder11 = folder + '/Erosion'
    for filename in os.listdir(folder11)[:5]:
        img = cv2.imread(os.path.join(os.getcwd(),folder11,filename))
        img = cv2.resize(img,(168,168))
        if img is not None:
            images.append(img)
    
    return images


images = load_images_from_folder(dir +'/split_0/')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

TRAINING_DIR = dir +"/split_0/"
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    class_mode='categorical',
                                                    target_size=(168, 168))

validation_datagen = ImageDataGenerator(rescale=1./255)

VALIDATION_DIR = dir +"/split_1/"
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              class_mode='categorical',
                                                              target_size=(168, 168))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(168, 168, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(11, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


historial = model.fit(train_generator,
                              epochs=5,
                              verbose=1,
                              validation_data=validation_generator)

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])


