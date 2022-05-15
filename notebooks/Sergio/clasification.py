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



#  Configuration
batch_size = 1
img_height = 256
img_width = 256
epochs=120
UseRam = True
UseWeights = True
print("Configuration=> Batch: " + str(batch_size) + " Size: "+ str(img_height) +"X" +  str(img_width) + " Epochs: " + str(epochs))


#                     Weights      Class
class_weight = {  
                  0:  0.2,         # Ampulla of vater         
                  1:  0.0285,      # Angiectasia   			          
                  2:  1.,          # Blood - fresh			    
                  3:  0.2,         # Blood - hematin         
                  4:  0.1068,      # Erosion					      
                  5:  0.1667,      # Erythema				        
                  6:  0.0373,      # Foreign body			      
                  7:  0.0196,      # Ileocecal valve		     
                  8:  0.0982,      # Lymphangiectasia		    
                  9:  0.0014,      # Normal clean mucosa        
                  10: 0.2,         # Polyp                     
                  11: 0.0235,      # Pylorus					        
                  12: 0.0236,      # Reduced mucosal view       
                  13: 0.0809       # Ulcer					        
                }

data_dir = Path(os.getcwd() + "/../../data/labelled_images/")
images = data_dir.glob('**/*.jpg')
# print("Num images: "+ str(len(list(images))))

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# print("List of class:")
class_names = train_ds.class_names
num_classes = len(class_names)
# for e in class_names:
#     print("    "+e)

if UseRam:
  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
else:
  train_ds = train_ds.shuffle(1000)

# Add more images due a Unbalanced input data
data_augmentation = keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal_and_vertical",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    tf.keras.layers.RandomRotation(1),
    tf.keras.layers.RandomZoom(1),
  ]
)


model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),                                     # Normalization 
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(256, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.1),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Add class_weight parameter
if UseWeights:
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weight
  )
else:
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

model.save('testclasification.h5')
print("Modelo guardado!")

end = time.time()
total_time = end - start
print("Total time: "+ str(total_time))


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# -------------------------------






