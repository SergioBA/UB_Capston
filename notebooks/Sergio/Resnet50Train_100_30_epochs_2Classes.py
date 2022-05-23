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


#  Configuration
batch_size_param = 16
img_height = 224        
img_width = 224
initial_epochs = 100
test_percent = 0.2

UseWeights = True
BasicModel = False
TrainWithoutTransference = False       # Only initial_epochs and no weight for base model
if BasicModel :
  TrainWithoutTransference = True       # Basic model --> no Predefined Weights
epochsFine = 30
DataAugmentation = True 
ReportName = "Resnet50Train_100_30_epochs_2Classes"

# For 11  Classification
#             Weights      Class                      Elements
# weights = {  
#           0:  0.0285,       # Angiectasia   	        866		     
#           1:  1.,           # Blood - fresh			      446
#           2:  0.1068,       # Erosion					        506
#           3:  0.1667,       # Erythema				        159
#           4:  0.0373,       # Foreign body			      776
#           5:  0.0196,       # Ileocecal valve		     4189 
#           6:  0.0982,       # Lymphangiectasia		    592
#           7:  0.0014,       # Normal clean mucosa   34338    
#           8:  0.0235,       # Pylorus					       1529  
#           9:  0.0236,       # Reduced mucosal view   2906    
#           10: 0.0809        # Ulcer					          854
#         }

# For dataSetOnlyIllness Classification
weights = {
            0: 1.,          # Illness                 4000
            1: 0.1         # Normal Class           37000
          }  
    

data_dir = Path(os.getcwd() + "/../../data/dataSetOnlyIllness/")
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

# TODO: para validar
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  validation_split=test_percent,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size_param)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  validation_split=test_percent,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size_param)

valImages = np.array([])
vallabels =  np.array([])
valDataList = []
vallabelsList = []
for image, label in val_ds.take(-1):
  valDataList.extend(image)
  vallabelsList.extend(label)
vallabels = np.array(vallabelsList)
valImages = np.array(valDataList)


# print("List of class:")
class_names = train_ds.class_names
num_classes = len(class_names)
# for e in class_names:
#     print("    "+e)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(10000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     print(gpu)
#     print("Mem Ext")
#     tf.config.experimental.set_memory_growth(gpu, True)


# Add more images due a Unbalanced input data
data_augmentation = keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(1),
    tf.keras.layers.RandomZoom(1),
  ]
)
if BasicModel:
  model = Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),                                     # Normalization 
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(num_classes)
  ])
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

else:
  #  Resnet model  DOC --->  https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet152V2
  #                          https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50

  if TrainWithoutTransference == False :
      base_model = tf.keras.applications.resnet50.ResNet50(                     # Alternative -> resnet_v2.ResNet152V2  /  resnet50.ResNet50
                                          include_top=True,
                                          weights='imagenet',
                                          input_tensor=None,
                                          input_shape=None,
                                          pooling=None,
                                          classes=1000,
                                          classifier_activation='softmax'
      )
      base_model.trainable = False 
  else :
      base_model = tf.keras.applications.resnet50.ResNet50(                     # Alternative -> resnet_v2.ResNet152V2
                                          include_top=True,
                                          weights=None,
                                          input_tensor=None,
                                          input_shape=None,
                                          pooling=None,
                                          classes=num_classes,
                                          classifier_activation='softmax') 

  if DataAugmentation:
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(img_height, img_width, 3)),
        data_augmentation,                                                                        # Data augmentation
        tf.keras.layers.Rescaling(1./127.5, offset=-1),                                                       # Data rescalation
        base_model,
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
  else:
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(img_height, img_width, 3)),
        tf.keras.layers.Rescaling(1./127.5, offset=-1),                                                       # Data rescalation
        base_model,
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy','crossentropy'])

  # print (model.summary())

# Add class_weight parameter
if UseWeights:
  history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = initial_epochs, batch_size = batch_size_param,
    class_weight = weights
  )
else:
  history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = initial_epochs, batch_size = batch_size_param
  )


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(initial_epochs)

end = time.time()
total_time = end - start
print("Total time train : "+ str(total_time))


loss0, accuracy0 = model.evaluate(valImages,vallabels)
print("Initial loss: {:.2f}".format(loss0))
print("Initial accuracy: {:.2f}".format(accuracy0))



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
plt.savefig(ReportName + '.png')

# Fine tune  Let's to train first 25% layers
if TrainWithoutTransference == False : 

    base_model.trainable = True
    # Fine-tune from this layer onwards 75%
    fine_tune_at = math.trunc(len(base_model.layers) * 0.75)                    #

    print("Number of layers in the base model: ", len(base_model.layers))
    print("First layers to fine train : ", len(base_model.layers) - fine_tune_at)

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False


    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001/10),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy','crossentropy'])


    total_epochs =  initial_epochs + epochsFine

    if UseWeights:
      history_fine = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=total_epochs, 
        batch_size = batch_size_param,
        initial_epoch=history.epoch[-1],
        class_weight = weights
      )
    else:
      history_fine = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=total_epochs, 
        batch_size = batch_size_param,
        initial_epoch=history.epoch[-1],
      )

    loss, accuracy = model.evaluate(val_ds)    
    print('Test accuracy :', accuracy)


    model.save(ReportName + '.h5')
    print("Modelo guardado!")
      
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    end = time.time()
    total_time = end - start
    print("Total time: "+ str(total_time))


    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    plt.savefig(ReportName + '_FineTune.png')

    end = time.time()
    total_time = end - start
    print("Total time train + FineTune : "+ str(total_time))

# Evaluate the model

def test_model(y_true, y_predicted):
    print("Accuracy = {:.3f}".format(mtc.accuracy_score(y_true, y_predicted)))
    print("Accuracy Balanced = {:.3f}".format(mtc.balanced_accuracy_score(y_true, y_predicted)))
    
    print("Precision micro = {:.3f}".format(mtc.precision_score(y_true,y_predicted, average="micro")))
    print("Precision macro = {:.3f}".format(mtc.precision_score(y_true,y_predicted, average="macro")))
    print("Precision weighted = {:.3f}".format(mtc.precision_score(y_true,y_predicted, average="weighted")))
    
    print("Recall micro = {:.3f}".format(mtc.recall_score(y_true, y_predicted, average="micro")))
    print("Recall macro = {:.3f}".format(mtc.recall_score(y_true, y_predicted, average="macro")))
    print("Recall weighted = {:.3f}".format(mtc.recall_score(y_true, y_predicted, average="weighted")))

    print("F1 micro = {:.3f}".format(mtc.f1_score(y_true, y_predicted, average="micro")))
    print("F1 macro = {:.3f}".format(mtc.f1_score(y_true, y_predicted, average="macro")))
    print("F1 weighted = {:.3f}".format(mtc.f1_score(y_true, y_predicted, average="weighted")))

    print("MCC = {:.3f}".format(mtc.matthews_corrcoef(y_true, y_predicted)))
    print("Kappa = {:.3f}".format(mtc.cohen_kappa_score(y_true, y_predicted)))


# Last number from the models
predValImages = model.predict(valImages)
predictionOfValImages = []
for value in predValImages:
  predictionOfValImages.append(np.argmax(value))
test_model(vallabels, predictionOfValImages)
print("Classes not found")
print(set(vallabels) - set(predictionOfValImages))
report = classification_report(vallabels, predictionOfValImages)
print(report)