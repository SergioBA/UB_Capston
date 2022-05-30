# Execute -->  cd /home/sergio/Postgrado/UB_Capston/notebooks/Sergio/ ; python3   Train_Capston.py
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
import sklearn.metrics as mtc
from sklearn.metrics import classification_report


#ReportName
ReportName = "Basic_dataSetdataClassIllness_300px"
data_dir = Path(os.getcwd() + "/../../data/dataClassIllnessExtend/")

#  Dataset Configuration
batch_size_param = 16
img_height = 240            #Max resolution 300     
img_width = 240
initial_epochs = 350
test_percent = 0.2
initial_seed=42

# OnlyCPU = True
# if OnlyCPU:
#   os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  # shuffle=True,
  validation_split=test_percent,
  color_mode = "rgb",
  subset="training",
  seed=initial_seed,
  image_size=(img_height, img_width),
  interpolation = "bilinear",
  follow_links = False,
  crop_to_aspect_ratio = False,
  batch_size=batch_size_param)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  # shuffle=True,
  validation_split=test_percent,
  color_mode = "rgb",
  subset="validation",
  seed=initial_seed,
  image_size=(img_height, img_width),
  interpolation = "bilinear",
  follow_links = False,
  crop_to_aspect_ratio = False,
  batch_size=batch_size_param)

#Load images for validation
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
num_classes = len(class_names)        #Number of classes
weights_temp = dict()                 # To load className and ClassNumber
DataSetPrintInfo = dict()
count = 0
for e in class_names:                 # Temporal dict with class name and class number
    # print("    "+e)
    weights_temp[e] = count
    count = count + 1

images = data_dir.glob('**/*.jpg')
totalImages = len(list(images))
weights = dict()                        #Create dictionary with 
initial_count = 0
for specificClass in data_dir.iterdir():          #Loop for each folder
  head, tail = os.path.split(specificClass)
  initial_count = 0
  for elementFile in specificClass.iterdir():
      if elementFile.is_file():
          initial_count += 1
  weights[weights_temp[tail]]= float(totalImages/(num_classes  * initial_count))
  DataSetPrintInfo[tail] = {"NumClass ":weights_temp[tail],"NumElems": initial_count, "Weight": float(totalImages/(num_classes  * initial_count))}
print("Weights per class: "+str(DataSetPrintInfo))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Add more images due a Unbalanced input data
# Doc ----> https://www.tensorflow.org/guide/keras/preprocessing_layers
data_augmentation = keras.Sequential(
  [
    # tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    # tf.keras.layers.RandomRotation(0.4),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.5)
  ]
)
model = Sequential([
  tf.keras.Input(shape=(img_height, img_width, 3)),
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(num_classes)
])
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
  train_ds,
  validation_data = val_ds,
  epochs = initial_epochs, batch_size = batch_size_param,
  class_weight = weights
)

model.save(ReportName + '.h5')
print("Modelo guardado!")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(initial_epochs)

end = time.time()
total_time = end - start
print("Total time train : "+ str(total_time))

loss0, accuracy0, *is_anything_else_being_returned = model.evaluate(valImages,vallabels)
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