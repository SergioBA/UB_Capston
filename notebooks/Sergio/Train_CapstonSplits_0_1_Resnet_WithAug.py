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
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


#ReportName
ReportName = "Resnet_WithAug_dataSetSplits_0_1"
ReportFile = open(ReportName+'.txt','w+')     # Create a report file
ReportFile.seek(0)
data_dir = Path(os.getcwd() + "/../../data/dataSetPlits_0_1/split_0/")
data_dir_validation = Path(os.getcwd() + "/../../data/dataSetPlits_0_1/split_1/")


#  Dataset Configuration
batch_size_param = 100
img_height = 100            #Max resolution 300     
img_width = 100
initial_epochs = 150
# test_percent = 0.2
# initial_seed=33

# OnlyCPU = True
# if OnlyCPU:
#   os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  # validation_split=test_percent,
  color_mode = "rgb",
  # subset="training",
  # seed=initial_seed,
  image_size=(img_height, img_width),
  interpolation = "bilinear",
  follow_links = False,
  crop_to_aspect_ratio = False,
  batch_size=batch_size_param)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_validation,
  shuffle=True,
  # validation_split=test_percent,
  color_mode = "rgb",
  # subset="validation",
  # seed=initial_seed,
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

ReportFile.write("List of class:\n")
class_names = train_ds.class_names
num_classes = len(class_names)        #Number of classes
weights_temp = dict()                 # To load className and ClassNumber
DataSetPrintInfo = dict()
count = 0
for e in class_names:                 # Temporal dict with class name and class number
    ReportFile.write("    "+str(count)+" : "+e+"\n")
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
ReportFile.write("Weights per class: "+str(DataSetPrintInfo)+"\n")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(10000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Add more images due a Unbalanced input data
# Doc ----> https://www.tensorflow.org/guide/keras/preprocessing_layers
data_augmentation = keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.6),
    # tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomContrast(0.5)
  ]
)
base_model = tf.keras.applications.resnet_v2.ResNet152V2(                     # Alternative -> resnet_v2.ResNet152V2  /  resnet50.ResNet50
                                          include_top=False,
                                          weights='imagenet',
                                          input_tensor=None,
                                          input_shape=None,
                                          pooling=None,
                                          classes=1000,
                                          classifier_activation='softmax')
base_model.trainable = False 
model = tf.keras.Sequential([
        tf.keras.Input(shape=(img_height, img_width, 3)),
        data_augmentation,                                                                        # Data augmentation
        tf.keras.layers.Rescaling(1./127.5, offset=-1),                                           # Data rescalation
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes)])

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
# print("Modelo guardado!")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(initial_epochs)

end = time.time()
total_time = end - start
ReportFile.write("Total time train : "+ str(total_time)+"\n")

loss0, accuracy0, *is_anything_else_being_returned = model.evaluate(valImages,vallabels)
ReportFile.write("Initial loss: {:.2f}\n".format(loss0))
ReportFile.write("Initial accuracy: {:.2f}\n".format(accuracy0))




# Evaluate the model

def test_model(y_true, y_predicted):
    ReportFile.write("Accuracy = {:.3f}\n".format(mtc.accuracy_score(y_true, y_predicted)))
    ReportFile.write("Accuracy Balanced = {:.3f}\n".format(mtc.balanced_accuracy_score(y_true, y_predicted)))
    
    ReportFile.write("Precision micro = {:.3f}\n".format(mtc.precision_score(y_true,y_predicted, average="micro")))
    ReportFile.write("Precision macro = {:.3f}\n".format(mtc.precision_score(y_true,y_predicted, average="macro")))
    ReportFile.write("Precision weighted = {:.3f}\n".format(mtc.precision_score(y_true,y_predicted, average="weighted")))
    
    ReportFile.write("Recall micro = {:.3f}\n".format(mtc.recall_score(y_true, y_predicted, average="micro")))
    ReportFile.write("Recall macro = {:.3f}\n".format(mtc.recall_score(y_true, y_predicted, average="macro")))
    ReportFile.write("Recall weighted = {:.3f}\n".format(mtc.recall_score(y_true, y_predicted, average="weighted")))

    ReportFile.write("F1 micro = {:.3f}\n".format(mtc.f1_score(y_true, y_predicted, average="micro")))
    ReportFile.write("F1 macro = {:.3f}\n".format(mtc.f1_score(y_true, y_predicted, average="macro")))
    ReportFile.write("F1 weighted = {:.3f}\n".format(mtc.f1_score(y_true, y_predicted, average="weighted")))

    ReportFile.write("MCC = {:.3f}\n".format(mtc.matthews_corrcoef(y_true, y_predicted)))
    ReportFile.write("Kappa = {:.3f}\n".format(mtc.cohen_kappa_score(y_true, y_predicted)))


# Last number from the models
predValImages = model.predict(valImages)
predictionOfValImages = []
for value in predValImages:
  predictionOfValImages.append(np.argmax(value))
test_model(vallabels, predictionOfValImages)
ReportFile.write("Classes not found" + str(set(vallabels) - set(predictionOfValImages))+"\n")
report = classification_report(vallabels, predictionOfValImages)
ReportFile.write(report+"\n")
ReportFile.truncate()
ReportFile.close() 

# Store images ina file
plt.figure(figsize=(8, 8))
plt.subplot(1, 3, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 3, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(ReportName + '_Train_Val.png')

# confusion matrix plot
plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
cm = confusion_matrix(vallabels, predictionOfValImages, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= list(range(num_classes)))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig(ReportName + '_ConfMatrix.png')

