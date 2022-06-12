# UB_Capston
Capston project repository for Data Science and Machine Learning Postgraduate Course. *Question to reject* The goal of the project was to analyze Kvasir-Capsule dataset and test different DL models and technics (i.e. Data augmentation) to compare their classification rates.

UB_Capston project relies on Kvasir-Capsule dataset, which url can be accessed via: https://osf.io/gr7bn. Also, dataset can be downloaded via: https://osf.io/dv2ag/.

## Repository Structure
This repository has the following structure:
 - data: contains the dataset used. There you can find:
    - OriginalDataset: dataset as provided from Kvasir-Capsule project (336 x 336)
    - dataSetOnly11Classes: original dataset but removing "Ampulla of vater", "Blood - hematin" and "Polyp" due to not enough samples to find patterns
    - dataSetPlits_0_1: original dataset splitted into two block for train & validation
 - info: contains reference information i.e. Kvasir-Capsule code, its thesis or DL links
 - doc: contains UB_Capstone project documentation i.e. Analysis, methodology or models used
 - notebooks: folder to store all different scripts/tests developed during the project

## Team members
 - Javier Sánchez Molino
 - Sergio Bravo Allué
 - Marc Bernabé Espinosa
 - Josep Fontana Castillo

## Prerequisites
Scripts were executed under Linux machine with Python 3 installed. The following packages have to be present:
 - tensorboard                  2.8.0
 - tensorboard-data-server      0.6.1
 - tensorboard-plugin-wit       1.8.1
 - tensorflow                   2.8.0
 - tensorflow-addons            0.17.0
 - tensorflow-hub               0.12.0
 - tensorflow-io-gcs-filesystem 0.25.0
 - scikit-learn                 1.1.1
 - pandas                       1.4.1
 - keras                        2.8.0
 - Keras-Preprocessing          1.1.2
 - numpy                        1.22.2
 - matplotlib                   3.5.2
 - matplotlib-inline            0.1.3
 - opencv-python                4.5.5.64
 - opencv-python-headless       4.5.5.64

*Although not necesary, GPU is recommended to executed provided scripts. For that, please find the instructions to install cuda https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

Cuda version used:
 - cuda-11-7_11.7.0-1_amd64.deb

Cuda libraries used:
 - cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz

## Execution details
Execute the following steps:
 - Go to <UB_Capston/notebooks/experiments>
 - Execute <python3 "model".py> where model can be:
    - Train_CapstonRandomSplits_Desnet_WithAug_30Percent.py
    - Train_CapstonRandomSplits_Desnet_WithAug_50Percent.py
    - Train_CapstonRandomSplits_EfficientNetB7_WithAug_30Percent.py
    - Train_CapstonRandomSplits_EfficientNetB7_WithAug_50Percent.py
    - Train_CapstonRandomSplits_Resnet_WithAug_30Percent.py
    - Train_CapstonRandomSplits_Resnet_WithAug_50Percent.py
    - Train_CapstonSplits_0_1_Basic_WithAug.py
    - Train_CapstonSplits_0_1_Desnet_WithAug.py
    - Train_CapstonSplits_0_1_Desnet_WithAug_NOWeights.py
    - Train_CapstonSplits_0_1_EfficientNetB7_WithAug.py
    - Train_CapstonSplits_0_1_Resnet_WithAug.py
    - Train_CapstonSplits_0_1_Resnet_WithAug_BigBatch.py
    - Train_CapstonSplits_0_1_Resnet_WithAug_NOWeights.py