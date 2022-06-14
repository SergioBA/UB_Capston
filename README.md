# UB_Capston
*Capston project repository for Data Science and Machine Learning Postgraduate Course.*

The goal of the project was to analyze Kvasir-Capsule dataset and test different DL models and technics (i.e. Data augmentation) to compare their classification rates.

UB_Capston project relies on Kvasir-Capsule dataset, which url can be accessed via: https://osf.io/gr7bn. Also, dataset can be downloaded via: https://osf.io/dv2ag/.

## Repository Structure
This repository has the following structure:
 * 📄 **[data](/data/)**: contains the dataset used. There you can find:
    * [OriginalDataset](/data/OriginalDataset/): dataset as provided from Kvasir-Capsule project (336 x 336)
    * dataSetOnly11Classes: original dataset but removing "Ampulla of vater", "Blood - hematin" and "Polyp" due to not enough samples to find patterns
    * dataSetPlits_0_1: original dataset splitted into two block for train & validation
 * 🤔 **info**: contains reference information i.e. Kvasir-Capsule code, its thesis or DL links
 * 📖 **docs**: contains UB_Capstone project documentation i.e. Analysis, methodology or models used
 * 🛠 **scripts**: folder to store all different tests developed during the project
    * experiments: contains the executed scripts
	* results: contains the scripts results

## Team members
 - Javier Sánchez Molino
 - Sergio Bravo Allué
 - Marc Bernabé Espinosa
 - Josep Fontana Castillo

## Prerequisites
Scripts were executed under Linux machine with Python 3 installed. The following packages have to be present:
| Package | Version |
| ----- |  ----- |
| tensorboard | 2.8.0 |
|  tensorboard-data-server | 0.6.1 |
|  tensorboard-plugin-wit | 1.8.1 |
|  tensorflow | 2.8.0 |
|  tensorflow-addons | 0.17.0 |
|  tensorflow-hub | 0.12.0 |
|  tensorflow-io-gcs-filesystem | 0.25.0 |
|  scikit-learn | 1.1.1 |
|  pandas | 1.4.1 |
|  keras | 2.8.0 |
|  Keras-Preprocessing | 1.1.2 |
|  numpy | 1.22.2 |
|  matplotlib | 3.5.2 |
|  matplotlib-inline | 0.1.3 |
|  opencv-python | 4.5.5.64 |
|  opencv-python-headless | 4.5.5.64 |

*Although not necesary, GPU is recommended to executed provided scripts. For that, please find the instructions to install cuda https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html*

Cuda version used:
 * cuda-11-7_11.7.0-1_amd64.deb

Cuda libraries used:
 * cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz

## Execution details
Execute the following steps:
 1. 💻 Go to <scripts/experiments>
 2. 🔥 Execute <python3 "model".py> where model can be:
    * Desnet_Splits01_NoAug_Weights.py
    * EfficenceB1_Splits01_NoAug_Weights.py
    * EfficenceB1_Splits01_NoAug_Weights_BigBatchSize.py
    * EfficenceB1_Splits01_NoAug_Weights_BigBatchSizex2.py
    * Resnet_Random30Per_NoAug_Weights.py
    * Resnet_Splits01_Aug_NoWeights.py
    * Resnet_Splits01_NoAug_NoWeights.py
    * Resnet_Splits01_NoAug_Weights.py
    * Resnet_Splits01_Aug_Weights.py       (Script with more comments)
