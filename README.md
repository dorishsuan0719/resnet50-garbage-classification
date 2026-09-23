# ResNet50 Garbage Classification

A garbage image classification project using ResNet50 transfer learning with TensorFlow.

本專題為深度學習課程期末作品，使用 ResNet50 與遷移學習建立垃圾影像分類模型，將垃圾影像分類為 cardboard、glass、metal、paper、plastic 與 trash 六種類別。

## Project Overview

垃圾分類在實際情境中可能因外觀或材質相似而產生判斷差異，因此本專題嘗試透過深度學習影像分類模型，建立較一致的垃圾分類方式。

本研究以 ResNet50 作為模型骨幹，使用 ImageNet 預訓練權重，搭配資料增強、Class Weight、Fine-tuning、Early Stopping 與 ReduceLROnPlateau 進行模型訓練。

## Classes

The model classifies images into six categories:

- cardboard
- glass
- metal
- paper
- plastic
- trash

## Dataset

The project uses garbage images from the TrashNet dataset.

本研究選擇六種類別進行分類實驗：

- cardboard: 891 images
- paper: 1050 images
- glass: 607 images
- plastic: 865 images
- metal: 769 images
- trash: 697 images

The dataset is divided into training and validation sets.

All images are resized to **224 x 224 RGB**.

Pixel values are normalized to the range **0 to 1**.

## Data Preprocessing

資料前處理主要包含：

- Resize images to 224 x 224
- Normalize pixel values to 0 to 1
- Separate training and validation datasets
- Load images using ImageDataGenerator

## Data Augmentation

Training images are augmented using:

- Rotation
- Width shift
- Height shift
- Brightness adjustment
- Zoom
- Shear
- Horizontal flip

The validation set is only normalized without augmentation.

## Model Architecture

The model is based on ResNet50 with ImageNet pretrained weights.

Model structure:

- Input Image: 224 x 224 RGB
- ResNet50 Backbone
- Global Average Pooling
- Dense Layer: 512 units with ReLU
- Dropout: 0.5
- Softmax Output Layer
- 6 Output Classes

The earlier ResNet50 layers are frozen while the final 50 layers remain trainable for fine-tuning.

## Training Configuration

- Image Size: 224 x 224
- Batch Size: 16
- Maximum Epochs: 20
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss Function: Categorical Crossentropy
- Class Weight for class imbalance
- ReduceLROnPlateau
- Early Stopping

## Class Imbalance Handling

Because the number of images differs between categories, Class Weight is used during training.

This helps reduce the effect of class imbalance and allows categories with fewer samples to contribute more fairly during model training.

## Training Results

During training, the training accuracy gradually increased and the validation accuracy also improved.

The training and validation loss both decreased during the training process, while the validation loss remained relatively stable.

The project successfully completed the basic six-class garbage image classification task.

## Prediction Demo

The `demo.py` program loads the trained model and predicts a single input image.

The system outputs:

- Predicted class
- Confidence score
- Input image with prediction result

Example result:

- Class: glass
- Confidence: 0.93

The prediction result is also displayed together with the input image using Matplotlib.

## Project Files

- `train.py` - Model training, data augmentation, fine-tuning, class weighting, and model saving
- `demo.py` - Single image prediction
- `report.pdf` - Course project report
- `README.md` - Project documentation

## Technologies

- Python
- TensorFlow
- Keras
- ResNet50
- NumPy
- Matplotlib
- scikit-learn
- Transfer Learning
- Image Classification

## Limitations

The project showed lower prediction confidence for some categories, especially metal, plastic, and trash.

Possible reasons include:

- Limited training samples
- Similar appearance between different garbage categories
- Similar materials between categories
- Class imbalance
- Differences between training images and real-world test images

## Future Improvements

Future improvements could include:

- Increasing the dataset size
- Expanding data augmentation strategies
- Collecting more real-world garbage images
- Testing different model architectures
- Improving classification performance for visually similar categories
- Comparing ResNet50 with other image classification models

## Team Project

Deep Learning Course Final Project

Developed by YuHsuan Chen and team.
