# resnet50-garbage-classification
A garbage image classification project using ResNet50 transfer learning with TensorFlow.

本專題為深度學習課程期末作品，使用 ResNet50 與遷移學習建立垃圾影像分類模型，將垃圾影像分類為 cardboard、glass、metal、paper、plastic 與 trash 六種類別。

Project Overview

垃圾分類在實際情境中可能因外觀或材質相似而產生判斷差異，因此本專題嘗試透過深度學習影像分類模型，建立較一致的垃圾分類方式。

本研究以 ResNet50 作為模型骨幹，使用 ImageNet 預訓練權重，搭配資料增強、Class Weight、Fine-tuning、Early Stopping 與 ReduceLROnPlateau 進行模型訓練。

Classes

The model classifies images into six categories:

cardboard

glass

metal

paper

plastic

trash

Dataset

The project uses garbage images from the TrashNet dataset.

本研究選擇六種類別進行分類實驗：

cardboard: 891 images

paper: 1050 images

glass: 607 images

plastic: 865 images

metal: 769 images

trash: 697 images

The dataset is divided into training and validation sets.

All images are resized to:

224 x 224 RGB

Pixel values are normalized to the range 0 to 1.

Data Augmentation

Training images are augmented using:

Rotation

Width shift

Height shift

Brightness adjustment

Zoom

Shear

Horizontal flip

The validation set is only normalized without augmentation.

Model Architecture

The model is based on ResNet50 with ImageNet pretrained weights.

Input Image
    |
    v
ResNet50 Backbone
    |
    v
Global Average Pooling
    |
    v
Dense (512, ReLU)
    |
    v
Dropout (0.5)
    |
    v
Softmax
    |
    v
6 Classes

The ResNet50 backbone is fine-tuned by freezing the earlier layers while keeping the final 50 layers trainable.

Training Configuration

Image Size: 224 x 224

Batch Size: 16

Maximum Epochs: 20

Optimizer: Adam

Learning Rate: 1e-4

Loss Function: Categorical Crossentropy

Class Weight for class imbalance

ReduceLROnPlateau

Early Stopping

Training Results

During training, the training accuracy gradually increased and the validation accuracy also improved.

The training and validation loss both decreased during the training process, and the validation loss remained relatively stable.

The project successfully completed the basic six-class garbage image classification task.

Prediction Demo

The demo.py program loads the trained model and predicts a single input image.

The system outputs:

Predicted class

Confidence score

Input image with prediction result

Example:

Prediction:
Class: glass
Confidence: 0.93

Project Files

train.py - Model training, fine-tuning, data augmentation, class weighting, and model saving

demo.py - Single image prediction

report.pdf - Course project report

README.md - Project documentation

Technologies

Python

TensorFlow

Keras

ResNet50

NumPy

Matplotlib

scikit-learn

Transfer Learning

Image Classification

Limitations

The project showed lower prediction confidence for some categories, especially metal, plastic, and trash.

Possible reasons include:

Limited training samples

Similar appearance or material between different garbage categories

Class imbalance

Differences between training images and real-world test images

Future Improvements

Future improvements could include:

Increasing the dataset size

Expanding data augmentation strategies

Testing different model architectures

Improving classification performance for visually similar categories

Team Project

Deep Learning Course Final Project

Developed by YuHsuan Chen and team.
