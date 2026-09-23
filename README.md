# resnet50-garbage-classification
A garbage image classification project using ResNet50 transfer learning with TensorFlow.
# ResNet50 Garbage Classification

A garbage image classification project using ResNet50 transfer learning with TensorFlow.

本專題為深度學習課程期末作品，
使用 ResNet50 與遷移學習建立垃圾影像分類模型，
將垃圾影像分類為 cardboard、glass、metal、paper、plastic 與 trash 六種類別。

## Project Overview

垃圾分類在實際情境中可能因外觀、材質相似而產生判斷差異，
因此本專題嘗試透過深度學習影像分類模型，
建立較一致的垃圾分類方式。

本研究使用 ResNet50 作為影像特徵擷取骨幹，
搭配資料增強、Class Weight 與 Fine-tuning 進行模型訓練。

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

Images are divided into training and validation sets.

All images are resized to:

```text
224 × 224 RGB
