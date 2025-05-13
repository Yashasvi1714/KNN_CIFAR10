# k-Nearest Neighbors (kNN) Classifier on CIFAR-10 Dataset

This repository contains the implementation of a k-Nearest Neighbors (kNN) classifier applied to the CIFAR-10 dataset. The purpose of the project is to evaluate the performance of the kNN classifier with different values of `k` and to demonstrate how hyperparameter tuning can be performed using a validation set.

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. The model's goal is to classify these images into the appropriate classes. This project involves:

- Preprocessing the dataset
- Hyperparameter tuning (`k` value for kNN)
- Training and evaluating the kNN model
- Visualizing the results

## Requirements

- Python 3.x
- TensorFlow (for CIFAR-10 dataset)
- scikit-learn (for kNN and model evaluation)
- NumPy (for numerical operations)
- Matplotlib (for plotting)

You can install the necessary Python packages using:

```bash
pip install tensorflow scikit-learn numpy matplotlib
