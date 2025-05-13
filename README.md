# KNN on CIFAR-10

This repository demonstrates the application of the k-Nearest Neighbors (kNN) classifier on the CIFAR-10 dataset. The project includes multiple assignments and a final implementation that explores various aspects of the kNN algorithm, including hyperparameter tuning, model evaluation, and result visualization.

---

## Table of Contents

1. [Overview](#overview)
2. [Assignments](#assignments)
    1. [Assignment 1](#assignment-1)
    2. [Assignment 2](#assignment-2)
    3. [Assignment 3](#assignment-3)
3. [Final Project](#final-project)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Evaluation and Results](#evaluation-and-results)
8. [Conclusion](#conclusion)
9. [License](#license)

---

## Overview

The k-Nearest Neighbors (kNN) classifier is a simple yet effective machine learning algorithm used for classification tasks. In this project, the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes, is used. The objective is to explore the impact of different values of `k` on the performance of the classifier, perform hyperparameter tuning, and evaluate the classifier’s performance.

---

## Assignments

### Assignment 1: kNN Classifier on CIFAR-10

**Objective:**  
Implement the kNN classifier on the CIFAR-10 dataset. The first assignment focuses on loading the CIFAR-10 dataset, applying the kNN algorithm, and evaluating the initial performance of the classifier.

**Steps:**
- Load and preprocess the CIFAR-10 dataset.
- Implement the kNN algorithm using scikit-learn.
- Evaluate the model with different values of `k` (e.g., 1, 3, 5, etc.).
- Visualize the classification results and compute accuracy.

### Assignment 2: Hyperparameter Tuning

**Objective:**  
Optimize the kNN model by tuning the hyperparameter `k`. This assignment involves using a validation set to find the optimal value of `k` and improving the model's performance.

**Steps:**
- Split the CIFAR-10 dataset into training, validation, and test sets.
- Implement a loop to test different values of `k` (e.g., 1, 3, 5, 7) and evaluate performance.
- Select the value of `k` that gives the highest validation accuracy.
- Retrain the model on the entire training set with the selected `k` and evaluate on the test set.

### Assignment 3: Model Evaluation and Visualization

**Objective:**  
Evaluate the performance of the kNN classifier using various metrics and visualize the results. This assignment explores confusion matrices, classification reports, and visual comparisons of predicted vs. actual labels.

**Steps:**
- Compute performance metrics such as accuracy, precision, recall, and F1-score.
- Generate a confusion matrix to understand misclassifications.
- Visualize a few sample images with their predicted and actual labels.

---

## Final Project

**Objective:**  
In the final project, the learnings from the previous assignments are consolidated into a comprehensive kNN implementation on the CIFAR-10 dataset. The final project involves:
- Applying the kNN classifier to the entire CIFAR-10 dataset.
- Performing hyperparameter tuning to select the best value of `k`.
- Evaluating and visualizing the results.
- Discussing the challenges and limitations of the kNN algorithm on the CIFAR-10 dataset.

**Steps:**
- Implement the kNN classifier.
- Tune the hyperparameter `k` using a validation set.
- Evaluate the final model using the test set.
- Visualize results with confusion matrices and other performance metrics.

---

## Requirements

- Python 3.x
- TensorFlow (for loading CIFAR-10 data)
- Scikit-learn (for kNN and model evaluation)
- Numpy (for numerical operations)
- Matplotlib (for plotting)

---

## Installation

To set up the environment and install the required dependencies, run:

```bash
pip install tensorflow scikit-learn numpy matplotlib

