# Softmax Regression on MNIST

This repository contains an implementation of Softmax Regression on the MNIST dataset using PyTorch. The MNIST dataset consists of handwritten digits and is commonly used for training various image processing systems.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction

Softmax Regression is a generalization of logistic regression that is used for multi-class classification problems. In this project, we use Softmax Regression to classify handwritten digits from the MNIST dataset.

## Dataset

The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. Each image is 28x28 pixels.

## Model Architecture

The Softmax Regression model consists of a single linear layer that maps the input image (flattened into a vector) to the output classes (digits 0-9).

```
class SoftmaxRegressionModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the images
        return self.linear(x)
```

## Training

The training process involves the following steps:
1. **Initialize the model, loss function, and optimizer**:
    - Model: Softmax Regression
    - Loss Function: Cross-Entropy Loss
    - Optimizer: Stochastic Gradient Descent (SGD)

2. **Training Loop**:
    - For each epoch, iterate over the training dataset in batches.
    - Perform the forward pass to compute the model's predictions.
    - Compute the loss between the predictions and the ground truth labels.
    - Perform the backward pass to compute the gradients.
    - Update the model's parameters using the optimizer.

## Evaluation

The evaluation process involves:
1. **Validation**:
    - Evaluate the model on the validation dataset after each epoch.
    - Compute the validation loss and accuracy to monitor the model's performance.

2. **Testing**:
    - After training, evaluate the model on the test dataset.
    - Compute the test accuracy to assess the model's generalization performance.

## Results

The results of the training and evaluation process include:
- Training and validation loss over epochs.
- Training and validation accuracy over epochs.
- Test accuracy after training.
- Confusion matrix for the test dataset predictions.
- Plots for visualization.
- With and without L2 Regularization.

## Usage

To use this implementation, follow these steps:
1. Clone the repository:
    ```
    git clone https://github.com/ranimeshehata/Softmax-Regression-on-MNIST.git
    cd Softmax-Regression-on-MNIST
    ```

2. Install the required dependencies:
    ```
    pip install torch torchvision
    ```

3. Run the model

4. Observe the outputs and you can change batch size, learning rate or number of epochs for model tuning.

## Dependencies

- Python 3.9
- PyTorch
- torchvision
- scikit-learn
- matplotlib