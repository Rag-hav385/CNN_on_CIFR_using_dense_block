# Convolutional Neural Network (CNN) on CIFAR-10 Dataset

## Introduction

This code demonstrates the implementation of a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using TensorFlow and Keras. The goal is to train a model for image classification.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- TensorFlow
- Keras

You can install them using pip:

pip install tensorflow


## Dataset

The CIFAR-10 dataset is used for training and testing. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.

## Model Architecture

The CNN model is designed with the following architecture:

- Initial Convolution Layer
- Dense Blocks
- Transition Blocks
- Output Layer

The number of filters, dropout rates, and other hyperparameters are customizable.

## Training

The model is trained with data augmentation using the ImageDataGenerator. Training stops automatically when the validation accuracy reaches 90%, thanks to the custom callback `stop_at_acc`. Additional callbacks include learning rate reduction and model checkpointing.

## Running the Code

1. Make sure you have installed the required dependencies.

2. Download the CIFAR-10 dataset using TensorFlow's built-in function.

3. Run the code to train the CNN model.

4. Monitor training progress using TensorBoard by running the command: `%load_ext tensorboard` and `%tensorboard --logdir .`

## Author

- Raghav Agarwal

Feel free to modify and experiment with the code to achieve better results or adapt it for your specific image classification tasks.
