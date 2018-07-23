# Udacity Image Classifier Project: Part 2
# Building a command line applications: train.py
################################
# Created by Vivekanand Sharma #
# Dated July 21, 2018          #
################################

# Requirements:
# The first file, train.py, will train a new network on a dataset and save the model as a checkpoint
# Train a new network on a data set with train.py
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu


import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import argparse

# CALL python train.py /home/workspace/aipnd-project/flowers/

train_parser = argparse.ArgumentParser()

 # User should be able to type python train.py data_directory
    # Non-optional argument - must be input (not as -- just the direct name i.e. python train.py flowers)
train_parser.add_argument('data_dir', action="store", nargs='*', default="/home/workspace/aipnd-project/flowers/")
    # Choose where to save the checkpoint
train_parser.add_argument('--save_dir', action="store", dest="save_dir", default="/home/workspace/aipnd-project/checkpoint.pth")
    # Choose model architecture
train_parser.add_argument('--arch', action="store", dest="arch", default="vgg16")
    # Choose learning rate
train_parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001)
    # Choose number of epochs
train_parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=1)
    # Choose number of hidden units
train_parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=4096)
    # Choose processor
train_parser.add_argument('--processor', action="store", dest="processor", default="GPU")

train_args = train_parser.parse_args()

print("Image Directory: ", train_args.data_dir)
print("Save Directory : ", train_args.save_dir)
print("Model          : ", train_args.arch)
print("Learning Rate  : ", train_args.learning_rate)
print("Epochs         : ", train_args.epochs)
print("Hidden units   : ", train_args.hidden_units)
print("Processor      : ", train_args.processor)


from reqd_funcs import trainer, validation

def main():
    print(train_args.arch)
    trainer(train_args.arch, train_args.data_dir, train_args.save_dir, train_args.learning_rate,
            train_args.epochs, train_args.hidden_units, train_args.processor)

# Call to get_input_args function to run the program
if __name__ == "__main__":
    main()
