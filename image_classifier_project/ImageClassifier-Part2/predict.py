# Udacity Image Classifier Project: Part 2
# Building a command line applications: predict.py
################################
# Created by Vivekanand Sharma #
# Dated July 21, 2018          #
################################

# Requirements:
# Basic usage: python predict.py /path/to/image checkpoint
# Return top K most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu


import matplotlib.pyplot as plt
import time
import json
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import argparse

predict_parser = argparse.ArgumentParser()

 # User should be able to type python predict.py input checkpoint
    # Non-optional image file input
predict_parser.add_argument('input', action="store", nargs='*', default='/home/workspace/aipnd-project/flowers/test/28/image_05214.jpg')
#predict_parser.add_argument('input', action="store", nargs='*', default='image_05214.jpg')
    # Non-optional checkpoint 
predict_parser.add_argument('checkpoint', action="store", nargs='*', default='/home/workspace/aipnd-project/checkpoint.pth')
#predict_parser.add_argument('checkpoint', action="store", nargs='*', default='checkpoint.pth')
    # Choose top K 
predict_parser.add_argument('--top_k', action="store", dest="top_k", type=int, default=3)
    # Choose category list
predict_parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')
  # Choose processor
predict_parser.add_argument('--processor', action="store", dest="processor", default="GPU")

predict_args = predict_parser.parse_args()

with open(predict_args.category_names, 'r') as f:
	cat_to_name = json.load(f)

from reqd_funcs import predict, load_checkpoint, process_image

def main():
	probs, classes, y_labels = predict(predict_args.input, predict_args.checkpoint, predict_args.top_k, cat_to_name, predict_args.processor)
	print("Probs:",probs)
	print("Class:",classes)
	print("Class Names:",y_labels)


if __name__ == "__main__":
    main()