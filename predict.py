import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import predict_load_functions
import train_save_functions

#Command Line Arguments

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/ImageClassifier/trained_modelOO7.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint


print('Loading Data........')
training_loader, testing_loader, validation_loader,train_datasets = train_save_functions.Load_Data()

print('Loading Checkpoint....')
model=predict_load_functions.load_checkpoint(path)


with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

print('Predicting.......')
top_probs, top_labels, top_flowers = predict_load_functions.predict(path_image, model, number_of_outputs, power)

print(top_flowers)
print(top_probs)
predict_load_functions.print_probability(top_flowers, top_probs)

print("Here you are")