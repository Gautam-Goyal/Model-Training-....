# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from workspace_utils import active_session
import numpy as np
import numpy

from math import ceil

import json 
from PIL import Image
def load_checkpoint(path='CHECKPOINT.pth',lr=0.001):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
       
    learningr=lr
    checkpoint = torch.load(path, map_location=map_location)
    model = models.vgg16(pretrained=True)
    model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer=optim.Adam(model.classifier.parameters(),lr=learningr)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    for param in model.parameters():
        param.requires_grad = False
       
    return model

def process_image(image):
    pic = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    pic2 = img_transform(pic)

    #np_image = np.array(pic2)

    return pic2

def predict(image_path, model, top_k=5,power='gpu'):
    if torch.cuda.is_available() and power=='gpu':
        model.to('cuda:0')
        
      # No need for GPU on this part (just causes problems)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
    model.to("cpu")
    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

def print_probability(probs, flowers):
    """
    Converts two lists into a dictionary to print on screen
    """
    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))
