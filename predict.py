import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import pandas as pd
import seaborn as sb
from PIL import Image
from collections import OrderedDict

def load_checkpoint(filepath):
    model = models.densenet121(pretrained=True)
    checkpoint = torch.load(filepath)
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1])),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(checkpoint['hidden_layers'][1],checkpoint['output_size'])), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model,checkpoint['class_to_idx']
    

model, class_to_idx = load_checkpoint('checkpoint.pth')
class_to_idx = dict(zip(class_to_idx.values(), class_to_idx.keys()))
print(class_to_idx)
model.cuda()
model.eval()
model
test(model, test_loader)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im.thumbnail((im.size[0]/im.size[1]*256,256))
    half_the_width = im.size[0] / 2
    half_the_height = im.size[1] / 2
    img = im.crop(
        (
            half_the_width - 112,
            half_the_height - 112,
            half_the_width + 112,
            half_the_height + 112
        )
    )
    
    np_image = np.array(img)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm = (np_image - mean) / std
    np_image = norm.transpose((2, 0, 1))
    new_img = torch.from_numpy(np_image)
    return new_img

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    img_tensor = process_image(image_path)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.type(torch.cuda.FloatTensor)
    output = model(Variable(img_tensor.cuda(), volatile=True))
    ps = torch.exp(output)
    probs, index = ps.topk(topk)
    print(probs)
    print(index)
    probs = probs.cpu().detach().numpy().tolist()[0]
    index = index.cpu().detach().numpy().tolist()[0]
    index = [class_to_idx[i] for i in index]
    return probs, index

model.cuda()
probs, classes = predict(train_dir+"/1/image_06735.jpg", model)
print(probs)
print(classes)