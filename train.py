# Imports here
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

train_dir = '/data/flowers/train'
valid_dir = '/data/flowers/valid'
test_dir = '/data/flowers/test'

#TODO: ??transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
# TODO: ??dir????datasets,????????transforms
train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir,transform=test_transforms)
test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)
# TODO: ??datasets??dataloaders
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=30)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=20)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = models.densenet121(pretrained=True)
model.cuda()

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('drop1',nn.Dropout(0.5)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(500, 128)),
                          ('drop2', nn.Dropout(0.5)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(128,102)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    

model.classifier = classifier
model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

def train(model, loader):
    model.train()
    train_loss = 0
    times = 0
    for ii,(inputs,labels) in enumerate(loader):
        inputs,labels = Variable(inputs),Variable(labels)
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        times += 1
        train_loss += loss.item()
        if times % 5 == 0:
            avg_loss = float(train_loss / 5)
            print(f"loss is: {avg_loss}")
            train_loss = 0
            
def test(model, loader):
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for ii,(inputs,labels) in enumerate(loader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model.forward(inputs)
            total += labels.size(0)
            _,predicts = torch.max(outputs.data,1)
            acc += ((predicts == labels).sum().item())
        avg_acc = float(acc/total)*100
        print(f"acc is: {avg_acc}")
        
epoch = 10
for _ in range(1, epoch):
    train(model, train_loader)
    test(model, valid_loader)

test(model, test_loader)

checkpoint = {'input_size':1024,'output_size':102,
               'hidden_layers':[500,128],
             'state_dict': model.state_dict(),
             'class_to_idx':train_datasets.class_to_idx}
torch.save(checkpoint, 'checkpoint.pth')

