import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time

from helper_functions import train_model

use_gpu = torch.cuda.is_available()

# create transfomation
#  - scale the images
#  - convet the images from PIL to tensors
#  - normalize  
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CIFAR10(root='data', download=True, transform=transform)
testset = CIFAR10(root='data', train= False, transform=transform)

train_images = dataset.train_data
train_labels = dataset.train_labels

test_images = testset.test_data
test_labels = testset.test_labels

dataset_sizes = {
    'train': len(dataset.train_data),
    'val': len(testset.test_data)
}

train_dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True)
dataloders = {'train': train_dataloader, 'val': test_dataloader}

# creating the model
model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10) 

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)
model = train_model(model_ft, dataloders, dataset_sizes,criterion, optimizer_ft,num_epochs=5, use_gpu = use_gpu)

optimizer_ft = optim.Adam(model.fc.parameters(), lr=0.0001)
model = train_model(model, dataloders, dataset_sizes, criterion, optimizer_ft,num_epochs=10, use_gpu = use_gpu)

optimizer_ft = optim.Adam(model.fc.parameters(), lr=0.00005)
model = train_model(model, dataloders, dataset_sizes, criterion, optimizer_ft,num_epochs=10, use_gpu = use_gpu)



