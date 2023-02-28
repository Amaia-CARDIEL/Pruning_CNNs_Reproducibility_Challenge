'''
This script can be executed to pretrain VGG16 (unpruned) on CIFAR10 for 200 epochs
'''

import time
START_SCRIPT = time.time()

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from model_classes import Unpruned_vgg16
from utils import load_CIFAR, eval_model, train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

#########################################################

print("Loading CIFAR")
    
CIFAR_train_loader, CIFAR_test_loader = load_CIFAR()

#########################################################

print("Defining VGG-16 (unpruned)")
        
vgg16 = Unpruned_vgg16().to(device)

#########################################################

print("Training VGG-16 on CIFAR")

epochs = 200 
print("epochs used:", epochs)
CIFAR_trained_vgg16 = train(vgg16, CIFAR_train_loader, CIFAR_test_loader, epochs, device, SGD=True, log_name='vgg_CIFAR_unpruned', return_metrics=False)

#########################################################

print("Saving trained parameters")

PATH ="VGG16_CIFAR"
torch.save(CIFAR_trained_vgg16.state_dict(), PATH)

#########################################################

END_SCRIPT = time.time()
print(f"Script running time : {END_SCRIPT - START_SCRIPT} seconds")