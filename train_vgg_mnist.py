'''
This script can be executed to pretrain VGG16 (unpruned) on MNIST for 200 epochs
'''

import time
START_SCRIPT = time.time()

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from model_classes import Unpruned_vgg16
from utils import load_MNIST, eval_model, train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

#########################################################

print("Loading MNIST")
    
MNIST_train_loader, MNIST_test_loader = load_MNIST()

#########################################################

print("Defining VGG-16 (unpruned)")
        
vgg16 = Unpruned_vgg16().to(device)

#########################################################

print("Training VGG-16 (unpruned) on MNIST")
 
epochs = 200
print("epochs used:", epochs)
log_name = "vgg_MNIST_unpruned_Adam" # log name for tensorboard 
MNIST_trained_vgg16, train_loss, train_acc, test_loss, test_acc = train(vgg16, MNIST_train_loader, MNIST_test_loader, epochs, device, SGD=False, log_name=log_name, return_metrics=True)

#########################################################

print("Saving trained parameters")

PATH ="VGG16_MNIST_Adam"
torch.save(MNIST_trained_vgg16.state_dict(), PATH)

#########################################################

END_SCRIPT = time.time()
print(f"Script running time : {END_SCRIPT - START_SCRIPT} seconds")