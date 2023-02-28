'''
This script can be executed to pretrain ResNet50 (unpruned) on CIFAR10 for 300 epochs
'''

import time
START_SCRIPT = time.time()

import json
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from model_classes import Unpruned_resnet, Identity_block, Convolutional_block
from utils import load_CIFAR, eval_model, train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

#########################################################

print("Loading CIFAR")
    
CIFAR_train_loader, CIFAR_test_loader = load_CIFAR()

#########################################################

print("Defining ResNet (unpruned)")
        
resnet = Unpruned_resnet().to(device)

#########################################################

print("Training ResNet on CIFAR")

epochs = 300 
print("epochs used:", epochs)
start = time.time()
trained_resnet, train_loss, train_acc, test_loss, test_acc = train(resnet, CIFAR_train_loader, CIFAR_test_loader, epochs, device, SGD=True, log_name='resnet_unpruned_cifar', return_metrics=True)
end = time.time()

#########################################################

print("Saving metrics")

unpruned_dict= {"train_loss":train_loss, "train_acc":train_acc, "test_loss":test_loss, "test_acc": test_acc, "runtime":end-start}
unpruned_dict_filepath = 'resnet_pretraining_metrics.json'
json.dump(unpruned_dict,open(unpruned_dict_filepath,"w"))

print("Saving trained parameters")

PATH ="RESNET_CIFAR"
torch.save(trained_resnet.state_dict(), PATH)

#########################################################

END_SCRIPT = time.time()
print(f"Script running time : {END_SCRIPT - START_SCRIPT} seconds")

