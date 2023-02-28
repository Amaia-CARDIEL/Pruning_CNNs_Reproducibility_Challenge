'''
This script can be executed to prune and finetune VGG16 on MNIST or CIFAR10.
The user can choose which filter scoring method to use to prune the model, 
between the article's method ("operator_norm") and 2 baselines methods ("l1_norm" and "GM_norm").
'''

import time
START_SCRIPT = time.time()

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import json

from model_classes import *
from utils import load_MNIST, load_CIFAR, eval_model, train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

#########################################################
#                 VALUES TO BE MODIFIED
#						BY USER
#########################################################

# Dataset to use 
dataset_name="CIFAR" # among: "MNIST" or "CIFAR"

# Pruning method to use 
scoring_name="operator_norm" # among: "operator_norm", "l1_norm" and "GM_norm"

# Do we save finetuned pruned models ?
SAVING = False # else: True

#########################################################

# Loading the dataset and defining path to pretrained model

if dataset_name=="MNIST":
	print("\nLoading MNIST")
	train_loader, test_loader = load_MNIST()
	LOAD_PATH = "VGG16_MNIST_Adam"

else:
	print("\nLoading CIFAR")
	train_loader, test_loader = load_CIFAR()
	LOAD_PATH = "VGG16_CIFAR"

#########################################################

# Dictionary containing all the ratios and layer combinations to prune

pruning_dict = {'v0': [0,0,0,0,0,0,0,0,0,0,0,0,0], # pruning no layer

				'v1_25': [0,0,0,0,0,0,0,0,0,0,0.25,0.25,0.25], # pruning C11_13
				'v1_50': [0,0,0,0,0,0,0,0,0,0,0.50,0.50,0.50],
				'v1_75': [0,0,0,0,0,0,0,0,0,0,0.75,0.75,0.75],
				'v1_90': [0,0,0,0,0,0,0,0,0,0,0.90,0.90,0.90],

				'v2_25': [0,0,0,0,0,0,0,0,0.25,0.25,0.25,0.25,0.25], # pruning C9_13
				'v2_50': [0,0,0,0,0,0,0,0,0.50,0.50,0.50,0.50,0.50],
				'v2_75': [0,0,0,0,0,0,0,0,0.75,0.75,0.75,0.75,0.75],
				'v2_90': [0,0,0,0,0,0,0,0,0.90,0.90,0.90,0.90,0.90],

				'v3_25': [0,0,0,0,0,0,0.25,0.25,0.25,0.25,0.25,0.25,0.25], # pruning C7_13
				'v3_50': [0,0,0,0,0,0,0.50,0.50,0.50,0.50,0.50,0.50,0.50],
				'v3_75': [0,0,0,0,0,0,0.75,0.75,0.75,0.75,0.75,0.75,0.75],
				'v3_90': [0,0,0,0,0,0,0.90,0.90,0.90,0.90,0.90,0.90,0.90],

				'v4_25': [0,0,0,0,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25], # pruning C5_13
				'v4_50': [0,0,0,0,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50],
				'v4_75': [0,0,0,0,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75],
				'v4_90': [0,0,0,0,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90],

				'v5_25': [0,0,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25], # pruning C3_13 
				'v5_50': [0,0,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50], # (only for MNIST)
				'v5_75': [0,0,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75],
				'v5_90': [0,0,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90],

				'v6_25': [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25], # pruning C1_13
				'v6_50': [0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50],
				'v6_75': [0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75],
				'v6_90': [0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90]

				}

#########################################################

# Loading pretrained model

print("Loading pretrained model (VGG-16 unpruned)")
pretrained_vgg16 = Unpruned_vgg16().to(device)
pretrained_vgg16.load_state_dict(torch.load(LOAD_PATH))
nb_params = sum(param.numel() for param in pretrained_vgg16.parameters())
print("The pretrained model has a total of", nb_params, "parameters")

# Iterating through the pruning ratios version

final_metrics_dict={} # to save final losses and accuracies
dict_filepath = 'vgg_' + scoring_name + '_' + dataset_name + '_metrics.json'

for version in pruning_dict.keys(): 

	# Defining the pruned model

	print("\nPruning VGG-16 with version", version, "and", scoring_name, "scoring")
	p_vec = pruning_dict[version]
	pruned_vgg = Pruned_vgg16(p_vec, pretrained_vgg16, dataset_name, scoring_name).to(device)
	nb_params = sum(param.numel() for param in pruned_vgg.parameters())
	print("The pruned model has a total of", nb_params, "parameters")

	# Training the pruned model 

	print("Finetuning pruned VGG-16")
	epochs = 100
	print("epochs used:", epochs)
	log_name = "vgg_" + scoring_name + '_' + dataset_name + "_pruned_" + version # log name for tensorboard 
	start=time.time()
	trained_pruned_vgg16, train_loss, train_acc, test_loss, test_acc = train(pruned_vgg, train_loader, test_loader, epochs, device, log_name=log_name, return_metrics=True)
	end=time.time()

	if SAVING==True:
		print("Saving finetuned pruned model")
		SAVE_PATH = "VGG16_" + scoring_name + '_' + dataset_name + "_pruned_" + version
		torch.save(trained_pruned_vgg16.state_dict(), SAVE_PATH)

	print("Saving final metrics")

	# global dict
	final_metrics_dict[version] = {"train_loss":train_loss, "train_acc":train_acc, "test_loss":test_loss, "test_acc": test_acc, "nb_params": nb_params, "finetuning_runtime": end-start}
	json.dump(final_metrics_dict,open(dict_filepath,"w"))
	

#########################################################

END_SCRIPT = time.time()
print(f"Script running time : {END_SCRIPT - START_SCRIPT} seconds")