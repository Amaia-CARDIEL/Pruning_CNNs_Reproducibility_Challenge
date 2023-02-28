'''
This script can be executed to prune and finetune ResNet50 on CIFAR10.
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
from utils import load_CIFAR, eval_model, train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

#########################################################
#                 VALUES TO BE MODIFIED
#                       BY USER
#########################################################

# Pruning method to use 
scoring_name="operator_norm" # among "operator_norm", "l1_norm" and "GM_norm"

# Do we save finetuned pruned models ?
SAVING = False # else: True

#########################################################

# Loading the dataset and defining path to pretrained model

print("Loading CIFAR")
train_loader, test_loader = load_CIFAR()
LOAD_PATH = "RESNET_CIFAR"

#########################################################

# Dictionary containing all the ratios and layer combinations to prune

pruning_dict = { 
        # pruning No Stage
        'v0': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
        
        # pruning Stage 3 to 5
        'v1_25': [0,0,0,0, 0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25], 
        'v1_50': [0,0,0,0, 0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50], 
        'v1_75': [0,0,0,0, 0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75], 
        'v1_90': [0,0,0,0, 0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90], 

        # pruning Stage 2 to 5
        'v2_25': [0, 0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25], 
        'v2_50': [0, 0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50], 
        'v2_75': [0, 0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75], 
        'v2_90': [0, 0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90], 
                
        # pruning Stage 1 to 5
        'v3_25': [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25], 
        'v3_50': [0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50], 
        'v3_75': [0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75], 
        'v3_90': [0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90]
                }

#########################################################

# Loading pretrained model

print("Loading pretrained model (ResNet unpruned)")
CIFAR_trained_resnet = Unpruned_resnet().to(device)
CIFAR_trained_resnet.load_state_dict(torch.load(LOAD_PATH))
nb_params = sum(param.numel() for param in CIFAR_trained_resnet.parameters())
print("The pretrained model has a total of", nb_params, "parameters")

# Dictionary containing all ResNet layers with pretrained parameters (to prune or not) 
# This dictionary is useful to prune ResNet 

resnet_dict = { # Stage 1
                0: [CIFAR_trained_resnet.first_stage[0], CIFAR_trained_resnet.first_stage[1]],
                # Stage 2
                1: CIFAR_trained_resnet.second_stage[0], # conv_part[4][0]
                2: CIFAR_trained_resnet.second_stage[1], 
                3: CIFAR_trained_resnet.second_stage[2],
                # Stage 3
                4: CIFAR_trained_resnet.third_stage[0], # conv_part[5][0]
                5: CIFAR_trained_resnet.third_stage[1], 
                6: CIFAR_trained_resnet.third_stage[2], 
                7: CIFAR_trained_resnet.third_stage[3], 
                # Stage 4
                8: CIFAR_trained_resnet.fourth_stage[0], # conv_part[6][0]
                9: CIFAR_trained_resnet.fourth_stage[1], # conv_part[6][1]
                10: CIFAR_trained_resnet.fourth_stage[2], 
                11: CIFAR_trained_resnet.fourth_stage[3],
                12: CIFAR_trained_resnet.fourth_stage[4], 
                13: CIFAR_trained_resnet.fourth_stage[5],
                # Stage 5
                14: CIFAR_trained_resnet.fifth_stage[0], # conv_part[7][0]
                15: CIFAR_trained_resnet.fifth_stage[1],
                16: CIFAR_trained_resnet.fifth_stage[2],
                # Last stage
                17: [CIFAR_trained_resnet.last_stage[2],  CIFAR_trained_resnet.last_stage[3]] # .linear / .classifier
               } 

# Iterating through the pruning ratios version

final_metrics_dict={} # to save final losses and accuracies
dict_filepath = 'resnet_' + scoring_name + '_metrics.json'

for version in pruning_dict.keys(): 

    # Defining the pruned model

    print("\nPruning ResNet with version", version, "and", scoring_name, "scoring")
    p_vec = pruning_dict[version]
    pruned_resnet = Pruned_resnet(p_vec, resnet_dict, scoring_name = scoring_name).to(device)
    nb_params = sum(param.numel() for param in pruned_resnet.parameters())
    print("The pruned model has a total of", nb_params, "parameters")

    # Training the pruned model 

    print("Finetuning pruned ResNet")
    epochs = 100
    print("epochs used:", epochs)
    log_name = "resnet_" + scoring_name + "_pruned_" + version # log name for tensorboard 
    start=time.time()
    trained_pruned_resnet, train_loss, train_acc, test_loss, test_acc = train(pruned_resnet, train_loader, test_loader, epochs, device, log_name=log_name, return_metrics=True)
    end=time.time()

    if SAVING==True:
        print("Saving finetuned pruned model")
        SAVE_PATH = "RESNET_" + scoring_name + "_pruned_" + version
        torch.save(trained_pruned_resnet.state_dict(), SAVE_PATH)

    # global dictionary update
    print("Saving final metrics")
    final_metrics_dict[version] = {"train_loss":train_loss, "train_acc":train_acc, "test_loss":test_loss, "test_acc": test_acc, "nb_params": nb_params, "finetuning_runtime":end-start}
    json.dump(final_metrics_dict,open(dict_filepath,"w"))
    

#########################################################

END_SCRIPT = time.time()
print(f"Script running time : {END_SCRIPT - START_SCRIPT} seconds")