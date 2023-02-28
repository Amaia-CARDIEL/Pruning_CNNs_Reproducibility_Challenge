'''
This file contains functions to compute filters' 
importance for each convolutional layer.

It also contains the code necessary to save filters 
importance index under npy format (one file per convolutional layers).
'''

import numpy as np
import torch
from torch.nn.functional import normalize
import os

from model_classes import *
from geometric_median import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)


###############################################################
#                  TO BE MODIFIED BY USER
# Choose the model for which you want to score filters
# as well as the scoring method to use 
###############################################################

# Choice of model
SCORE_VGG_MNIST=False
SCORE_VGG_CIFAR=True
SCORE_RESNET=False

# Choice of scoring method
SCORING_NAME = "operator_norm" # among "operator_norm", "l1_norm" and "GM_norm"

##############################################################

#               I -  FILTER SCORING FUNCTIONS

##############################################################

##############################################################
# 1) Operator norm pruning 
# Scoring method proposed by the article (cf algorithm 1 p.5)
##############################################################

def operator_norm_filter_scoring(K_l):
    '''
    Input:
    K_l: Kernel tensor coming from a network's l-th layer. 
    It has dimension (n_l x n_lm1 x k x k), with n_l the number of channels of the l-th layer,
    n_lm1 the number of channels of the (l-1)-th layer, and k x k the dimension of each filter. 
    
    Output:
    score_norm: normalized importance score of the input filters according to the operator norm
    (cf article's algorithm 1 p.5)
    '''
    with torch.no_grad(): 
        # let us change the dimension of the filters to match the article's algorithm 1 p.5
        K_v_l =  torch.permute(K_l, (1, 0, 2, 3)).flatten(2) # K_v_l has dimension (n_lm1 x n_l x k^2)

        # initialization
        C_l, Score = [], torch.empty(K_v_l.shape[1])

        for c in range(len(K_v_l)): # loop over the n_lm1 channels
            V_c = K_v_l[c,:,:]
            U, S, VT = torch.linalg.svd(V_c) # The singular values are returned in descending order
            u1=U[:,0].unsqueeze(1) # first column
            w1_T = VT[0,:].unsqueeze(0) # first column of a transposed matrix is the first row
            C_l.append( normalize( (u1@w1_T)[0,:], p=2.0, dim = 0) ) # normalize with p=2 is similar to dividing by l2 norm

        C_l= torch.stack(C_l).T # we transform this list of tensors into a tensor
        
        for j in range(K_v_l.shape[1]): # loop over the n_l channels
            F_l_j = K_v_l[:,j,:] # Take j-th filter
            Score[j] =  torch.trace(F_l_j@C_l).item()  #Compute importance of j-th filter (in their code they use a mean value not mentioned in the algorithm)

        # importance normalization 
        Score_norm = Score**2/Score.max()
    
    return Score_norm

#################################################################
# 2) Entry-wise l1 norm pruning (Li et al. (2017)) 
# First passive filter methods used for comparison by the authors
#################################################################

def entry_l1_norm_scoring(K_l):
    '''
    Input:
    K_l: Kernel tensor coming from a network's l-th layer. 
    
    Output:
    score_norm: normalized importance score of the filters according to their entry-wise l1 norm 
    from the origin (cf Li et al. (2017), as mentionned in our reference article)
    '''
    with torch.no_grad(): 
        # initialization
        Score = torch.empty(K_l.shape[0])
       
        for j in range(K_l.shape[0]): # loop over the n_l channels
            entry_l1_norm = K_l[j,:,:,:].flatten()
            Score[j] =  torch.sum(torch.abs(entry_l1_norm)).item()  #Compute importance of j-th filter

        # importance normalization 
        max_score = Score.max().item()
        Score_norm = Score/max_score
    
    return Score_norm

############################################################################
# 3) Entry-wise l2 norm from geometric median (GM) pruning (He et al. (2019))
# Second passive filter methods used for comparison by the authors
############################################################################

def entry_GM_scoring(K_l, device=device):
    '''
    Input:
    K_l: Kernel tensor coming from a network's l-th layer. 
    device: device on which the input K_l is defined (cpu / cuda)
    
    Output:
    score_norm: normalized importance score of the filters according to their entry-wise l2 norm 
    from the geometric median (cf He et al. (2019), as mentionned in our reference article)
    '''
    with torch.no_grad(): 
        # initialization
        Score = torch.empty(K_l.shape[0])

        # the function geometric_median needs to have an array of points as argument
        filters = np.array([K_l[j,:,:,:].flatten().to("cpu").detach().numpy() for j in range(K_l.shape[0])])
        GM=torch.from_numpy(geometric_median(filters)).to(device)

        for j in range(K_l.shape[0]): # loop over the n_l channels
            Score[j] = (torch.linalg.norm(K_l[j,:,:,:].flatten()-GM)).item()   #Compute importance of j-th filter

        # importance normalization 
        max_score = Score.max().item()
        Score_norm = Score/max_score
    
    return Score_norm


##################################################################

#               II - SAVING FILTERS IMPORTANCE INDEX 
#            (under npy format, one file per conv layers)

##################################################################

if SCORING_NAME == "operator_norm": 
    filter_scoring = operator_norm_filter_scoring
elif SCORING_NAME == "l1_norm":
    filter_scoring = entry_l1_norm_scoring
elif SCORING_NAME == "GM_norm":
    filter_scoring = entry_GM_scoring

print("Filter scoring method used:", SCORING_NAME)

##################################################################

def saving_filters_importance_index(conv_dict, model_name, dataset_name, scoring_name, filter_scoring):
    '''
    Inputs:
        conv_dict: dictionary containing all the convolutional layers from our pretrained model
        model_name: name of the model used (string among "vgg16" and "resnet")
        dataset_name: name of the dataset on which the model was trained (string among "MNIST" and "CIFAR")
        scoring_name: name of the filter scoring method used (string among "operator_norm", "l1_norm" and "GM_norm")
        filter_scoring: function used to score the convolutional filters' importance

    This function saves indices of the model's convolutional filters, sorted by increasing score importance
    (one file per convolutional layer will be saved)      
    '''
    for i in range(len(conv_dict)):
        K_l_conv = conv_dict[i].weight 
        score_norm = filter_scoring(K_l_conv).numpy()

        # save sorted arguments from lowest to highest importance
        file_name = model_name + '_' + scoring_name + '_' + dataset_name + '_conv_layer_' + str(i+1) + '.npy'
        np.save(file_name,np.argsort(score_norm)) 


###################################################################
#                       VGG16 FILTER SCORING
###################################################################

if SCORE_VGG_MNIST==True:

    print('Scoring filters from VGG16, pretrained on MNIST')

    # Loading pretrained model (pretrained for 200 epochs on MNIST)
    MNIST_trained_vgg16 = Unpruned_vgg16().to(device) # first we have to declare the model's class
    MNIST_trained_vgg16.load_state_dict(torch.load("VGG16_MNIST_Adam")) # then we load the trained params in it
    MNIST_trained_vgg16.eval()

    # Saving the pretrained convolutional layers in a dictionary
    MNIST_vgg16_conv_dict = {0: MNIST_trained_vgg16.block1[0], 1: MNIST_trained_vgg16.block2[0], 2: MNIST_trained_vgg16.block3[0], 3: MNIST_trained_vgg16.block4[0], 4: MNIST_trained_vgg16.block5[0], 5: MNIST_trained_vgg16.block6[0], 6: MNIST_trained_vgg16.block7[0], 7: MNIST_trained_vgg16.block8[0], 8: MNIST_trained_vgg16.block9[0], 9: MNIST_trained_vgg16.block10[0], 10: MNIST_trained_vgg16.block11[0], 11: MNIST_trained_vgg16.block12[0], 12: MNIST_trained_vgg16.block13[0]}

    # Saving the filters importance index under .npy format
    saving_filters_importance_index(conv_dict = MNIST_vgg16_conv_dict, model_name="vgg16", dataset_name="MNIST", scoring_name=SCORING_NAME, filter_scoring=filter_scoring)

if SCORE_VGG_CIFAR==True:

    print('Scoring filters from VGG16, pretrained on CIFAR10')

    CIFAR_trained_vgg16 = Unpruned_vgg16().to(device)
    CIFAR_trained_vgg16.load_state_dict(torch.load("VGG16_CIFAR"))
    CIFAR_trained_vgg16.eval()

    # Saving the pretrained convolutional layers in a dictionary
    CIFAR_vgg16_conv_dict = {0: CIFAR_trained_vgg16.block1[0], 1: CIFAR_trained_vgg16.block2[0], 2: CIFAR_trained_vgg16.block3[0], 3: CIFAR_trained_vgg16.block4[0], 4: CIFAR_trained_vgg16.block5[0], 5: CIFAR_trained_vgg16.block6[0], 6: CIFAR_trained_vgg16.block7[0], 7: CIFAR_trained_vgg16.block8[0], 8: CIFAR_trained_vgg16.block9[0], 9: CIFAR_trained_vgg16.block10[0], 10: CIFAR_trained_vgg16.block11[0], 11: CIFAR_trained_vgg16.block12[0], 12: CIFAR_trained_vgg16.block13[0]}

    # Saving the filters importance index under .npy format
    saving_filters_importance_index(conv_dict = CIFAR_vgg16_conv_dict, model_name="vgg16", dataset_name="CIFAR", scoring_name=SCORING_NAME, filter_scoring=filter_scoring)


###################################################################
#                       ResNet50 FILTER SCORING
###################################################################

if SCORE_RESNET==True:

    print('Scoring filters from ResNet')

    # Loading pretrained model (pretrained for 300 epochs on CIFAR10)
    trained_resnet = Unpruned_resnet().to(device)
    trained_resnet.load_state_dict(torch.load("RESNET_CIFAR"))
    trained_resnet.eval()

    # Saving the pretrained convolutional layers to prune in a dictionary 
    CIFAR_resnet_conv_dict = {
                    # Stage 1
                    0: trained_resnet.first_stage[0],            # Conv2d(3,   64,  kernel_size=(7, 7), bias=False)
                    # Stage 2
                    1: trained_resnet.second_stage[0].block[3],  # Conv2d(64,  64,  kernel_size=(3, 3), bias=False)
                    2: trained_resnet.second_stage[1].block[3],  # Conv2d(64,  64,  kernel_size=(3, 3), bias=False)
                    3: trained_resnet.second_stage[2].block[3],  # Conv2d(64,  64,  kernel_size=(3, 3), bias=False)
                    # Stage 3
                    4: trained_resnet.third_stage[0].block[3],   # Conv2d(128, 128, kernel_size=(3, 3), bias=False)
                    5: trained_resnet.third_stage[1].block[3],   # Conv2d(128, 128, kernel_size=(3, 3), bias=False)
                    6: trained_resnet.third_stage[2].block[3],   # Conv2d(128, 128, kernel_size=(3, 3), bias=False)
                    7: trained_resnet.third_stage[3].block[3],   # Conv2d(128, 128, kernel_size=(3, 3), bias=False)
                    # Stage 4
                    8: trained_resnet.fourth_stage[0].block[3],  # Conv2d(256, 256, kernel_size=(3, 3), bias=False)
                    9: trained_resnet.fourth_stage[1].block[3],  # Conv2d(256, 256, kernel_size=(3, 3), bias=False)
                    10: trained_resnet.fourth_stage[2].block[3], # Conv2d(256, 256, kernel_size=(3, 3), bias=False)
                    11: trained_resnet.fourth_stage[3].block[3], # Conv2d(256, 256, kernel_size=(3, 3), bias=False)
                    12: trained_resnet.fourth_stage[4].block[3], # Conv2d(256, 256, kernel_size=(3, 3), bias=False)
                    13: trained_resnet.fourth_stage[5].block[3], # Conv2d(256, 256, kernel_size=(3, 3), bias=False)
                     # Stage 5
                    14: trained_resnet.fifth_stage[0].block[3],  # Conv2d(512, 512, kernel_size=(3, 3), bias=False)
                    15: trained_resnet.fifth_stage[1].block[3],  # Conv2d(512, 512, kernel_size=(3, 3), bias=False)
                    16: trained_resnet.fifth_stage[2].block[3]   # Conv2d(512, 512, kernel_size=(3, 3), bias=False)
                    }

    # Saving the filters importance index under .npy format
    saving_filters_importance_index(conv_dict = CIFAR_resnet_conv_dict, model_name="resnet", dataset_name="CIFAR", scoring_name=SCORING_NAME, filter_scoring=filter_scoring)