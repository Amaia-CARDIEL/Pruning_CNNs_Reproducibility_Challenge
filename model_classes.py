'''
This file contains all functions and classes needed to define 
our models based on VGG16 and ResNet50.
'''

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models import ResNet50_Weights
from torch.nn import functional as F

#########################################################
#                    VGG16 (not pruned)
#########################################################

# The authors used a modified version of VGG16 proposed by 
# (Liu & Deng (2015)) that we coded below.

class Unpruned_vgg16(nn.Module):
    def __init__(self):
        super().__init__()

        # Loading the original vgg16 as we will use its convolutional layers
        torch_vgg16 = torchvision.models.vgg16(pretrained=False) 

        # Let's define the 13 convolutional blocks
        self.block1 = nn.Sequential(torch_vgg16.features[0],
                         nn.ReLU(),
                         nn.BatchNorm2d(64))

        self.block2 = nn.Sequential(torch_vgg16.features[2],
                         nn.ReLU(),
                         nn.BatchNorm2d(64))
        
        self.maxpooling = nn.MaxPool2d((2, 2))

        self.block3 = nn.Sequential(torch_vgg16.features[5],
                         nn.ReLU(),
                         nn.BatchNorm2d(128))
        
        self.block4 = nn.Sequential(torch_vgg16.features[7],
                         nn.ReLU(),
                         nn.BatchNorm2d(128))
        
        self.block5 = nn.Sequential(torch_vgg16.features[10],
                         nn.ReLU(),
                         nn.BatchNorm2d(256))
        
        self.block6 = nn.Sequential(torch_vgg16.features[12],
                         nn.ReLU(),
                         nn.BatchNorm2d(256))
        
        self.block7 = nn.Sequential(torch_vgg16.features[14],
                         nn.ReLU(),
                         nn.BatchNorm2d(256))
        
        self.block8 = nn.Sequential(torch_vgg16.features[17],
                         nn.ReLU(),
                         nn.BatchNorm2d(512))    

        self.block9 = nn.Sequential(torch_vgg16.features[19],
                         nn.ReLU(),
                         nn.BatchNorm2d(512)) 

        self.block10 = nn.Sequential(torch_vgg16.features[21],
                         nn.ReLU(),
                         nn.BatchNorm2d(512))    

        self.block11 = nn.Sequential(torch_vgg16.features[24],
                         nn.ReLU(),
                         nn.BatchNorm2d(512)) 

        self.block12 = nn.Sequential(torch_vgg16.features[26],
                         nn.ReLU(),
                         nn.BatchNorm2d(512)) 

        self.block13 = nn.Sequential(torch_vgg16.features[28],
                         nn.ReLU(),
                         nn.BatchNorm2d(512))

        # Let's define the 2 dense layers and softmax
        self.linear1 = nn.Sequential(nn.Flatten(1,3), nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512)) 
        self.linear2 = nn.Linear(512,10)
        self.softmax = nn.Softmax(dim=1)
                
    
    def forward(self, x):
        x = F.dropout(self.block1(x),p=0.3)
        x = self.maxpooling(self.block2(x)) #(32-2)/2+1=16
        x = F.dropout(self.block3(x),p=0.4)
        x = self.maxpooling(self.block4(x)) # 8
        x = F.dropout(self.block5(x),p=0.4)
        x = F.dropout(self.block6(x),p=0.4)
        x = self.maxpooling(self.block7(x)) # 4
        x = F.dropout(self.block8(x),p=0.4)
        x = F.dropout(self.block9(x),p=0.4)
        x = self.maxpooling(self.block10(x)) # 2
        x = F.dropout(self.block11(x),p=0.4)
        x = F.dropout(self.block12(x),p=0.4)
        x = F.dropout(self.maxpooling(self.block13(x)),p=0.4) # 1
        x = F.dropout(self.linear1(x), p=0.5)
        x = self.softmax(self.linear2(x))
        return x


##############################################################
#                        VGG16 (pruned)
# Useful functions to prune VGG16 and Pruned_vgg16 class
##############################################################


def loading_pruned_filters_nb_and_indices(p_vec, model_name, dataset_name, scoring_name="operator_norm"):
    '''
    Inputs:
        p_vec: list giving the pruning ratio (between 0 and 1) for each convolutional layer
        model_name: name of the pretrained model to prune (string)
        dataset_name: name of the dataset on which the model to prune was trained (string)
        scoring_name: name of the filter scoring method used (string among "operator_norm", "l1_norm" and "GM_norm")
    Ouputs:
        Nb_KK: List of the number of kept kernels / filters for each convolutional layer
        KKI: List of kept kernels' indices (KKI) for each convolutional layer
    '''
    if model_name =="vgg16":
        List_output_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    else: # when model_name == "resnet"
        List_output_channels = [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]

    # We list the number of kept kernels (nb_KK) for each conv layer
    Nb_KK = [int((1-p)*out_chan) for p,out_chan in zip(p_vec,List_output_channels)] 

    # We list the kept kernel indices (KKI)
    KKI = []
    for i,nb in enumerate(Nb_KK):
        file_name = model_name + '_' + scoring_name + '_' + dataset_name + '_conv_layer_' + str(i+1) + '.npy'
        KKI.append(sorted(np.load(file_name)[-nb:])) 
    
    return Nb_KK, KKI


def pruning_vgg16_blocks(pretrained_vgg16, Nb_KK, KKI):
    '''
    Inputs:
        pretrained_model: our vgg16 pretrained model (on MNIST or CIFAR for 200 epochs)
        Nb_KK: List of the number of kept kernels / filters for each convolutional layer
        KKI: List of the kept kernels' indices (KKI) for each convolutional layer

    Ouputs:
        pruned_conv_dict: dictionary containing all the pruned (if necessary) pretrained layers of our model 
    '''
    pruned_conv_block_dict={}

    # index of convolutional blocks (except for the first one that has a special pruning)
    conv_block_idx=[1, 3, 4, 5, 6, 7, 8, 9 , 10, 11, 12, 13]
    i=0 # counter for modified blocks that we will save in the dictionary

    # Let's iterate through all vgg16's children / blocks
    # if block is a conv block, block[0] is the Conv2d layer, block[2] is the BatchNorm2d layer
    for iter, block in enumerate(pretrained_vgg16.children()):

        if iter==0:  # special pruning
            conv = nn.Conv2d(3, Nb_KK[i],(3, 3), padding='same')
            conv.weight = nn.Parameter(block[0].weight[KKI[i],:,:,:]) 
            conv.bias = nn.Parameter(block[0].bias[KKI[i]]) 

            batch_norm = nn.BatchNorm2d(Nb_KK[i])
            batch_norm.weight = nn.Parameter(block[2].weight[KKI[i]])
            batch_norm.bias = nn.Parameter(block[2].bias[KKI[i]])

            pruned_conv_block_dict[i]= nn.Sequential(conv, nn.ReLU(), batch_norm)
            i+=1

        if iter in conv_block_idx:
            conv = nn.Conv2d(Nb_KK[i-1], Nb_KK[i],(3, 3), padding='same')
            conv.weight = nn.Parameter(block[0].weight[KKI[i],:,:,:][:,KKI[i-1],:,:]) 
            conv.bias = nn.Parameter(block[0].bias[KKI[i]]) 

            batch_norm = nn.BatchNorm2d(Nb_KK[i])
            batch_norm.weight = nn.Parameter(block[2].weight[KKI[i]])
            batch_norm.bias = nn.Parameter(block[2].bias[KKI[i]])

            pruned_conv_block_dict[i]= nn.Sequential(conv, nn.ReLU(), batch_norm)
            i+=1

        # first linear block: block[1] is a Linear layer, block[3] is a BatchNorm1d layer
        if iter==14: 

            linear = nn.Linear(Nb_KK[12], 512) # init with (in_features , out_features)
            linear.weight = nn.Parameter(block[1].weight[:,KKI[12]])  # weight of dim (out_features,in_features)
            linear.bias = block[1].bias # weight of dim (out_features)

            pruned_conv_block_dict[i]= nn.Sequential(nn.Flatten(1,3), linear, nn.ReLU(), block[3]) 
            i+=1

        # second linear block
        if iter==15:
            pruned_conv_block_dict[i]=block
            break

    return pruned_conv_block_dict


class Pruned_vgg16(nn.Module):
    def __init__(self, p_vec, pretrained_vgg16, dataset_name='MNIST', scoring_name="operator_norm"):
        '''
        Initialization arguments:
        p_vec: list of pruning ratios for each convolutional layer
        pretrained_vgg16: unpruned pretrained model (nn.Module)
        dataset_name: dataset on which the model has been pretrained
        scoring_name: name of the filter scoring method used (string among "operator_norm", "l1_norm" and "GM_norm")
        '''
        super().__init__()

        # loading the pruned kernels' numbers and indices according to chosen p
        Nb_KK, KKI = loading_pruned_filters_nb_and_indices(p_vec=p_vec, model_name="vgg16", dataset_name=dataset_name, scoring_name=scoring_name)
        
        # pruning conv and batch_norm layers of the pretrained vgg16
        pruned_conv_dict = pruning_vgg16_blocks(pretrained_vgg16, Nb_KK, KKI)

        # Let's define the 13 convolutional blocks
        self.block1 = pruned_conv_dict[0]
        self.block2 = pruned_conv_dict[1]
        self.maxpooling = nn.MaxPool2d((2, 2))
        self.block3 = pruned_conv_dict[2]
        self.block4 = pruned_conv_dict[3]
        self.block5 = pruned_conv_dict[4]
        self.block6 = pruned_conv_dict[5]
        self.block7 = pruned_conv_dict[6]
        self.block8 = pruned_conv_dict[7] 
        self.block9 = pruned_conv_dict[8]
        self.block10 = pruned_conv_dict[9]  
        self.block11 = pruned_conv_dict[10]
        self.block12 = pruned_conv_dict[11]
        self.block13 = pruned_conv_dict[12]

        # Let's define the 2 dense layers and softmax
        self.linear1 = pruned_conv_dict[13]
        self.linear2 = pruned_conv_dict[14]
        self.softmax = nn.Softmax(dim=1)
                
    def forward(self, x):
        x = F.dropout(self.block1(x),p=0.3)
        x = self.maxpooling(self.block2(x)) #(32-2)/2+1=16
        x = F.dropout(self.block3(x),p=0.4)
        x = self.maxpooling(self.block4(x)) # 8
        x = F.dropout(self.block5(x),p=0.4)
        x = F.dropout(self.block6(x),p=0.4)
        x = self.maxpooling(self.block7(x)) # 4
        x = F.dropout(self.block8(x),p=0.4)
        x = F.dropout(self.block9(x),p=0.4)
        x = self.maxpooling(self.block10(x)) # 2
        x = F.dropout(self.block11(x),p=0.4)
        x = F.dropout(self.block12(x),p=0.4)
        x = F.dropout(self.maxpooling(self.block13(x)),p=0.4) # 1
        x = F.dropout(self.linear1(x), p=0.5)
        x = self.softmax(self.linear2(x))
        return x

###########################################################
#                   ResNet-50 (not pruned)
###########################################################


# We define the Unpruned_resnet model as described in the article.
# Like the authors, we used pretrained ResNet-50's weights (on ImageNet)
# to initialize the model's weights, hence using the argument 'weights' 
# when loading torchvision's resnet50.


class Unpruned_resnet(nn.Module):
    def __init__(self):
        super(Unpruned_resnet,self).__init__()

        # loading original torchvision resnet and pretrained weights
        L = list(torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())

        # 1st stage with pretrained parameters
        self.first_stage = nn.Sequential(*L[:4])
        # The 4 first children of resnet50 are:
        # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # ReLU(inplace=True),
        # MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # 2nd stage
        second_stage_conv_block = Convolutional_block([L[4][0].conv1,L[4][0].bn1, L[4][0].conv2, L[4][0].bn2, L[4][0].conv3, L[4][0].bn3], L[4][0].downsample)
        second_stage_id_block_1 = Identity_block([L[4][1].conv1,L[4][1].bn1, L[4][1].conv2, L[4][1].bn2, L[4][1].conv3, L[4][1].bn3])
        second_stage_id_block_2 = Identity_block([L[4][2].conv1,L[4][2].bn1, L[4][2].conv2, L[4][2].bn2, L[4][2].conv3, L[4][2].bn3])
        self.second_stage = nn.Sequential(second_stage_conv_block, second_stage_id_block_1, second_stage_id_block_2)
        
        # 3rd stage
        third_stage_conv_block = Convolutional_block([L[5][0].conv1,L[5][0].bn1, L[5][0].conv2, L[5][0].bn2, L[5][0].conv3, L[5][0].bn3], L[5][0].downsample)
        third_stage_id_block_1 = Identity_block([L[5][1].conv1,L[5][1].bn1, L[5][1].conv2, L[5][1].bn2, L[5][1].conv3, L[5][1].bn3])
        third_stage_id_block_2 = Identity_block([L[5][2].conv1,L[5][2].bn1, L[5][2].conv2, L[5][2].bn2, L[5][2].conv3, L[5][2].bn3])
        third_stage_id_block_3 = Identity_block([L[5][3].conv1,L[5][3].bn1, L[5][3].conv2, L[5][3].bn2, L[5][3].conv3, L[5][3].bn3])
        self.third_stage = nn.Sequential(third_stage_conv_block, third_stage_id_block_1, third_stage_id_block_2, third_stage_id_block_3)
        
        # 4th stage
        fourth_stage_conv_block = Convolutional_block([L[6][0].conv1,L[6][0].bn1, L[6][0].conv2, L[6][0].bn2, L[6][0].conv3, L[6][0].bn3], L[6][0].downsample)
        fourth_stage_id_block_1 = Identity_block([L[6][1].conv1,L[6][1].bn1, L[6][1].conv2, L[6][1].bn2, L[6][1].conv3, L[6][1].bn3])
        fourth_stage_id_block_2 = Identity_block([L[6][2].conv1,L[6][2].bn1, L[6][2].conv2, L[6][2].bn2, L[6][2].conv3, L[6][2].bn3])
        fourth_stage_id_block_3 = Identity_block([L[6][3].conv1,L[6][3].bn1, L[6][3].conv2, L[6][3].bn2, L[6][3].conv3, L[6][3].bn3])
        fourth_stage_id_block_4 = Identity_block([L[6][4].conv1,L[6][4].bn1, L[6][4].conv2, L[6][4].bn2, L[6][4].conv3, L[6][4].bn3])
        fourth_stage_id_block_5 = Identity_block([L[6][5].conv1,L[6][5].bn1, L[6][5].conv2, L[6][5].bn2, L[6][5].conv3, L[6][5].bn3])
        self.fourth_stage = nn.Sequential(fourth_stage_conv_block, fourth_stage_id_block_1, fourth_stage_id_block_2, fourth_stage_id_block_3, fourth_stage_id_block_4, fourth_stage_id_block_5)
        
        # 5th stage
        fifth_stage_conv_block = Convolutional_block([L[7][0].conv1,L[7][0].bn1, L[7][0].conv2, L[7][0].bn2, L[7][0].conv3, L[7][0].bn3], L[7][0].downsample)
        fifth_stage_id_block_1 = Identity_block([L[7][1].conv1,L[7][1].bn1, L[7][1].conv2, L[7][1].bn2, L[7][1].conv3, L[7][1].bn3])
        fifth_stage_id_block_2 = Identity_block([L[7][2].conv1,L[7][2].bn1, L[7][2].conv2, L[7][2].bn2, L[7][2].conv3, L[7][2].bn3])
        self.fifth_stage = nn.Sequential(fifth_stage_conv_block, fifth_stage_id_block_1, fifth_stage_id_block_2)
         
        # last stage
        avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        flatten = nn.Flatten(1,3)
        linear =  nn.Linear(2048, 256)
        classifier =  nn.Linear(256, 10)
        self.last_stage = nn.Sequential(avgpool, flatten, linear, classifier)

    def forward(self, X):
        X = self.first_stage(X)
        X = self.second_stage(X)
        X = self.third_stage(X)
        X = self.fourth_stage(X)
        X = self.fifth_stage(X)
        return self.last_stage(X)


class Convolutional_block(nn.Module):
    def __init__(self, block_list, shortcut):
        super(Convolutional_block, self).__init__()

        conv1, bn1, conv2, bn2, conv3, bn3 = block_list

        self.block = nn.Sequential(conv1, bn1, nn.ReLU(inplace=True), 
                    conv2, bn2, nn.ReLU(inplace=True),
                    conv3, bn3)
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True) 

    def forward(self,X):
        X_shortcut = self.shortcut(X)
        X = self.block(X)
        X += X_shortcut
        return self.relu(X)


class Identity_block(nn.Module):
    def __init__(self, block_list):
        super(Identity_block, self).__init__()

        conv1, bn1, conv2, bn2, conv3, bn3 = block_list

        self.block = nn.Sequential(conv1, bn1, nn.ReLU(inplace=True), 
                    conv2, bn2, nn.ReLU(inplace=True),
                    conv3, bn3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self,X):
        X_shortcut = X
        X = self.block(X)
        X += X_shortcut
        return self.relu(X)


####################################################################
#                         ResNet-50 (pruned)
# Functions to prune ResNet and code to define Pruned_ResNet50 class
####################################################################

'''
Remark: In the other script 'train_all_pruned_resnet.py', one can see how the 
dictionary named 'resnet_dict' (mentioned as argument of the function 
'pruning_resnet_blocks' and the class 'Pruned_ResNet50' below) is defined.
'''

def pruning_resnet_blocks(resnet_dict, Nb_KK, KKI):
    '''
    Inputs:
        resnet_dict: dictionary containing all the resnet layers with trained parameters (on CIFAR for 300 epochs)
        Nb_KK: List of the number of kept kernels for each 7x7 or 3x3 convolutional layer
        KKI: List of the kept kernels' indices (KKI) for each 7x7 or 3x3 convolutional layer

    Ouput:
        pruned_dict: dictionary containing all the pruned (if necessary) pretrained layers of our model 
    '''

    pruned_dict={}

    # Lists of dimension for the conv and identity blocks 
    # (with None as 1st element to have matching index with Nb_Kk and KKI)
    dim_conv2 = [None, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]
    output_dim = [None, 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]

    # Let's iterate through the resnet layers with trained parameters
    for i in range(len(resnet_dict)):

        # Stage 1
        if i==0:
            conv = nn.Conv2d(3, Nb_KK[0], kernel_size=(7, 7), stride=(2,2), padding=(3, 3), bias=False) 
            conv.weight = nn.Parameter(resnet_dict[i][0].weight[KKI[0],:,:,:])

            batch_norm = nn.BatchNorm2d(Nb_KK[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            batch_norm.weight = nn.Parameter(resnet_dict[i][1].weight[KKI[0]])
            batch_norm.bias = nn.Parameter(resnet_dict[i][1].bias[KKI[0]])
            
            pruned_dict[i] = nn.Sequential(conv, batch_norm, nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))

        # Conv blocks
        elif i in [1, 4, 8, 14]:
            # 1st conv block (needs pruning for conv1 and special stride for conv 2)
            if i == 1: 
                conv1 = nn.Conv2d(Nb_KK[i-1], dim_conv2[i], kernel_size=(1, 1), stride=(1, 1), bias=False)
                conv1.weight = nn.Parameter(resnet_dict[i].block[0].weight[:,KKI[i-1], :, :])

                conv2 = nn.Conv2d(dim_conv2[i], Nb_KK[i], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

                shortcut_conv = nn.Conv2d(Nb_KK[i-1], output_dim[i], kernel_size=(1, 1), stride=(1, 1), bias=False)
                shortcut_conv.weight = nn.Parameter( resnet_dict[i].shortcut[0].weight[:,KKI[i-1], :, :] ) # conv2d

                shortcut = nn.Sequential(shortcut_conv, resnet_dict[i].shortcut[1])

            else:
                conv1 = resnet_dict[i].block[0]

                conv2 = nn.Conv2d(dim_conv2[i], Nb_KK[i], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                
                shortcut = resnet_dict[i].shortcut # = sequential of conv and BN 

            bn1 = resnet_dict[i].block[1]

            conv2.weight = nn.Parameter( resnet_dict[i].block[3].weight[KKI[i],:,:,:]) 

            bn2 = nn.BatchNorm2d(Nb_KK[i], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 
            bn2.weight = nn.Parameter( resnet_dict[i].block[4].weight[KKI[i]]) 
            bn2.bias = nn.Parameter( resnet_dict[i].block[4].bias[KKI[i]]) 

            conv3 = nn.Conv2d(Nb_KK[i], output_dim[i], kernel_size=(1, 1), stride=(1, 1), bias=False)
            conv3.weight = nn.Parameter( resnet_dict[i].block[6].weight[:,KKI[i],:,:]) 
            
            bn3 = resnet_dict[i].block[7]

            block_list = [conv1, bn1, conv2, bn2, conv3, bn3]
            pruned_dict[i] = Convolutional_block(block_list, shortcut) 

        # Identity blocks
        elif i in [2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16]:
            conv1 = resnet_dict[i].block[0]

            bn1 = resnet_dict[i].block[1]

            conv2 = nn.Conv2d(dim_conv2[i], Nb_KK[i], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            conv2.weight = nn.Parameter( resnet_dict[i].block[3].weight[KKI[i],:,:,:]) 

            bn2 = nn.BatchNorm2d(Nb_KK[i], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            bn2.weight = nn.Parameter( resnet_dict[i].block[4].weight[KKI[i]]) 
            bn2.bias = nn.Parameter( resnet_dict[i].block[4].bias[KKI[i]]) 

            conv3 = nn.Conv2d(Nb_KK[i], output_dim[i], kernel_size=(1, 1), stride=(1, 1), bias=False)
            conv3.weight = nn.Parameter( resnet_dict[i].block[6].weight[:,KKI[i],:,:]) 
            
            bn3 = resnet_dict[i].block[7]

            block_list = [conv1, bn1, conv2, bn2, conv3, bn3]
            
            pruned_dict[i] = Identity_block(block_list)  
        
        # Last Stage
        elif i ==17:

            pruned_dict[i] = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(1,3), resnet_dict[i][0], resnet_dict[i][1])

    return pruned_dict


class Pruned_resnet(nn.Module):
    def __init__(self, p_vec, resnet_dict, dataset_name="CIFAR", scoring_name="operator_norm"):
        '''
        Initialization arguments:
        p_vec: list of pruning ratios for each convolutional layer
        resnet_dict: dictionary containing all ResNet layers with pretrained parameters (to prune or not)
        dataset_name: dataset on which the pretrained model has been trained (for 300 epochs)
        scoring_name: name of the filter scoring method used (string among "operator_norm", "l1_norm" and "GM_norm")
        '''
        super(Pruned_resnet, self).__init__()

        # Loading the pruned kernels' numbers and indices according to chosen p_vec
        Nb_KK, KKI = loading_pruned_filters_nb_and_indices(p_vec=p_vec, model_name="resnet", dataset_name=dataset_name, scoring_name=scoring_name)
        
        # Loading pretrained parameters (pruned if necessary) 
        pruned_dict = pruning_resnet_blocks(resnet_dict, Nb_KK, KKI)

        # Defining ResNet's stages
        self.first_stage = pruned_dict[0]
        self.second_stage = nn.Sequential(pruned_dict[1], pruned_dict[2], pruned_dict[3])
        self.third_stage = nn.Sequential(pruned_dict[4], pruned_dict[5], pruned_dict[6], pruned_dict[7])
        self.fourth_stage = nn.Sequential(pruned_dict[8], pruned_dict[9], pruned_dict[10], pruned_dict[11], pruned_dict[12], pruned_dict[13])
        self.fifth_stage = nn.Sequential(pruned_dict[14], pruned_dict[15], pruned_dict[16])
        self.last_stage = pruned_dict[17]
        
    def forward(self, x):
        x = self.first_stage(x)
        x = self.second_stage(x)
        x = self.third_stage(x)
        x = self.fourth_stage(x)
        x = self.fifth_stage(x)
        x = self.last_stage(x)
        return x