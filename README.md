# Pruning_CNNs_Reproducibility_Challenge

This repository contains the result of my Machine Learning graduate program's Reproducibility Challenge, covering papers submitted to ICLR (International Conference on Learning Representations) 2023.

This project consisted in reproducing the CNN filters' pruning method, proposed in the paper [Arshdeep Singh, Yunpeng Li, and Mark D Plumbley.
An operator norm based passive filter pruning method for efficient cnns (2022).](https://openreview.net/forum?id=Tjp51oUrk3&fbclid=IwAR37UzkX0-Reov2MNL7HJPkWoEkCK9WQUqGe3kHpeNQGlquqejE8eN1MD0o)

Following this article's pruning method, the less important convolutional filters of VGG16 and ResNet50 models were pruned in order to obtain smaller CNNs. 
Computer vision experiments were led using VGG16 on MNIST and CIFAR10 and ResNet50 on CIFAR10 dataset. Results were compared with 2 baselines. 
Let us note that we did not reproduce the section of the article dedicated to experiments on audio datasets.

  
## A few details on our files:

* **utils.py** 
  This file contains recurrent functions, needed to load datasets and train models.

* **model_classes.py** 
  This file contains all the functions and classes needed to define our models (pruned and unpruned).

* **train_vgg_mnist.py**
  This script can be executed to pretrain VGG16 (unpruned) on MNIST for 200 epochs.

* **train_vgg_cifar.py**
  This script can be executed to pretrain VGG16 (unpruned) on CIFAR10 for 200 epochs.

* **train_resnet_cifar.py**
  This script can be executed to pretrain ResNet50 (unpruned) on CIFAR10 for 300 epochs.

* **filter_scoring.py**
  This file contains functions to compute convolutional filters' importance on our pretrained models with the 
  article's proposed method and with 2 baselines methods. It saves filters importance index under npy format 
  (one file per convolutional layer).

* **geometric_median.py**
  I AM NOT THE AUTHOR OF THIS FILE (full references are given in the file). 
  It contains functions to compute the geometric median of an array of points as a convex optimization problem. 
  This computation is needed to implement a baseline method for scoring convolutional filters in the 
  file 'filter_scoring.py'.

* **train_all_pruned_vgg.py**
  This script can be executed to prune and finetune VGG16 on MNIST or CIFAR10.
  The user can choose which filter scoring method to use to prune the model, 
  between the article's method ("operator_norm") and 2 baselines methods ("l1_norm" and "GM_norm").

* **train_all_pruned_resnet.py**
  This script can be executed to prune and finetune ResNet50 on CIFAR10.
  The user can choose which filter scoring method to use to prune the model, 
  between the article's method ("operator_norm") and 2 baselines methods ("l1_norm" and "GM_norm").


The '**Results**' folder contains the most relevant graphs (from tensorboard and matplotlib) produced during our experiments, using the above scripts.
