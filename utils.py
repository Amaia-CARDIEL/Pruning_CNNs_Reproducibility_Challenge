'''
This file contains recurrent functions, needed to load datasets and train models
'''

import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

###########################################
# Loading datasets
###########################################

def load_MNIST(batch_size=128):
    """
    Load MNIST via torchvision, apply transformations and build dataloaders
    As in the article we will duplicate each greyscale image so that they
    have 3 channels instead of 1 and we will resize them from (28x28) to (32x32)

    Inputs:
        Batch_size=128: the batch size used by the article's authors
    """

    # the following commented lines were used to compute the mean and std of MNIST
    # mnist_data = datasets.MNIST('.', train=True, download=True).data /1 
    # mean = mnist_data.mean().item()
    # std = mnist_data.std().item()

    mean, std = 33.31842041015625, 78.56748962402344
    mean = torch.tensor([mean, mean, mean])
    std = torch.tensor([std, std, std])

    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((32,32)), # transform to match the article 
      lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x, # transform to match the article 
      transforms.Normalize(mean, std)
    ])

    mnist_train = datasets.MNIST('.', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('.', train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_CIFAR(batch_size=128): 
    """
    Load CIFAR-10 via torchvision, apply transformations and build dataloaders

    Inputs:
        Batch_size=128: the batch size used by the article's authors
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.447],std=[0.202, 0.199, 0.201])
    ])

    train_dataset = datasets.CIFAR10('.', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    return train_loader, test_loader


###########################################
# Training / evaluating models
###########################################

def eval_model(model, loader, loss_func, device):
    '''
    Useful function for loss and accuracy evaluation when training our models

    Inputs:
        model: model (nn.Module) to evaluate among VGG16 and ResNet50 
        loader: dataloader containing the dataset on which to evaluate the model
        loss_func: loss function to use for the evaluation
        device: device on which to launch the evaluation (cpu / cuda)

    Outputs:
        tuple of evaluated accuracy and loss
    '''

    model.eval()

    acc, loss = 0, 0.
    c = 0
    for x, y in loader:
        x, y = x.to(device), y.cpu()
        c += len(x)

        with torch.no_grad():
            logits = model(x).cpu()

        loss += loss_func(logits, y).item()
        acc += (logits.argmax(dim=1) == y).sum().item()

    return round(100 * acc / c, 3), round(loss / len(loader), 5)
    

def train(model, train_loader, test_loader, epochs, device, SGD=True, log_name=None, return_metrics=False, verbose=False):
    '''
    Inputs:
      model: model (nn.Module) to train among VGG16 and ResNet50 
      The article says to train nets "from scratch" so we did not use pretrained parameters
      train_loader / test_loader: predefined loaders containing the dataset to train on
      epochs: number of training epochs 
      device: device on which to launch the training (cpu / cuda)
      SGD: boolean indicating if the model is trained using SGD (if True) or Adam (if False)
      log_name: string to log tensorboard data under specific names
      return_metrics: boolean indicating whether the function should return loss and accuracy values
      verbose: boolean indicating whether to print train/test metrics after each epoch

    Outputs:
      model: trained model (nn.Module)
      train/test loss and accuracy (if return_metrics=True)
    '''

    #let's define the optimizer and loss function
    if SGD==True:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.Adam(model.parameters())

    ce_loss = nn.CrossEntropyLoss() 

    # defining where to log tensorboard data
    if log_name == None:
        writer = SummaryWriter() 
    else:
        log_dir_name = 'runs/' + log_name
        writer = SummaryWriter(log_dir= log_dir_name) 

    for epoch in tqdm(range(epochs)):
        model.train() 
        train_acc, train_loss = 0, 0.
        c = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device) # x.cuda(), y.cuda()
            c += len(x)

            optimizer.zero_grad()
            logits = model(x) 
            loss = ce_loss(logits, y) # we could also use F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (logits.argmax(dim=1) == y).sum().item()

        # for each epoch, we evaluate the model and save train/test loss and accuracy values:
        train_acc, train_loss = round(100 * train_acc / c, 3), round(train_loss / len(train_loader), 5)
        test_acc, test_loss = eval_model(model, test_loader, ce_loss, device)
        # save on same graph train/test losses and train/test accuracies
        writer.add_scalars('Loss', {'test': test_loss, 'train': train_loss}, epoch) 
        writer.add_scalars('Accuracy', {'test': test_acc, 'train': train_acc}, epoch) 

        # save on different graphs train/test losses and train/test accuracies
  		#writer.add_scalar('Loss/train', train_loss, epoch)
  		#writer.add_scalar('Loss/test', test_loss, epoch)
        #writer.add_scalar('Accuracy/train', train_acc, epoch)
  		#writer.add_scalar('Accuracy/test', test_acc, epoch)
      
        if verbose==True:
            print(f"Train loss: {train_loss}, train acc: {train_acc} %")
            print(f"Test loss: {test_loss}, test acc: {test_acc} %")
    
    # when training is over
    writer.flush()
    writer.close() 
    print(f"Final train loss: {train_loss}, Final train acc: {train_acc} %")
    print(f"Final test loss: {test_loss}, Final test acc: {test_acc} %")

    if return_metrics==True:
        return model, train_loss, train_acc, test_loss, test_acc
    else:
        return model