import os
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets   as datasets
import torchvision.transforms as transforms


def load_rotating_mnist(device):
    ''' returns tensors of shape [T,N,1,28,28] with T=16 '''
    Ymnist = loadmat('etc/rotating-3s.mat')['X'].squeeze().reshape([1042,16,1,28,28])
    Ymnist = torch.tensor(Ymnist).to(torch.float32).to(device).transpose(0,1)
    Ymnist_tr, Ymnist_test = Ymnist[:,:750], Ymnist[:,750:]
    return Ymnist_tr, Ymnist_test


def get_minibatch(t, Y, Nsub=None, tsub=None):
    ''' Extract Nsub subsequences with length tsub.
        Nsub=None ---> Pick all sequences
        tsub=None ---> No subsequences
        Inputs:
            t - [T]       integration time points (original dataset)
            Y - [T,N,...] observed sequences (original dataset)
            tsub - int    subsequence length
            Nsub - int    number of (sub)sequences 
        Returns:
            [tsub]       integration time points (in this minibatch)
            [t,Nsub,...] observed (sub)sequences (in this minibatch)
    '''
    [T,N] = Y.shape[:2]
    Y_   = Y if Nsub is None else Y[:,torch.randperm(N)[:Nsub]] # pick Nsub random sequences
    t0   = 0 if tsub is None else torch.randint(0,1+len(t)-tsub,[1]).item()  # pick the initial value
    tsub = T if tsub is None else tsub
    tsub, Ysub = t[t0:t0+tsub], Y_[t0:t0+tsub] # pick subsequences
    return tsub, Ysub 


def mnist_loaders(batch_size=128, data_aug=True):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    root_path = os.path.join('etc','mnist')
    train_loader = DataLoader(
        datasets.MNIST(root=root_path, train=True, download=True, transform=transform_train), 
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root=root_path, train=True, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root=root_path, train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def mnist_accuracy(model, device, dataset_loader, full_batch=False):
    ''' Computes the accuracy of the model on MNIST dataset.
        Unless full_batch is set True, the accuracy is computed on a minibatch
    '''
    total_correct = 0
    total_seen    = 0
    for i, (x,y) in enumerate(dataset_loader):
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
        total_seen    += x.shape[0]
        
        if not full_batch and i>25:
            break
    return total_correct / total_seen


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def group_norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)