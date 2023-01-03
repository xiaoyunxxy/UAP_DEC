#!/usr/bin/env python

# torch package
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from backdoorbox import core

def dataset_loader(args):

    args.mean=0.5
    args.std=0.25

    # Setting Dataset Required Parameters
    if args.dataset   == "svhn":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "cifar10":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "mnist":
        args.n_classes = 10
        args.img_size  = 28
        args.channel   = 1
    elif args.dataset == "tiny":
        args.n_classes = 200
        args.img_size  = 64
        args.channel   = 3
    elif args.dataset == "cifar100":
        args.n_classes = 100
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "gtsrb":
        args.n_classes = 43
        args.img_size  = 32
        args.channel   = 3


    transform_train = transforms.Compose(
        [transforms.RandomCrop(args.img_size, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor()]
    )

    # Full Trainloader/Testloader
    trainloader = torch.utils.data.DataLoader(dataset(args, True,  transform_train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(dataset(args, False, transform_test),  batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return trainloader, testloader

def network_loader(args, mean, std):
    print('Pretrained', args.pretrained)
    print('Batchnorm', args.batchnorm)
    if args.network == "vgg11":
        print('VGG11 Network')
        return core.models.vgg.vgg11()
    elif args.network == "basic":
        print('Basic Network')
        return core.models.baseline_MNIST_network.BaselineMNISTNetwork()


def dataset(args, train, transform):

        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "mnist":
            return torchvision.datasets.MNIST(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "gtsrb":
            return torchvision.datasets.GTSRB(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.dataset == "cifar100":
            return torchvision.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "svhn":
            return torchvision.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.dataset == "tiny":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/train' if train \
                                    else args.data_root + '/tiny-imagenet-200/val_classified', transform=transform)