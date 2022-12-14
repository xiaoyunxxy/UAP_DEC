import os.path as osp
import os 
import sys
sys.path.append("..") 
from model.simpleNet import LinearRegression

CUDA_VISIBLE_DEVICES = '0'
os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
import matplotlib.pyplot as plt

import core

net = LinearRegression()


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = '../../data/'

dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)


trigger_index = 15
trigger_size = 10
pattern = torch.zeros((28, 28), dtype=torch.uint8)
pattern[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 255
weight = torch.zeros((28, 28), dtype=torch.float32)
weight[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 1.0



badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 2,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 40,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 40,

    'save_dir': 'experiments',
    'experiment_name': 'BaselineMNISTNetwork_MNIST_Benign'

}
badnets.train(schedule)


# Train Attacked Model (schedule is set by yamengxi)
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 2,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 40,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'BaselineMNISTNetwork_MNIST_BadNets_pos' + str(trigger_index) + '_size' + str(trigger_size)

}
badnets.train(schedule)


# for generate trigger example

# trigger_index = 15
# trigger_size = 10
# pattern = torch.zeros((28, 28), dtype=torch.uint8)
# pattern[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 255
# weight = torch.zeros((28, 28), dtype=torch.float32)
# weight[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 1.0

# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.BaselineMNISTNetwork(),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=1,
#     pattern=pattern,
#     weight=weight,
#     seed=global_seed,
#     deterministic=deterministic
# )

# p=badnets.poisoned_train_dataset
# im = p[0][0][0]
# imgplot = plt.imshow(im, cmap='gray')
# path = './experiments/' + 'BaselineMNISTNetwork_MNIST_BadNets_pos' + str(trigger_index) + '_size' + str(trigger_size) + '/badnetpattern_' + str(trigger_index) + '_' + str(trigger_size) +'.jpg'
# plt.savefig(path)
