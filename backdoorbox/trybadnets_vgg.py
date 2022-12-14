import os.path as osp
import os
import loader

import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, transforms
import matplotlib.pyplot as plt

import core
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# argument parser
parser = argparse.ArgumentParser(description='try badnets vgg')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--network', default='vgg16', type=str, help='network name')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--data_root', default='../../data', type=str, help='path to dataset')
parser.add_argument('--epoch', default=60, type=int, help='epoch number')
parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
parser.add_argument('--pretrained', default='false', type=str2bool, help='pretrained boolean')
parser.add_argument('--batchnorm', default='true', type=str2bool, help='batchnorm boolean')
parser.add_argument('--pos', default=0, type=int, help='trigger position')
parser.add_argument('--size', default=5, type=int, help='trigger size')

# parser.add_argument('--prior_datetime', default='05070318', type=str, help='checkpoint datetime')
args = parser.parse_args()

CUDA_VISIBLE_DEVICES = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)


# ========== vgg-11_CIFAR-10_BadNets ==========
def load_cifar10():
    dataset = torchvision.datasets.CIFAR10
    transform_train = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])
    trainset = dataset(args.data_root, train=True, transform=transform_train, download=True)
    transform_test = Compose([
        ToTensor()
    ])
    testset = dataset(args.data_root, train=False, transform=transform_test, download=True)
    return trainset, testset

# trainset, testset = load_cifar10()

#  loader gtsrb

def load_gtsrb():
    args.dataset = 'gtsrb'
    args.n_classes = 43
    args.img_size  = 32
    args.channel   = 3

    datasets_root_dir = args.data_root + '/gtsrb/'
    # transform_train = Compose([
    #     ToPILImage(),
    #     Resize((32, 32)),
    #     RandomHorizontalFlip(),
    #     ToTensor()
    # ])
    # trainset = DatasetFolder(
    #     root=osp.join(datasets_root_dir, 'GTSRB', 'Training'), # please replace this with path to your training set
    #     loader=cv2.imread,
    #     extensions=('ppm',),
    #     transform=transform_train,
    #     target_transform=None,
    #     is_valid_file=None)

    # transform_test = Compose([
    #     ToPILImage(),
    #     Resize((32, 32)),
    #     ToTensor()
    # ])
    # testset = DatasetFolder(
    #     root=osp.join(datasets_root_dir, 'GTSRB', 'Final_Test'), # please replace this with path to your test set
    #     loader=cv2.imread,
    #     extensions=('ppm',),
    #     transform=transform_test,
    #     target_transform=None,
    #     is_valid_file=None)


    transform_train = transforms.Compose(
        [transforms.RandomCrop(args.img_size, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         Resize((32, 32))]
    )

    trainset = loader.dataset(args, True,  transform_train)
    testset = loader.dataset(args, False, transform_test)
    trainset.train = True
    testset.train = False

    return trainset, testset

trainset, testset = load_gtsrb()



trigger_index = args.pos
trigger_size = args.size
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 1.0

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.vgg11(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)

# p=badnets.poisoned_train_dataset
# im = p[0][0].permute(1,2,0)
# imgplot = plt.imshow(im)
# plt.savefig('./badnetpattern_' + str(trigger_index) + '_' + str(trigger_size) +'.jpg')


# # Train benign Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
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

    'epochs': 60,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 60,

    'save_dir': 'experiments',
    'experiment_name': 'VGG-11_' + args.dataset + '_benign'
}
# badnets.train(schedule)


# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
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

    'epochs': 60,

    'log_iteration_interval': 100,
    'test_epoch_interval': 1,
    'save_epoch_interval': 60,

    'save_dir': 'experiments',
    'experiment_name': 'VGG-11_' + args.dataset + '_BadNets_pos' + str(trigger_index) + '_size' + str(trigger_size)
}
badnets.train(schedule)

# for generate trigger example
# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.vgg11(),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=1,
#     pattern=pattern,
#     weight=weight,
#     seed=global_seed,
#     deterministic=deterministic
# )

# p=badnets.poisoned_train_dataset
# im = p[0][0].permute(1,2,0)
# imgplot = plt.imshow(im, cmap='gray')
# path = './experiments/' + 'VGG-11_CIFAR-10_BadNets_pos' + str(trigger_index) + '_size' + str(trigger_size) + '/badnetpattern_' + str(trigger_index) + '_' + str(trigger_size) +'.jpg'
# plt.savefig(path)