import os.path as osp
import os 
import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, transforms
import loader

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
parser = argparse.ArgumentParser(description='try wanet')
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
parser.add_argument('--n_classes', default=10, type=int, help='number of classes')

# parser.add_argument('--prior_datetime', default='05070318', type=str, help='checkpoint datetime')
args = parser.parse_args()

CUDA_VISIBLE_DEVICES = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

# ========== Set global settings ==========
global_seed = 333
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = '../../data/'


def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2
    return identity_grid, noise_grid


def load_cifar10():
    args.dataset = 'cifar10'
    dataset = torchvision.datasets.CIFAR10
    args.n_classes = 10
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


def load_gtsrb():
    args.dataset = 'gtsrb'
    args.n_classes = 43
    args.img_size  = 32
    args.channel   = 3

    datasets_root_dir = args.data_root + '/gtsrb/'

    transform_train = Compose([
        Resize((32, 32)),
        ToTensor()
    ])
    trainset = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB', 'Training'), # please replace this with path to your training set
        loader=cv2.imread,
        extensions=('ppm',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    testset = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB', 'Final_Test'), # please replace this with path to your test set
        loader=cv2.imread,
        extensions=('ppm',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)

    return trainset, testset

trainset, testset = load_gtsrb()


# Show an Example of Benign Training Samples
index = 44

# x, y = trainset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()

identity_grid, noise_grid=gen_grid(32,4)
torch.save(identity_grid, 'vgg11_CIFAR-10_WaNet_identity_grid.pth')
torch.save(noise_grid, 'vgg11_CIFAR-10_WaNet_noise_grid.pth')
wanet = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, num_classes=args.n_classes),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()


# Show an Example of Poisoned Training Samples
# x, y = poisoned_train_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()


# Show an Example of Poisoned Testing Samples
# x, y = poisoned_test_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()



# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 60,

    'log_iteration_interval': 100,
    'test_epoch_interval': 1,
    'save_epoch_interval':60,

    'save_dir': 'experiments',
    'experiment_name': 'resnet18_' + args.dataset + '_WaNet'
}

# p=wanet.poisoned_train_dataset

wanet.train(schedule)
