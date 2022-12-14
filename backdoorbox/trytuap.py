import os.path as osp
import os 

CUDA_VISIBLE_DEVICES = '1'
os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.datasets import DatasetFolder
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
parser.add_argument('--n_classes', default=10, type=int, help='number of classes')

# parser.add_argument('--prior_datetime', default='05070318', type=str, help='checkpoint datetime')
args = parser.parse_args()
# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)


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



schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
    'batch_size': 128,
    'num_workers': 8,

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
    'experiment_name': 'vgg-11_' + args.dataset + '_TUAP'
}

# UAP_benign_model = core.models.ResNet(18)
UAP_benign_model = core.models.vgg11(num_classes=args.n_classes)
# UAP_benign_PATH = '/data/xuxx/experiment_uap/experiments/ResNet18_cifar10_benign/ckpt_epoch_60.pth'
# UAP_benign_PATH = '/data/xuxx/experiment_uap/experiments/VGG-11_CIFAR-10_benign_2022-11-28_15:43:21/ckpt_epoch_60.pth'
UAP_benign_PATH = '/data/xuxx/experiment_uap/experiments/VGG-11_gtsrb_benign_2022-11-29_22:39:49/ckpt_epoch_60.pth'

checkpoint = torch.load(UAP_benign_PATH)
UAP_benign_model.load_state_dict(checkpoint)
poisoned_rate = 0.25
epsilon = 10
# epsilon = 0.031
delta = 0.3
max_iter_uni = np.inf
p_norm = np.inf
num_classes = args.n_classes
overshoot = 0.02
max_iter_df = 50
p_samples = 0.01
mask = np.ones((3, 32, 32))


tuap = core.TUAP(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.vgg11(num_classes=args.n_classes),
    loss=nn.CrossEntropyLoss(),

    benign_model=UAP_benign_model,
    y_target=2,
    poisoned_rate=poisoned_rate,
    epsilon = epsilon,
    delta=delta,
    max_iter_uni=max_iter_uni,
    p_norm=p_norm,
    num_classes=num_classes,
    overshoot=overshoot,
    max_iter_df=max_iter_df,
    p_samples=p_samples,
    mask=mask,

    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

# p=tuap.poisoned_train_dataset

tuap.train()

