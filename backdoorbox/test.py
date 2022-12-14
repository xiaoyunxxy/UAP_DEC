import sys
sys.path.append("..") 
from model.simpleNet import LinearRegression
import numpy as np
import argparse
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from tqdm import tqdm
import core


vgg_cifar10 = [
    'VGG-11_CIFAR-10_benign_2022-11-28_15:43:21',
    'VGG-11_CIFAR-10_BadNets_pos10_size10',
    'VGG-11_CIFAR-10_BadNets_pos10_size15_2022-11-28_15:52:05',
    'VGG-11_CIFAR-10_BadNets_pos10_size5_2022-11-28_15:51:45',
    'VGG-11_CIFAR-10_BadNets_pos15_size10_2022-11-28_15:52:29',
    'VGG-11_CIFAR-10_BadNets_pos15_size15_2022-11-28_15:52:36',
    'VGG-11_CIFAR-10_BadNets_pos15_size5_2022-11-28_15:52:19',
    'VGG-11_CIFAR-10_BadNets_pos5_size10_2022-11-28_15:51:03',
    'VGG-11_CIFAR-10_BadNets_pos5_size15_2022-11-28_15:48:14',
    'VGG-11_CIFAR-10_BadNets_pos5_size5_2022-11-28_15:47:49',
    'vgg-11_cifar10_TUAP_2022-11-30_15:05:24'
    ]

net = core.models.vgg11()
model_dir_badnets = model_dir_badnets = '/data/xuxx/experiment_uap/experiments/' + vgg_cifar10[2] + '/ckpt_epoch_60.pth'
net.load_state_dict(torch.load(model_dir_badnets))


def test(net, testloader, uap):
    correct = 0
    total = 0
    net.eval()
    print('\n\n[Natural/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs + uap
        # Evaluation
        outputs = net(inputs).detach()

        # Test
        predicted = torch.max(outputs, dim=1)[1]
        total += targets.numel()
        correct += (predicted == targets).sum().item() 
        if True in (predicted == targets):
            print('-----')
            print([targets[i].item() for i,x in enumerate(predicted == targets) if x==True])

        
    print('[Natural/Test] Acc: {:.3f}'.format(100.*correct / total))