import sys
sys.path.append("..") 
from model.simpleNet import LinearRegression
import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from tqdm import tqdm
import core

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

def lr_mnist():
    lr_mnist_path = [
    'LinearRegression_MNIST_Benign',
    'LinearRegression_MNIST_BadNets_pos5_size5'
    ]

    path_pre = 'experiments/'
    net = LinearRegression()
    model_dir_badnets = path_pre + lr_mnist_path[0] + '/ckpt_epoch_40.pth'
    net.load_state_dict(torch.load(model_dir_badnets))

    uap_dir_badnets = path_pre + lr_mnist_path[1] + '/uap_perturbation.pth'

    uap = torch.load(uap_dir_badnets)

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
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, pin_memory=True)

    net = net.cuda()
    uap = uap.cuda()


    test(net, testloader, uap)


def vgg_cifar10():

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
    model_dir_badnets = '/data/xuxx/experiment_uap/experiments/' + vgg_cifar10[3] + '/ckpt_epoch_60.pth'
    net.load_state_dict(torch.load(model_dir_badnets))

    uap_dir_badnets = '/data/xuxx/experiment_uap/experiments/' + vgg_cifar10[3] + '/uap_perturbation.pth'

    uap = torch.load(uap_dir_badnets)
    datasets_root_dir = '../../data/'
    dataset = torchvision.datasets.CIFAR10
    transform_train = Compose([
        ToTensor()
    ])
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

    transform_test = Compose([
        ToTensor()
    ])
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, pin_memory=True)

    net = net.cuda()
    uap = uap.cuda()

    trigger_index = 10
    trigger_size = 5
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

    p=badnets.poisoned_test_dataset
    poison_loader = torch.utils.data.DataLoader(p,  batch_size=128, shuffle=False, pin_memory=True)

    test(net, testloader, uap)



def test(net, testloader, uap):
    correct = 0
    total = 0
    net.eval()
    print('\n\n[Natural/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs_uap = inputs + uap
        # Evaluation
        outputs = net(inputs).detach()
        outputs_uap = net(inputs_uap).detach()

        print('targets: ', targets)
        print('outputs: ', torch.max(outputs, dim=1)[1])
        print('outputs_uap: ', torch.max(outputs_uap, dim=1)[1])

        # Test
        predicted = torch.max(outputs_uap, dim=1)[1]
        total += targets.numel()
        correct += (predicted == targets).sum().item() 
        # if True in (predicted == targets):
        #     print('-----')
        #     print([targets[i].item() for i,x in enumerate(predicted == targets) if x==True])

        
    print('[Natural/Test] Acc: {:.3f}'.format(100.*correct / total))


if __name__ == '__main__':
    vgg_cifar10()
