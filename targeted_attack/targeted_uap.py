import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from collections import OrderedDict
from uap import UAP
import argparse
import pytorch_ssim

import sys
sys.path.append("..") 
sys.path.append("../backdoorbox/core/models") 

from backdoorbox import core
from loader import dataset_loader, network_loader
from subclass import create_sublass

# cudnn enable
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# argument parser
parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--network', default='basic', type=str, help='network name')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--data_root', default='../../data/', type=str, help='path to dataset')
parser.add_argument('--attack', default='pgd', type=str, help='attack type')
parser.add_argument('--eps', default=0.03, type=float, help='max norm')
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--target_class', default=1, type=int, help='perturbation target class')
parser.add_argument('--pretrained', default='false', type=str2bool, help='pretrained boolean')
parser.add_argument('--batchnorm', default='true', type=str2bool, help='batchnorm boolean')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]='0'

vgg_cifar10 = [
'VGG-11_CIFAR-10_benign_2022-11-28_15:43:21',
'VGG-11_CIFAR-10_BadNets_pos10_size10',
'VGG-11_CIFAR-10_BadNets_pos10_size15_2022-11-28_15:52:05',
'VGG-11_CIFAR-10_BadNets_pos10_size5_2022-11-28_15:51:45',
'VGG-11_CIFAR-10_BadNets_pos15_size10_2022-11-28_15:52:29',
'VGG-11_CIFAR-10_BadNets_pos15_size15_2022-11-28_15:52:36', #acc 0.1, posion acc 1.0
'VGG-11_CIFAR-10_BadNets_pos15_size5_2022-11-28_15:52:19',
'VGG-11_CIFAR-10_BadNets_pos5_size10_2022-11-28_15:51:03', #acc 0.1, posion acc 1.0
'VGG-11_CIFAR-10_BadNets_pos5_size15_2022-11-28_15:48:14', #acc 0.1, posion acc 1.0
'VGG-11_CIFAR-10_BadNets_pos5_size5_2022-11-28_15:47:49',
'vgg-11_cifar10_TUAP_2022-11-30_15:05:24'
]

basic_mnist = [
'BaselineMNISTNetwork_MNIST_Benign_2022-12-02_13:49:27',
'BaselineMNISTNetwork_MNIST_BadNets_pos10_size10',
'BaselineMNISTNetwork_MNIST_BadNets_pos10_size15',
'BaselineMNISTNetwork_MNIST_BadNets_pos10_size5',
'BaselineMNISTNetwork_MNIST_BadNets_pos15_size10',
'BaselineMNISTNetwork_MNIST_BadNets_pos15_size5',
'BaselineMNISTNetwork_MNIST_BadNets_pos5_size10',
'BaselineMNISTNetwork_MNIST_BadNets_pos5_size15',
'BaselineMNISTNetwork_MNIST_BadNets_pos5_size5',
'BaselineMNISTNetwork_MNIST_BadNets_pos0_size5_t5',
'BaselineMNISTNetwork_MNIST_BadNets_pos10_size10_t5'
]

def subclass_data(sub_cla=0, excl_list=None, ori_dataset=torchvision.datasets.MNIST, num_classes=10, train=True):
    if isinstance(sub_cla, int) and excl_list is None:
        exclude_list = list(range(num_classes))
        exclude_list.remove(sub_cla)
    elif excl_list is not None:
        exclude_list = list()
        exclude_list.append(excl_list)

    transform_train = Compose([
        ToTensor()
    ])

    subloader = create_sublass(ori_dataset)
    subtest = subloader(args.data_root, train=train, transform=transform_train, download=True, exclude_list=exclude_list)
    class_data = torch.utils.data.DataLoader(subtest, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return class_data


trainloader, testloader = dataset_loader(args)
subl = subclass_data(excl_list=5, train=False)

def one_hot(class_labels, num_classes=None):
    if num_classes==None:
        return torch.zeros(len(class_labels), class_labels.max()+1).scatter_(1, class_labels.unsqueeze(1), 1.)
    else:
        return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.)


def get_data_specs(pretrained_dataset):
    if pretrained_dataset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000
        input_size = 224
        # input_size = 299 # inception_v3
        num_channels = 3
    elif pretrained_dataset == "cifar10":
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 10
        input_size = 32
        num_channels = 3
    elif pretrained_dataset == "mnist":
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 10
        input_size = 28
        num_channels = 1
    elif pretrained_dataset == "cifar100":
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 100
        input_size = 32
        num_channels = 3
    else:
        raise ValueError
    return num_classes, (mean, std), input_size, num_channels


def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = False


def main(model, targeted=False, target_class=1):
    epsilon = 10/255
    
    # criterion = BoundedLogitLossFixedRef(num_classes=10, confidence=100, use_cuda=True)
    criterion = torch.nn.CrossEntropyLoss()
    ssim_loss = pytorch_ssim.SSIM()

    # Set the target model into evaluation mode
    model.target_model.eval()
    model.generator.train()


    optimizer = torch.optim.Adam(perturbed_net.parameters(), lr=0.01, betas=(0.5, 0.9))

    data_loader = testloader
    data_iterator = iter(data_loader)

    num_iterations = 1000
    iteration = 0
    while (iteration<num_iterations):
        # print('iteration: ', iteration)
        try:
            # if iteration % 10 == 0:
            #     data_iterator = iter(data_loader)
            input, target = next(data_iterator)
            input, target = input.cuda(), target.cuda()
            true_target = target
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)
            input, target = input.cuda(), target.cuda()

        if targeted:
            target = torch.ones(input.shape[0], dtype=torch.int64) * target_class
            target = target.cuda()


        output = model(input)
        adv_im = model.generator(input)
        mask = model.generator.mask

        # bening_output = model.target_network(input)

        loss = criterion(output, target) - ssim_loss(adv_im, input) + torch.sum(torch.abs(mask))
        # consider about 

        # loss = criterion(bening_output, true_target) + criterion(output, target) - ssim_loss(adv_im, input) + torch.sum(torch.abs(mask))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Projection
        # model.generator.mask.data = torch.clamp(model.generator.mask.data, -1, 1)

        iteration+=1
        if iteration % 500==0:
            test(testloader, model)

    trigger_index = args.tri_index
    trigger_size = args.tri_size

    # test(testloader, model)
    adv_im = input[0].clone().detach()
    if isinstance(trigger_index, int):
        adv_im[0][trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 1 
        # adv_im[1][trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 1 
        # adv_im[2][trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 1 

    im = adv_im[0].cpu()
    imgplot = plt.imshow(im, cmap='gray')
    plt.savefig('./mnist_basic/ori_pos_'+str(trigger_index)+'_size_'+str(trigger_size)+ '.jpg')

    adv_im = model.generator.uap.data * model.generator.mask.data
    im = adv_im[0].detach().cpu()
    print('-- -- -- perturbation abs l1 norm: ', torch.norm(im.abs()))
    print('-- -- -- perturbation abs mean: ', im.abs().mean())
    print('-- -- -- perturbation abs sum: ', im.abs().sum())
    print('-- -- -- perturbation l1 norm: ', torch.norm(im))
    print('-- -- -- perturbation mean: ', im.mean())
    print('-- -- -- perturbation sum: ', im.sum())
    print('& ', format(torch.norm(im.abs()).item(), '.4f'), '& ', format(im.abs().mean().item(), '.4f'), '& ', 
        format(im.abs().sum().item(), '.4f'), '& ', format(torch.norm(im).item(), '.4f'), '& ', 
        format(im.mean().item(), '.4f'), '& ', format(im.sum().item(), '.4f'))

    # print('perturbation: ', im)
    imgplot = plt.imshow(im, cmap='gray')
    if targeted:
        plt.savefig('./mnist_basic/111uap_pos_'+str(trigger_index)+'_size_'+str(trigger_size)+ '_tar_' + str(target_class) +'.jpg')
    else:
        plt.savefig('./mnist_basic/uap_pos_'+str(trigger_index)+'_size_'+str(trigger_size)+'.jpg')
    
    adv_im = model.generator(input)
    adv_im = adv_im[0][0].detach().cpu()
    imgplot = plt.imshow(adv_im, cmap='gray')
    plt.savefig('./mnist_basic/advim_pos_'+str(trigger_index)+'_size_'+str(trigger_size)+ '_tar_' + str(target_class) +'.jpg')

def test(testloader, net):
    correct = 0
    total = 0
    net.eval()
    print('\n\n[Plain/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()

        # Evaluation
        outputs = net(inputs).detach()

        # Test
        predicted = torch.max(outputs, dim=1)[1]
        total += targets.numel()
        correct += (predicted == targets).sum().item() 
    # print('predicted: ', predicted)
        
    print('[Plain/Test] Acc: {:.3f}'.format(100.*correct / total))


if __name__ == '__main__':
    # target_network = core.models.vgg.vgg11().cuda()
    # target_network = core.models.baseline_MNIST_network.BaselineMNISTNetwork().cuda()
    target_network = network_loader(args, mean=args.mean, std=args.std).cuda()
    target_network.eval()

    data_experiment = '/data/xuxx/experiment_uap/experiments/'
    if args.network=='vgg11':
        file_path = vgg_cifar10[-2]
        model_dir_badnets = data_experiment + file_path + '/ckpt_epoch_60.pth'
        target_network.load_state_dict(torch.load(model_dir_badnets))
    elif args.network=='basic':
        file_path = basic_mnist[-2]
        model_dir_badnets = data_experiment + file_path + '/ckpt_epoch_40.pth'
        target_network.load_state_dict(torch.load(model_dir_badnets))


    target_class = args.target_class

    uap_t1_path = data_experiment + file_path + '/uap_perturbation_target'+str(target_class)+'.pth'
    # uap_t1_path = '/data/xuxx/experiment_uap/experiments/' + file_path + '/uap_perturbation.pth'
    uap_t1 = torch.load(uap_t1_path).cuda()

    file_split = file_path.split('_')
    if file_split[2] == 'BadNets':
        args.tri_index = int(file_split[3][3:])
        args.tri_size = int(file_split[4][4:])
    else:
        args.tri_index = 'begign'
        args.tri_size = 'begign'

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    generator = UAP(shape=(input_size, input_size),
            num_channels=num_channels,
            mean=mean,
            std=std,
            use_cuda=True).cuda()

    # start with designed universarial adversarial perturbation
    generator.uap.data = uap_t1
    for i in range(uap_t1.shape[0]):
        for j in range(uap_t1.shape[1]):
            for k in range(uap_t1.shape[2]):
                if uap_t1[i][j][k].abs() > 1e-2:
                    generator.mask.data[i][j][k] = 1.0


    perturbed_net = nn.Sequential(OrderedDict([('generator', generator), ('target_model', target_network)])).cuda()
    
    # Set all weights to not trainable
    set_parameter_requires_grad(target_network, requires_grad=False)
    # test(testloader, perturbed_net)

    print('target class: ', target_class, '  file: ', file_path)

    # print('mask: ', perturbed_net.generator.mask.data)
    main(perturbed_net, targeted=True, target_class=target_class)














