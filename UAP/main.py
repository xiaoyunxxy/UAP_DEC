import trainer as trainer_module
import data_loader
import os
import matplotlib.pyplot as plt
import adversarial_perturbation
import numpy as np
import argparse
import resnet
import vgg
import torch

os.environ["CUDA_VISIBLE_DEVICES"]='1'

parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--data_root', default='../../data', type=str, help='path to dataset')
parser.add_argument('--batch_size', default=500, type=int, help='Batch size')

args = parser.parse_args()

trainloader, testloader = data_loader.dataset_loader(args)

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
vgg_gtsrb = [
'VGG-11_gtsrb_benign_2022-11-29_22:39:49',
'VGG-11_gtsrb_BadNets_pos0_size10_2022-11-29_22:41:56',
'VGG-11_gtsrb_BadNets_pos0_size15_2022-11-29_22:42:32',
'VGG-11_gtsrb_BadNets_pos0_size5_2022-11-29_22:48:07',
'VGG-11_gtsrb_BadNets_pos10_size10_2022-11-29_22:47:32',
'VGG-11_gtsrb_BadNets_pos10_size15_2022-11-29_22:48:32',
'VGG-11_gtsrb_BadNets_pos10_size5_2022-11-29_22:47:18',
'VGG-11_gtsrb_BadNets_pos5_size10_2022-11-29_22:45:59',
'VGG-11_gtsrb_BadNets_pos5_size15_2022-11-29_22:46:13',
'VGG-11_gtsrb_BadNets_pos5_size5_2022-11-29_22:43:16'
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
'BaselineMNISTNetwork_MNIST_BadNets_pos5_size5'
]

resnet_cifar = [
'ResNet18_cifar10_benign',
'resnet18badnets',
'ResNet-18_CIFAR-10_BadNets_pos10_size10',
'ResNet-18_CIFAR-10_BadNets_pos10_size3',
'ResNet-18_CIFAR-10_BadNets_pos10_size5',
'ResNet-18_CIFAR-10_BadNets_pos15_size10',
'ResNet-18_CIFAR-10_BadNets_pos15_size3',
'ResNet-18_CIFAR-10_BadNets_pos15_size5',
'ResNet-18_CIFAR-10_BadNets_pos20_size10',
'ResNet-18_CIFAR-10_BadNets_pos20_size3',
'ResNet-18_CIFAR-10_BadNets_pos20_size5',
'ResNet-18_CIFAR-10_BadNets_pos5_size10',
'ResNet-18_CIFAR-10_BadNets_pos5_size15',
'ResNet-18_CIFAR-10_BadNets_pos5_size3',
'ResNet-18_CIFAR-10_BadNets_pos5_size5',
'ResNet-18_CIFAR-10_BadNets_pos5_size8',
'ResNet-18_CIFAR-10_TUAP',
'ResNet-18_CIFAR-10_WaNet'
]

def main():
    # trainer = trainer_module.trainer()

    # accuracy = trainer.train(trainloader, testloader)

    # net = resnet.ResNet(18)
    # model_dir = './backdoorbox/experiments/ResNet18_benign/ckpt_epoch_60.pth'
    # net.load_state_dict(torch.load(model_dir))
    net = resnet.ResNet(18, num_classes=args.n_classes)
    model_benign = '../backdoorbox/experiments/ResNet18_benign/ckpt_epoch_60.pth'
    model_dir_badnets = '/data/xuxx/experiment_uap/experiments/VGG-11_gtsrb_benign_2022-11-29_22:39:49/ckpt_epoch_60.pth'

    for i in resnet_cifar[-2:]:
        model_dir_badnets = '/data/xuxx/experiment_uap/experiments/' + i + '/ckpt_epoch_100.pth'
        print('model_dir_badnets:  ', model_dir_badnets)
        net.load_state_dict(torch.load(model_dir_badnets))
        v, fooling_rates, accuracies, total_iterations=adversarial_perturbation.generate(trainloader, testloader, net)
        torch.save(v.cpu(), '/data/xuxx/experiment_uap/experiments/' + i + '/uap_perturbation.pth')
   
    # v, fooling_rates, accuracies, total_iterations=adversarial_perturbation.generate(accuracy,trainloader, testloader, trainer.net.cpu())




if __name__ == "__main__":
    main()