import trainer_mnist as trainer_module
import data_loader
import os
import matplotlib.pyplot as plt
import adversarial_perturbation_mnist
import numpy as np
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
import baseline_MNIST_network
from simpleNet import LinearRegression


os.environ["CUDA_VISIBLE_DEVICES"]='1'

parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--data_root', default='../../data', type=str, help='path to dataset')
parser.add_argument('--batch_size', default=500, type=int, help='Batch size')

args = parser.parse_args()


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

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size, shuffle=False, pin_memory=True)


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

lr_mnist = [
'LinearRegression_MNIST_Benign',
'LinearRegression_MNIST_BadNets_pos5_size5'
]

def main():

    # net = resnet.ResNet(18)
    # model_dir = './backdoorbox/experiments/ResNet18_benign/ckpt_epoch_60.pth'
    # net.load_state_dict(torch.load(model_dir))

    net = LinearRegression()
    # model_dir_badnets_pre = '/data/xuxx/experiment_uap/experiments/'
    model_dir_badnets_pre = '../backdoorbox/experiments/'

    for i in lr_mnist:
        model_dir_badnets = model_dir_badnets_pre + i + '/ckpt_epoch_40.pth'
        net.load_state_dict(torch.load(model_dir_badnets))
        v, fooling_rates, accuracies, total_iterations=adversarial_perturbation_mnist.generate(trainloader, testloader, net)
        torch.save(v.cpu(), model_dir_badnets_pre + i + '/uap_perturbation.pth')
    


    # v, fooling_rates, accuracies, total_iterations=adversarial_perturbation_mnist.generate(trainloader, testloader, trainer.net.cpu())


if __name__ == "__main__":
    main()