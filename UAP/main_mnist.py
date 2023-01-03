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
from tqdm import tqdm
from hsic import hsic_normalized_cca
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]='0'

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
'BaselineMNISTNetwork_MNIST_BadNets_pos5_size5',
'BaselineMNISTNetwork_MNIST_BadNets_pos0_size5_t5',
'BaselineMNISTNetwork_MNIST_BadNets_pos10_size10_t5'
]

lr_mnist = [
'LinearRegression_MNIST_Benign',
'LinearRegression_MNIST_BadNets_pos5_size5'
]

def main():

    # net = resnet.ResNet(18)
    # model_dir = './backdoorbox/experiments/ResNet18_benign/ckpt_epoch_60.pth'
    # net.load_state_dict(torch.load(model_dir))

    # net = LinearRegression()
    net = baseline_MNIST_network.BaselineMNISTNetwork()
    # model_dir_badnets_pre = '/data/xuxx/experiment_uap/experiments/'
    model_dir_badnets_pre = '/data/xuxx/experiment_uap/experiments/'

    for i in basic_mnist[-1:]:
        model_dir_badnets = model_dir_badnets_pre + i + '/ckpt_epoch_40.pth'
        if not os.path.exists(model_dir_badnets):
            continue
        net.load_state_dict(torch.load(model_dir_badnets))
        for target_label in range(10):
            v, fooling_rates, accuracies, total_iterations=adversarial_perturbation_mnist.generate(trainloader, testloader, net, target_label=target_label)
            torch.save(v.cpu(), model_dir_badnets_pre + i + '/uap_perturbation_target' + str(target_label) + '.pth')
    


    # v, fooling_rates, accuracies, total_iterations=adversarial_perturbation_mnist.generate(trainloader, testloader, trainer.net.cpu())
def test(perturbation, model_path):
    net = baseline_MNIST_network.BaselineMNISTNetwork().cuda()
    model_dir_badnets = '/data/xuxx/experiment_uap/experiments/' + model_path + '/ckpt_epoch_40.pth'
    net.load_state_dict(torch.load(model_dir_badnets))

    correct = 0
    total = 0
    net.eval()
    # p = '/data/xuxx/experiment_uap/experiments/' + basic_mnist[0] + '/uap_perturbation_target0.pth'
    # perturbation = torch.load(p).cuda()

    # print('\n\n[Natural/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()
        # Evaluation
        outputs = net(inputs + perturbation).detach()

        # Test
        predicted = torch.max(outputs, dim=1)[1]
        # print('predicted: ', predicted)
        total += targets.numel()
        correct += (predicted == targets).sum().item() 
     
    # print('[Natural/Test] Acc: {:.3f}'.format(100.*correct / total))
    print(' & ', '{:.3f}'.format(100.*correct / total))
    print('-----')

def area_mean(inputs, trigger_index, trigger_size):
    trigger_area = inputs[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size]

    all_pixels = inputs.shape[0] * inputs.shape[1]
    trigger_pixels = trigger_size * trigger_size

    all_sum = inputs.sum()
    trigger_sum = trigger_area.sum()

    trigger_mean = trigger_sum / trigger_pixels
    non_trigger_mean = (all_sum-trigger_sum) / (all_pixels-trigger_pixels)

    return trigger_mean, non_trigger_mean


def means_acc():
    for j in range(len(basic_mnist)):
        print('------- ', basic_mnist[j])
        for i in range(10):
            p = '/data/xuxx/experiment_uap/experiments/' + basic_mnist[j] + '/uap_perturbation_target'+str(i)+'.pth'
            perturbation = torch.load(p).cuda()
            print('perturbation: ', format(perturbation.abs().mean().item(), '.5f'), end='')

            file_split = basic_mnist[j].split('_')
            if file_split[2] == 'BadNets':
                
                tri_index = int(file_split[3][3:])
                tri_size = int(file_split[4][4:])
                # print('tri_index: ', tri_index)
                # print('tri_size: ', tri_size)

                for sigle_channel in perturbation:
                    tri_mean, non_mean = area_mean(sigle_channel, tri_index, tri_size)
                    print(' & ', format(tri_mean.abs().item(), '.5f'), end='')
                    print(' & ', format(non_mean.abs().item(), '.5f'), end='')

            test(perturbation, basic_mnist[j])


def hsic_check():
    for i in range(len(basic_mnist)):
        print('------------- ', basic_mnist[i])
        for j in range(10):
            pj = '/data/xuxx/experiment_uap/experiments/' + basic_mnist[i] + '/uap_perturbation_target'+str(j)+'.pth'
            pj = torch.load(pj)
            pj = pj.view(784, 1)
            minmi = 1000
            minmi_index = ''
            for k in range(10):
                pk = '/data/xuxx/experiment_uap/experiments/' + basic_mnist[i] + '/uap_perturbation_target'+str(k)+'.pth'
                pk = torch.load(pk)
                pk = pk.view(784, 1)

                mi = hsic_normalized_cca(pj, pk, sigma=5, k_type_y='gaussian')
                if mi < minmi:
                    minmi = mi
                    minmi_index = 'target_' + str(j) + '& target_'+ str(k)
                print('target_', str(j), '& target_', str(k), mi.item())
            print('--- ', minmi_index)

def hsic_benign_check():
    for i in range(1, len(basic_mnist)):
        print('------------- ', basic_mnist[i])
        for j in range(10):
            p = '/data/xuxx/experiment_uap/experiments/' + basic_mnist[0] + '/uap_perturbation_target'+str(j)+'.pth'
            p = torch.load(p)
            p = p.view(784)
            pj = '/data/xuxx/experiment_uap/experiments/' + basic_mnist[i] + '/uap_perturbation_target'+str(j)+'.pth'
            pj = torch.load(pj)
            pj = pj.view(784)

            # mi = hsic_normalized_cca(p, pj, sigma=5, k_type_y='gaussian')
            mi = torch.cosine_similarity(p, pj, dim=0)
            print('target_', str(j), ':   ', mi)




if __name__ == "__main__":
    means_acc()







