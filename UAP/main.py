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
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]='0'

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
'VGG-11_CIFAR-10_BadNets_pos15_size15_2022-11-28_15:52:36', #acc 0.1, posion acc 1.0
'VGG-11_CIFAR-10_BadNets_pos15_size5_2022-11-28_15:52:19',
'VGG-11_CIFAR-10_BadNets_pos5_size10_2022-11-28_15:51:03', #acc 0.1, posion acc 1.0
'VGG-11_CIFAR-10_BadNets_pos5_size15_2022-11-28_15:48:14', #acc 0.1, posion acc 1.0
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
# 'resnet18badnets',
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
    # net = resnet.ResNet(18, num_classes=args.n_classes)
    net = vgg.vgg11()
    model_benign = '../backdoorbox/experiments/ResNet18_benign/ckpt_epoch_60.pth'
    model_dir_badnets = '/data/xuxx/experiment_uap/experiments/VGG-11_gtsrb_benign_2022-11-29_22:39:49/ckpt_epoch_60.pth'

    for i in vgg_cifar10[9:]:
        model_dir_badnets = '/data/xuxx/experiment_uap/experiments/' + i + '/ckpt_epoch_60.pth'
        if not os.path.exists(model_dir_badnets):
            continue
        print('model_dir_badnets:  ', model_dir_badnets)
        net.load_state_dict(torch.load(model_dir_badnets))
        for target_label in range(10):
            v, fooling_rates, accuracies, total_iterations=adversarial_perturbation.generate(trainloader, testloader, net, target_label=target_label)
            torch.save(v.cpu(), '/data/xuxx/experiment_uap/experiments/' + i + '/uap_perturbation_target' + str(target_label) + '.pth')

        break


# def test():
#     net = resnet.ResNet(18, num_classes=args.n_classes).cuda()
#     model_dir_badnets = '/data/xuxx/experiment_uap/experiments/' + resnet_cifar[0] + '/ckpt_epoch_60.pth'
#     net.load_state_dict(torch.load(model_dir_badnets))

#     correct = 0
#     total = 0
#     net.eval()
#     p = '/data/xuxx/experiment_uap/experiments/' + resnet_cifar[0] + '/uap_perturbation_target2.pth'
#     perturbation = torch.load(p).cuda()
#     print('perturbation: ', perturbation.abs().mean())
#     return
#     # print('\n\n[Natural/Test] Under Testing ... Wait PLZ')
#     for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
#         # dataloader parsing and generate adversarial examples
#         inputs, targets = inputs.cuda(), targets.cuda()
#         # Evaluation
#         outputs = net(inputs + perturbation).detach()

#         # Test
#         predicted = torch.max(outputs, dim=1)[1]
#         print('predicted: ', predicted)
#         total += targets.numel()
#         correct += (predicted == targets).sum().item() 
     
#     print('[Natural/Test] Acc: {:.3f}'.format(100.*correct / total))


def save_figs():
    for i in range(len(basic_mnist)):
        for j in range(10):
            p = '/data/xuxx/experiment_uap/experiments/' + basic_mnist[i] + '/uap_perturbation_target'+str(j)+'.pth'
            if not os.path.exists(p):
                break
            pe = torch.load(p).abs()
            # save fig
            im = pe.cpu().permute(1,2,0)
            imgplot = plt.imshow(im, cmap='gray')

            b = './' + basic_mnist[i]
            if not os.path.exists(b):
                os.makedirs(b)
            plt.savefig(b + '/uap_perturbation'+str(j)+'.jpg')

def test(perturbation, model_path):
    # net = resnet.ResNet(18, num_classes=args.n_classes).cuda()
    net = vgg.vgg11().cuda()
    model_dir_badnets = '/data/xuxx/experiment_uap/experiments/' + model_path + '/ckpt_epoch_60.pth'
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
    data_path = vgg_cifar10
    for j in range(0, len(data_path)):
        print('------- ', data_path[j])
        for i in range(10):
            p = '/data/xuxx/experiment_uap/experiments/' + data_path[j] + '/uap_perturbation_target'+str(i)+'.pth'
            if not os.path.exists(p):
                continue
            perturbation = torch.load(p).cuda()
            print('perturbation: ', format(perturbation.abs().mean().item(), '.5f'), end='')

            file_split = data_path[j].split('_')
            if file_split[2] == 'BadNets':
                
                tri_index = int(file_split[3][3:])
                tri_size = int(file_split[4][4:])
                # print('tri_index: ', tri_index)
                # print('tri_size: ', tri_size)

                for sigle_channel in perturbation:
                    tri_mean, non_mean = area_mean(sigle_channel, tri_index, tri_size)
                    print(' & ', format(tri_mean.abs().item(), '.5f'), end='')
                    print(' & ', format(non_mean.abs().item(), '.5f'), end='')

            test(perturbation, data_path[0])

if __name__ == "__main__":
    # for i in range(10):
    #     p = '/data/xuxx/experiment_uap/experiments/' + vgg_cifar10[-2] + '/uap_perturbation_target'+str(i)+'.pth'
    #     perturbation = torch.load(p).abs().cuda()
    #     print('perturbation: ', perturbation.mean())

    # save_figs()
    means_acc()











