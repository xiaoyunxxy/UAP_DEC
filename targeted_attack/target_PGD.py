import os
import numpy as np
import argparse
import torch
import torchattacks
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from tqdm import tqdm
import sys
sys.path.append("..") 
sys.path.append("../backdoorbox/core/models") 

from backdoorbox import core
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
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--data_root', default='../../data/', type=str, help='path to dataset')
parser.add_argument('--attack', default='pgd', type=str, help='attack type')
parser.add_argument('--eps', default=0.03, type=float, help='max norm')
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

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

net = core.models.vgg.vgg11().cuda()
model_dir_badnets = '/data/xuxx/experiment_uap/experiments/' + vgg_cifar10[1] + '/ckpt_epoch_60.pth'
net.load_state_dict(torch.load(model_dir_badnets))


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


def test(net, testloader, attack):
    correct = 0
    total = 0
    net.eval()
    perturbation = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    print('\n\n[Natural/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()
        if batch_idx == 0:
            adv_inputs = attack(inputs, targets)
            perturbation = adv_inputs - inputs
        # Evaluation
        u_inputs = perturbation + inputs
        outputs = net(u_inputs).detach()

        # for i in range(adv_inputs.shape[0]):
        #     perturbation += (adv_inputs[i] - inputs[i]).abs()
            # print('perturbation: ', perturbation)

        if batch_idx >= 50:
            target_map_function = lambda a, b : torch.ones(b.shape[0], dtype=torch.int64).cuda()*9
            attack.set_mode_targeted_by_function(target_map_function)

        # Test
        predicted = torch.max(outputs, dim=1)[1]
        print(predicted)
        total += targets.numel()
        correct += (predicted == targets).sum().item() 
        # if True in (predicted == targets):
        #     print('-----')
        #     print([predicted[i].item() for i,x in enumerate(predicted == targets) if x==False])

        
    print('[Natural/Test] Acc: {:.3f}'.format(100.*correct / total))
    print('perturbation.mean: ', perturbation.mean())


def subclass_data(sub_cla=0, ori_dataset=torchvision.datasets.MNIST, num_classes=10, train=True):
    exclude_list = list(range(num_classes))
    exclude_list.remove(sub_cla)

    transform_train = Compose([
        ToTensor()
    ])

    subloader = create_sublass(ori_dataset)
    subtest = subloader(args.data_root, train=train, transform=transform_train, download=True, exclude_list=exclude_list)
    class_data = torch.utils.data.DataLoader(subtest, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return class_data

def target_attack(target_cla=0):
    attack = torchattacks.PGD(model=net, eps=args.eps,
                                    alpha=2/255, steps=args.steps, random_start=True)
    target_map_function = lambda a, b : torch.ones(b.shape[0], dtype=torch.int64).cuda()*target_cla
    attack.set_mode_targeted_by_function(target_map_function)
    return attack


if __name__ == '__main__':
    trainset, testset = load_cifar10()
    args.batch_size = 100
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size, shuffle=False, pin_memory=True)
    attack = target_attack(target_cla=1)
    test(net, testloader, attack)

    # count = 0
    # for i in range(10):
    #     for j in range(10):
    #         subl = subclass_data(sub_cla=j, train=False, ori_dataset=torchvision.datasets.CIFAR10)
    #         print('--------------')
    #         print('sub_class: ', j, '  target_cla:', i)
    #         attack = target_attack(target_cla=i)
    #         test(net, subl, attack)












