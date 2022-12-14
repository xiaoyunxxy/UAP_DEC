#!/usr/bin/env python

# numpy package
import numpy as np

# torch package
import torch
import torchvision
from torch.nn.functional import cross_entropy
import torch.nn.functional as F

# basic package
import os
import argparse
from tqdm import tqdm
from datetime import datetime

# custom package
from loader.loader import dataset_loader, network_loader, attack_loader
from loader.argument_print import argument_print

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
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
parser.add_argument('--network', default='lr', type=str, help='network name')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--data_root', default='../data', type=str, help='path to dataset')
parser.add_argument('--epoch', default=60, type=int, help='epoch number')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
parser.add_argument('--pretrained', default='false', type=str2bool, help='pretrained boolean')
parser.add_argument('--batchnorm', default='true', type=str2bool, help='batchnorm boolean')
parser.add_argument('--save_dir', default='/data/xuxx/experiment_MI/', type=str, help='save directory')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

# loading dataset, network
args.attack = 'Plain'
trainloader, testloader = dataset_loader(args)
net = network_loader(args, mean=args.mean, std=args.std).cuda()
if len(args.gpu_id.split(','))!=1:
    net = torch.nn.DataParallel(net)
args.eps = 0

# Adam Optimizer with KL divergence, and Scheduling Learning rate
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
# criterion = torch.nn.CrossEntropyLoss()
criterion= torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
criterion_kl = torch.nn.KLDivLoss(reduction='none')

# Setting checkpoint date time
date_time = datetime.today().strftime("%m%d%H%M")

# checkpoint_name
checkpoint_name = 'Plain_'+args.network+'_'+args.dataset+'_'+date_time+'.pth'


# mi loss parameters
args.ly =0.0
args.lx = 0.0

# argument print
argument_print(args, checkpoint_name)


# attack
args.eps = 0.03
args.attack = 'pgd'
args.steps = 10
attack = attack_loader(args, net)


def train():

    for epoch in range(args.epoch):

        # train environment
        net.train()

        print('\n\n[Plain/Epoch] : {}'.format(epoch+1))

        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            # dataloader parsing and generate adversarial examples
            inputs, targets = inputs.cuda(), targets.cuda()
            targets_onehot = F.one_hot(targets).float()
            # learning network parameters
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets_onehot)
            loss.backward()
            optimizer.step()

            # validation
            pred = torch.max(net(inputs).detach(), dim=1)[1]
            correct += torch.sum(pred.eq(targets)).item()
            total += targets.numel()

            # logging two types loss and total loss
            running_loss += loss.item()

            if batch_idx % 50 == 0 and batch_idx != 0:
                print('[Plain/Train] Iter: {}, Acc: {:.3f}, Loss: {:.3f}'.format(
                    batch_idx, # Iter
                    100.*correct / total, # Acc
                    running_loss / (batch_idx+1) # CrossEntropy
                    )
                )

        # Adversarial validation
        test()
        # adv_test()
        
        # Save checkpoint file
        torch.save(net.state_dict(), os.path.join(args.save_dir,checkpoint_name))

        # argument print
        argument_print(args, checkpoint_name)
    


def test():
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
        
    print('[Plain/Test] Acc: {:.3f}'.format(100.*correct / total))


def adv_test():
    correct = 0.0
    total = 0
    # validation loop

    net.eval()
    print('\n\n[Plain/Adv_Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        inputs, targets = inputs.cuda(), targets.cuda()

        adv_input = attack(inputs, targets)
        pred = net(adv_input).detach()

        _, predicted = torch.max(pred, dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item() 


    print('[Adv/Test] Acc: {:.3f}'.format(100.*correct / total))


if __name__ == "__main__":
    train()
