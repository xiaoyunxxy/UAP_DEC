import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

def get_activation(atype):

    if atype=='relu':
        nonlinear = nn.ReLU()
    elif atype=='tanh':
        nonlinear = nn.Tanh()
    elif atype=='sigmoid':
        nonlinear = nn.Sigmoid() 
    elif atype=='elu':
        nonlinear = nn.ELU()

    return nonlinear

def get_activation_functional(atype):

    if atype=='relu':
        nonlinear = torch.relu
    elif atype=='tanh':
        nonlinear = torch.tanh
    elif atype=='sigmoid':
        nonlinear = torch.sigmoid
    elif atype=='elu':
        nonlinear = torch.elu

    return nonlinear

def get_primative_block(model_type, hid_in, hid_out, atype):
    if model_type=='simple-dense':
        block = makeblock_dense(hid_in, hid_out, atype)
    elif model_type=='simple-conv':
        block = makeblock_conv(hid_in, hid_out, atype)
    elif model_type=='resnet-dense':
        block = BasicResidualBlockDense(hid_in, hid_out, atype)
    elif model_type=='resnet-conv':
        block = BasicResidualBlockConv(hid_in, hid_out, atype)
    return block

def makeblock_dense(in_dim, out_dim, atype):
    
    layer = nn.Linear(in_dim, out_dim)
    bn = nn.BatchNorm1d(out_dim, affine=False)
    if atype=='linear':
        out = nn.Sequential(*[layer, bn])
    else:
        nonlinear = get_activation(atype)
        out = nn.Sequential(*[layer, bn, nonlinear])
    return out

def makeblock_conv(in_chs, out_chs, atype, stride=1):

    layer = nn.Conv2d(in_channels=in_chs, 
        out_channels=out_chs, kernel_size=5, stride=stride)
    bn = nn.BatchNorm2d(out_chs, affine=False)
    nonlinear = get_activation(atype)

    return nn.Sequential(*[layer, bn, nonlinear])

class BasicBlockConv(nn.Module):
    """docstring for BasicBlockConv"""
    def __init__(self, in_chs, out_chs, atype):
        super(BasicBlockConv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_chs, 
            out_channels=out_chs, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(out_chs, affine=False)
        self.nfunc = get_activation_functional(atype)

    def forward(self, x):        
        out = self.nfunc(self.bn(self.conv(x)))
        return out
       
class BasicBlockDense(nn.Module):
    """docstring for BasicBlockDense"""
    def __init__(self, in_dim, out_dim, atype):
        super(BasicBlockDense, self).__init__()

        self.dense = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.nfunc = get_activation_functional(atype)

    def forward(self, x):        
        out = self.nfunc(self.bn(self.dense(x)))
        return out

class BasicResidualBlockDense(nn.Module):

    def __init__(self, in_dim, out_dim, atype):
        super(BasicResidualBlockDense, self).__init__()

        self.dense1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim, affine=False)
        self.shortcut = nn.Sequential()
        self.nfunc = get_activation_functional(atype)
        self.bn3 = nn.BatchNorm1d(out_dime, affine=False)

    def forward(self, x):
        out = self.nfunc(self.bn1(self.dense1(x)))
        out += self.shortcut(x)
        out = self.bn3(out)
        out = self.nfunc(out)
        return out

class BasicResidualBlockConv(nn.Module):

    def __init__(self, in_chs, out_chs, atype):
        super(BasicResidualBlockConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(out_chs, affine=False)
        self.shortcut = nn.Sequential()
        self.nfunc = get_activation_functional(atype)
        self.bn3 = nn.BatchNorm2d(out_chs, affine=False)

    def forward(self, x):

        out = self.nfunc(self.bn1(self.conv1(x)))
        out += self.shortcut(x)
        out = self.bn3(out)
        out = self.nfunc(out)

        return out



def get_current_timestamp():
    return strftime("%y%m%d_%H%M%S")

def get_in_channels(data_code):
    in_ch = -1
    if data_code == 'mnist':
        in_ch = 1
    elif data_code == 'cifar10':
        in_ch = 3
    elif data_code == 'fmnist':
        in_ch = 1
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_ch

def get_in_dimensions(data_code):
    in_dim = -1    
    if data_code == 'mnist':
        in_dim = 784
    elif data_code == 'cifar10':
        in_dim = 1024
    elif data_code == 'fmnist':
        in_dim = 784
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_dim

def get_accuracy_epoch(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    acc = []
    loss = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    model = model.to('cuda')
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        output, hiddens = model(data)
        loss.append(cross_entropy_loss(output, target).cpu().detach().numpy())
        acc.append(get_accuracy(output, target)[0].cpu().detach().numpy())
    return np.mean(acc), np.mean(loss)


def get_accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_accuracy_hsic(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    for batch_idx, (data, target) in enumerate(dataloader):
        output, hiddens = model(data.to(next(model.parameters()).device))
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy().reshape(-1,1)
        output_list.append(output)
        target_list.append(target)
    output_arr = np.vstack(output_list)
    target_arr = np.vstack(target_list)
    avg_acc = 0
    reorder_list = []
    for i in range(10):
        indices = np.where(target_arr==i)[0]
        select_item = output_arr[indices]
        out = np.array([np.argmax(vec) for vec in select_item])
        y = np.mean(select_item, axis=0)
        while np.argmax(y) in reorder_list:
            y[np.argmax(y)] = 0
        reorder_list.append(np.argmax(y))
        num_correct = np.where(out==np.argmax(y))[0]
        accuracy = float(num_correct.shape[0])/float(out.shape[0])
        avg_acc += accuracy
    avg_acc /= 10.

    return avg_acc*100., reorder_list

def get_layer_parameters(model, idx_range):

    param_out = []
    param_out_name = []
    for it, (name, param) in enumerate(model.named_parameters()):
        if it in idx_range:
            param_out.append(param)
            param_out_name.append(name)

    return param_out, param_out_name


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes)[y])
