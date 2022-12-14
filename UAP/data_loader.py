import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, transforms
import os


device = torch.device('cpu')
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Transform to tensor
        transforms.Normalize((0.5,), (0.5,))  # Min-max scaling to [-1, 1]
    ])

    data_dir = os.path.join("./", 'fashion_mnist')
    print('Data stored in %s' % data_dir)
    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)
    return trainloader,testloader



def dataset_loader(args):

    args.mean=0.5
    args.std=0.25

    # Setting Dataset Required Parameters
    if args.dataset   == "svhn":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "cifar10":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "tiny":
        args.n_classes = 200
        args.img_size  = 64
        args.channel   = 3
    elif args.dataset == "cifar100":
        args.n_classes = 100
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "gtsrb":
        args.n_classes = 43
        args.img_size  = 32
        args.channel   = 3


    transform_train = transforms.Compose(
        [transforms.RandomCrop(args.img_size, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        Resize((args.img_size, args.img_size))]
    )

    # Full Trainloader/Testloader
    trainloader = torch.utils.data.DataLoader(dataset(args, True,  transform_train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(dataset(args, False, transform_test),  batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return trainloader, testloader


def dataset(args, train, transform):

        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        if args.dataset == "mnist":
            return torchvision.datasets.MNIST(root=args.data_root, transform=transform, download=True, train=train)
        if args.dataset == "cifar100":
            return torchvision.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        if args.dataset == "gtsrb":
            return torchvision.datasets.GTSRB(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.dataset == "svhn":
            return torchvision.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.dataset == "tiny":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/train' if train \
                                    else args.data_root + '/tiny-imagenet-200/val_classified', transform=transform)