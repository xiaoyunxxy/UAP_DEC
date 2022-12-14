import torch
import torch.nn
import os
from hsic import hsic_normalized_cca
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


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
'ResNet-18_CIFAR-10_BadNets_pos5_size8'
# 'ResNet-18_CIFAR-10_TUAP',
# 'ResNet-18_CIFAR-10_WaNet'
]

# trigger_index = 15
# trigger_size = 5
# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 255
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size] = 1.0

def area_mean(inputs, trigger_index, trigger_size):
	trigger_area = inputs[trigger_index:trigger_index+trigger_size, trigger_index:trigger_index+trigger_size]

	all_pixels = inputs.shape[0] * inputs.shape[1]
	trigger_pixels = trigger_size * trigger_size

	all_sum = inputs.sum()
	trigger_sum = trigger_area.sum()

	trigger_mean = trigger_sum / trigger_pixels
	non_trigger_mean = (all_sum-trigger_sum) / (all_pixels-trigger_pixels)

	return trigger_mean, non_trigger_mean


def distance_networks(files_path=basic_mnist):
	dis_list = torch.zeros([len(files_path), len(files_path)])
	for i in range(len(files_path)):
		for j in range(len(files_path)):
			file1 = '/data/xuxx/experiment_uap/experiments/' + files_path[i] + '/uap_perturbation.pth'
			file2 = '/data/xuxx/experiment_uap/experiments/' + files_path[j] + '/uap_perturbation.pth'

			if os.path.exists(file1) and os.path.exists(file2):
				uap1 = torch.load(file1).view(1, -1)
				uap2 = torch.load(file2).view(1, -1)
				dis_list[i][j] = cos(uap1, uap2)
				print('& ', format(dis_list[i][j].item(), '.2f'), end='')
			else:
				print('& ', format(dis_list[i][j].item(), '.2f'), end='')
		print('')

	return dis_list

def means_uap(files_path=basic_mnist):
	for i in range(len(files_path)):
		file1 = '/data/xuxx/experiment_uap/experiments/' + files_path[i] + '/uap_perturbation.pth'
		if os.path.exists(file1):
			uap1 = torch.load(file1).abs()
			print('----')
			print('overall mean: ', format(uap1.mean(), '.9f'))

			file_split = files_path[i].split('_')
			if file_split[2] == 'BadNets':
				
				tri_index = int(file_split[3][3:])
				tri_size = int(file_split[4][4:])
				print('tri_index: ', tri_index)
				print('tri_size: ', tri_size)

				for sigle_channel in uap1:
					tri_mean, non_mean = area_mean(sigle_channel, tri_index, tri_size)
					print(' & ', tri_mean.item(), end='')
					print(' & ', non_mean.item())

if __name__ == '__main__':
	dis_list = distance_networks(files_path=vgg_cifar10)
	# print(dis_list)
	means_uap(files_path=resnet_cifar)

