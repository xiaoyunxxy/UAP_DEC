import numpy as np
import deepfool
from PIL import Image
import trainer
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def project_perturbation(data_point, p, perturbation):

    if p == 2:
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation


def generate(trainset, testset, net, delta=0.5, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.2, max_iter_df=20):
    '''
    :param trainset: Pytorch Dataloader with train data
    :param testset: Pytorch Dataloader with test data
    :param net: Network to be fooled by the adversarial examples
    :param delta: 1-delta represents the fooling_rate, and the objective
    :param max_iter_uni: Maximum number of iterations of the main algorithm
    :param p: Only p==2 or p==infinity are supported
    :param num_class: Number of classes on the dataset
    :param overshoot: Parameter to the Deep_fool algorithm
    :param max_iter_df: Maximum iterations of the deep fool algorithm
    :return: perturbation found (not always the same on every run of the algorithm)
    '''

    net.eval()
    device = 'cuda'
    net = net.to(device)

    # Importing images, batch_size == 100, take 100 training data
    img_trn = iter(trainset)
    img_trn = img_trn.next()[0]

    # Setting the number of images to 300  (A much lower number than the total number of instances on the training set)
    # To verify the generalization power of the approach
    num_img_trn = 500
    index_order = np.arange(num_img_trn)

    # Initializing the perturbation to 0s
    v=torch.zeros([3, 32, 32]).to(device)

    #Initializing fooling rate and iteration count
    fooling_rate = 0.0
    iter_count = 0

  
    fooling_rates=[0]
    accuracies = []
    # accuracies.append(accuracy)
    total_iterations = [0]
    # Begin of the main loop on Universal Adversarial Perturbations algorithm
    while fooling_rate < 1-delta and iter_count < max_iter_uni:
        np.random.shuffle(index_order)
        print("Iteration  ", iter_count)

        for index in index_order:
            # Generating the original image from data
            cur_img = img_trn[index:index+1].to(device)

            # Feeding the original image to the network and storing the label returned
            r2 = (net(cur_img).max(1)[1])
            torch.cuda.empty_cache()

            # Generating a perturbed image from the current perturbation v and the original image

            per_img = cur_img+v[None, :, :, :]


            # Feeding the perturbed image to the network and storing the label returned
            r1 = (net(per_img).max(1)[1])
            torch.cuda.empty_cache()

            # If the label of both images is the same, the perturbation v needs to be updated
            if r1 == r2:
                print(">> k =", np.where(index==index_order)[0][0], ', pass #', iter_count, end='      ')

                # Finding a new minimal perturbation with deepfool to fool the network on this image
                dr, iter_k, label, k_i, pert_image = deepfool.deepfool(per_img[0], net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
                if iter_k < max_iter_df-1:
                    tmp = torch.Tensor(dr).to(device)

                    v += tmp[0]

                    v = project_perturbation(xi, p, v.cpu()).to(device)

        iter_count = iter_count + 1


        with torch.no_grad():
            print('\nComputing fooling_rate...')

            # Compute fooling_rate
            labels_original_images = torch.tensor(np.zeros(0, dtype=np.int64)).to(device)
            labels_pertubed_images = torch.tensor(np.zeros(0, dtype=np.int64)).to(device)

            i = 0
            # Finding labels for original images
            for batch_index, (inputs, _) in enumerate(testset):
                i += inputs.shape[0]
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                labels_original_images = torch.cat((labels_original_images, predicted.to(device)))
            torch.cuda.empty_cache()
            correct = 0
            # Finding labels for perturbed images
            for batch_index, (inputs, labels) in enumerate(testset):
                inputs, labels = inputs.to(device), labels.to(device)
                inputs += v
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                labels_pertubed_images = torch.cat((labels_pertubed_images, predicted.to(device)))
                correct += (predicted == labels).sum().item()
            torch.cuda.empty_cache()


            # Calculating the fooling rate by dividing the number of fooled images by the total number of images
            fooling_rate = float(torch.sum(labels_original_images != labels_pertubed_images))/float(i)

            print()
            print("FOOLING RATE: ", fooling_rate)
            fooling_rates.append(fooling_rate)
            accuracies.append(correct / i)
            total_iterations.append(iter_count)
        # torch.save(v.cpu(), './uap_perturbation.pth')
        im = v.cpu().permute(1,2,0) * 10
        imgplot = plt.imshow(im)
        plt.savefig('./uap_perturbation.jpg')

    return v,fooling_rates,accuracies,total_iterations
