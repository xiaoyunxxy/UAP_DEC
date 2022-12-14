import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch



path = 'benign_perturbation.pth'
im = torch.load(path)
im = im.permute(1,2,0)
imgplot = plt.imshow(im)
plt.savefig('./uap_perturbation.jpg')