from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import os
root = '/database/cifar10/'

from PIL import Image

dataset_train = CIFAR10(root)
for k, (img, label) in enumerate(dataset_train):
    print('processsing' + str(k))
    if not os.path.exists(root + 'CIFAR10_image/' + str(label)+ '/'):
        os.mkdir(root + 'CIFAR10_image/' + str(label)+ '/')
    img.save(root + 'CIFAR10_image/' + str(label) + '/' + str(k) + '.png')

