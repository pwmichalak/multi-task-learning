import matplotlib.pyplot as plt 
import numpy as np
import torchvision
from utils import CIFAR10Loader

batch_size = 128
classes = ['cat','dog']
percentages = [0,0,0,1,0,0.3,0,0,0,0]
cifar = CIFAR10Loader(batch_size=batch_size,train=True,shuffle=True,drop_last=False,classes=classes,percentages=percentages)
labels = cifar.get_labels()
train_data = cifar.get_loader()
cifar._create_TaskDataLoaders()

def imgplot(img,title=''):
    """
    Brief:
        Function that plots an image from CIFAR10
    Arguments:
        img - image to plot 
    """
    # renormalize the image to be in range(0..1); otherwise the colors are all weird 
    img = img / 2 + 0.5

    # convert the image to a numpy image; otherwise the colors are all weird
    npimg = img.numpy()

    # transpose the image to the correct shape: (xdim, ydim, colors)
    im = np.transpose(npimg, (1,2,0))

    # plot the image
    plt.imshow(im)
    plt.title(title)
    plt.show()

CIFARclasses = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img, lab = iter(train_data).next()

# convert the batch of images into an image grid that can be plotted
im_grid = torchvision.utils.make_grid(img)
imgplot(im_grid)