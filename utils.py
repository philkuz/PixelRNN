from mnist import load_mnist
from cifar10 import load_cifar
import numpy as np

def binarize(images):
    rand = np.random.uniform(size=images.shape)
    return (rand < images).astype('float32')
    #return (images < 0.5).astype('float32') # TODO research why we don't use this standard binarization


def get_shape(tensor):
    return tensor.get_shape().as_list()
# TEST
if __name__=="__main__":
    load_cifar(False, 16)
