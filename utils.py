from mnist import load_mnist
from cifar10 import load_cifar

def get_shape(tensor):
    return tensor.get_shape().as_list()
# TEST
if __name__=="__main__":
    load_cifar(False, 16)
