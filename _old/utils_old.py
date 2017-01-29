from mnist import load_mnist
from cifar10 import load_cifar
import numpy as np
import scipy.misc
import datetime
import dateutil.tz
import mnist
import cifar10
import os

# TODO unify these with train.py
BATCH_SIZE = 16
MNIST_PARAMS = 28, 28, 1
CIFAR_PARAMS = cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3


def get_timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y_%m_%d_%H_%M_%S')
def save_images(images, height, width, n_row, n_col,
      cmin=0.0, cmax=1.0, directory="./", prefix="sample"):
    print(images.shape)
    images = images.reshape((n_row, n_col, height, width))
    images = images.transpose(1, 2, 0, 3)
    images = images.reshape((height * n_row, width * n_col))

    filename = '%s_%s.jpg' % (prefix, get_timestamp())
    scipy.misc.toimage(images, cmin=cmin, cmax=cmax) \
      .save(os.path.join(directory, filename))
def binarize(images):
    rand = np.random.uniform(size=images.shape)
    return (rand < images).astype('float32')

def load_images(dataset_name, normalize=True):
    if dataset_name == 'mnist':
        dataset = mnist.load_mnist()
        image_height, image_width, num_channels = MNIST_PARAMS
        next_train_batch = lambda x: dataset.train.next_batch(x)[0]
        next_test_batch = lambda x: dataset.test.next_batch(x)[0]
    elif dataset_name == 'cifar':
        dataset = cifar10.load_cifar()
        image_height, image_width, num_channels = CIFAR_PARAMS
        # TODO
        # next_train_batch
        # next_test_batch
    else:
        raise('{0} is not a supported dataset'.format(dataset_name))
    return dataset, image_height, image_width, num_channels, next_train_batch, next_test_batch

def get_shape(tensor):
    return tensor.get_shape().as_list()
# TEST
if __name__=="__main__":
    load_cifar(False, 16)
