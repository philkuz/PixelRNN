from convnet import Network
from utils import binarize
import tensorflow as tf
import numpy as np
import cifar10
import mnist

BATCH_SIZE = 16
MNIST_PARAMS = 28, 28, 1
CIFAR_PARAMS = cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3


def load_images(dataset_name, normalize=True):
    if dataset_name == 'mnist':
        dataset = mnist.load_mnist()
        image_height, image_width, num_channels = MNIST_PARAMS
        next_train_batch = lambda x: mnist.train.next_batch(x)[0]
        next_test_batch = lambda x: mnist.test.next_batch(x)[0]
    elif dataset_name == 'cifar':
        dataset = cifar10.load_cifar()
        image_height, image_width, num_channels = CIFAR_PARAMS
        # TODO
        # next_train_batch
        # next_test_batch
    else:
        raise('{0} is not a supported dataset'.format(dataset_name))
    return dataset, image_height, image_width, num_channels, next_train_batch, next_test_batch

def train(dataset_name,
          max_epochs=10000,
          test_period=250
          ):
    # Load dataset
    dataset, image_height, image_width, num_channels, next_train_batch, next_test_batch = load_images(dataset_name)

    # setup train, test
    train = dataset.train
    test = dataset.test

    num_train_batches = train.num_examples / BATCH_SIZE
    num_test_batches = test.num_examples / BATCH_SIZE

    with tf.Session() as sess:
        network = Network(sess, image_height, image_width, num_channels)

        sampled_images = []
        for epoch in xrange(max_epochs):
            training_costs = []
            for i in xrange(num_train_batches):
                images = binarize(next_train_batch(BATCH_SIZE)).reshape([BATCH_SIZE, image_height, image_width, num_channels])
                cost = network.test(images, with_update=True)
                training_costs.append(cost)
            # test
            if epoch % test_period == 0:
                testing_costs = []
                for i in xrange(num_test_batches):
                    images = binarize(next_test_batch(BATCH_SIZE)).reshape(
                        [BATCH_SIZE, image_height, image_width, num_channels])

                    cost = network.test(images, with_update=False)
                    testing_costs.append(cost)
                avg_test_cost = np.average(np.array(testing_costs))
                print('Test cost at epoch %d: %04f' % (epoch, avg_test_cost))

            # save the images some how
            #samples = network.generate_image()
            #sampled_images.append(samples)







    
