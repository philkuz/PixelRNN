from convnet import Network
import tensorflow as tf
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

        for epoch in xrange(max_epochs):
            training_costs = []
            for i in xrange(num_train_batches):
                next_train_batch



    
