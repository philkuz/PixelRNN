from convnet import Network
import tensorflow as tf
import cifar10
import mnist
def train(dataset_name,
          max_epochs=10000,
          ):
    # Load dataset
    if dataset_name == 'mnist':
        dataset =  mnist.load_mnist()
        img_height, img_width, n_channels = 28, 28, 1
        batch_size = 16

        next_train_batch = lambda x: mnist.train.next_batch(x)[0]
        next_test_batch = lambda x: mnist.test.next_batch(x)[0]

    elif dataset_name == 'cifar':
        dataset = cifar10.load_cifar()
        img_height = img_width = cifar10.IMAGE_SIZE
        n_channels = 3
        batch_size = 16

    else:
        raise('{0} is not a supported dataset'.format(dataset_name))

    # setup train, test
    train = dataset.train
    test = dataset.test

    num_train_batches = train.num_examples / batch_size
    num_test_batches = test.num_examples / batch_size

    with tf.Session() as sess:
        network = Network(sess, img_height, img_width, n_channels)

        for epoch in xrange(max_epochs):
            training_costs = []
            for i in xrange(num_train_batches):
                next_train_batch



    
