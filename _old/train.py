from convnet import Network
from utils import binarize
import tensorflow as tf
import numpy as np
import cifar10
import mnist
from statistic import Statistic
from utils import load_images, save_images
import os

# TODO unify these with utils.py
BATCH_SIZE = 16
MNIST_PARAMS = 28, 28, 1
CIFAR_PARAMS = cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3





def train(dataset_name,
          max_epochs=10000,
          test_period=1
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
        # tf.initialize_all_variables().run()

        # TODO make more general
        stat = Statistic(sess, 'mnist', 'train', tf.trainable_variables(), test_period)
        stat.load_model()
        SAMPLE_DIR = os.path.join('samples', 'mnist', 'train')
        initial_step = stat.get_t() if stat else 0

        sampled_images = []
        for epoch in xrange(max_epochs):
            print('Current epoch: %i' % epoch)
            training_costs = []
            for i in xrange(num_train_batches):
                images = binarize(next_train_batch(BATCH_SIZE)).reshape([BATCH_SIZE, image_height, image_width, num_channels])
                cost = network.test(images, with_update=True)
                training_costs.append(cost)
            # test
            if epoch % test_period   == 0:
                print('Running tests...')
                testing_costs = []
                for i in xrange(num_test_batches):
                    images = binarize(next_test_batch(BATCH_SIZE)).reshape(
                        [BATCH_SIZE, image_height, image_width, num_channels])

                    cost = network.test(images, with_update=False)
                    testing_costs.append(cost)
                avg_train_cost = np.average(training_costs)
                avg_test_cost = np.average(testing_costs)
                print('Test cost at epoch %d: %04f' % (epoch, avg_test_cost))
                stat.on_step(avg_train_cost, avg_test_cost)

                samples = network.generate_images(100)
                save_images(samples, image_height, image_width, 10, 10, directory=SAMPLE_DIR)


if __name__ == '__main__':
    train('mnist', test_period=5)
    
