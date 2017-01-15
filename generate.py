""" Loads up a trained model and generates images """
from statistic import Statistic
from convnet import Network
import tensorflow as tf

from utils import load_images, save_images
import os.path


def generate(dataset_name, test_period=250):
    # Load dataset
    dataset, image_height, image_width, num_channels, next_train_batch, next_test_batch = load_images(dataset_name)

    # setup train, test
    train = dataset.train
    test = dataset.test
    SAMPLE_DIR = os.path.join('samples', 'mnist', 'train')

    with tf.Session() as sess:
        network = Network(sess, image_height, image_width, num_channels)
        # tf.initialize_all_variables().run()

        stat = Statistic(sess, 'mnist', 'train', tf.trainable_variables(), test_period)
        stat.load_model()
        samples = network.generate_images(100)
        save_images(samples, image_height, image_width, 10, 10, directory=SAMPLE_DIR)


if __name__ == "__main__":
    generate('mnist')
