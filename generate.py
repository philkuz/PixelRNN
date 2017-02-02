""" Loads up a trained model and generates images """
from statistic import Statistic
from convnet import Network
import tensorflow as tf

from utils import load_images, save_images
import os.path
import argparse


def generate(dataset_name, occlusions=False):
    # Load dataset
    dataset, image_height, image_width, num_channels, next_train_batch, next_test_batch = load_images(dataset_name)

    # setup train, test
    train = dataset.train
    test = dataset.test
    SAMPLE_DIR = os.path.join('samples', dataset_name, 'generate')

    with tf.Session() as sess:
        network = Network(sess, image_height, image_width, num_channels)
        # tf.initialize_all_variables().run()

        stat = Statistic(sess, dataset_name, './', tf.trainable_variables(), 0)
        stat.load_model()
        num_images = 100
        if occlusions:
            orig_images = next_test_batch(num_images).reshape(
                        [num_images, image_height, image_width, num_channels])
            orig_images[:,image_height/2:,:,:] = 0
            samples = network.generate_images(num_images, starting_pos=[0, image_height / 2], starting_image=orig_images)
            # original_occlusions
            occlusion_dir = os.path.join('samples', dataset_name, "occlusions")
            save_images(orig_images, image_height, image_width, 10, 10, directory=occlusion_dir)
        else:
            samples=network.generate_images(num_images)
        save_images(samples, image_height, image_width, 10, 10, directory=SAMPLE_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from a trained PixelRNN')
    parser.add_argument('--occlusion', action='store_true')
    args = parser.parse_args()
    generate('mnist', occlusions=args.occlusion)
