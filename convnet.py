import tensorflow as tf
from utils import *
import numpy as np
from ops import conv2d

USE_RESIDUALS = False

NUM_HIDDEN_UNITS = 6
INPUT_RECURRENT_LENGTH = 4
OUTPUT_RECURRENT_LENGTH = 4
COLOR_RANGE = 256
USE_MULTICHANNEL = False
LEARNING_RATE = 1e-3


class Network:
    def __init__(self, sess, image_height, image_width, num_channels):
        self.image_height = image_height
        self.image_width = image_width
        self.num_channels = num_channels
        self.sess = sess

        input_shape = [None, image_height, image_width, num_channels]
        self.inputs = tf.placeholder(tf.float32, input_shape)

        kernel_height, kernel_width = 7, 7
        if USE_RESIDUALS:
            self.conv_2d_inputs = conv2d(self.inputs, NUM_HIDDEN_UNITS * 2, kernel_height, kernel_width, 'A')
        else:

            # apply initial convlution layer with A masking
            self.conv_2d_inputs = conv2d(self.inputs, NUM_HIDDEN_UNITS, kernel_height, kernel_width, 'A')

        # construct LSTM convolutional layers
        self.recurrent_layers = []
        last_input = self.conv_2d_inputs
        for i in range(INPUT_RECURRENT_LENGTH):
            diag_bilstm_layer = # TODO PLUG BILSTM HERE w/input as Last_input
            last_input = diag_bilstm_layer
            self.recurrent_layers.append(diag_bilstm_layer)

        # construct post-recurrent layer convolutions with ReLU activation
        self.recurrent_layer_outputs = []
        for i in range(OUTPUT_RECURRENT_LENGTH):
            kernel_height, kernel_width = 1, 1
            conv_layer = conv2d(last_input, NUM_HIDDEN_UNITS, kernel_height, kernel_width, 'B', 'conv2d_out_%i' % i)
            recurrent_out = tf.nn.relu(conv_layer)
            self.recurrent_layer_outputs.append(recurrent_out)
            last_input = recurrent_out
        recurrent_out_logits = last_input

        # apply final convolution layer with B masking
        conv2d_recurrent_logits = conv2d(recurrent_out_logits,
                    COLOR_RANGE, [1, 1], 'B', scope='conv2d_recurrent_output')

        if USE_MULTICHANNEL or num_channels > 1:
            raise NotImplementedError("We don't support multiple image channels yet")

            # now calculate loss
            # flatten so each pixel has a 256-dim vector for all possible colors
            flattened_shape = [-1, image_height * image_width, COLOR_RANGE]
            conv2d_recurrent_logits_flattened = tf.reshape(conv2d_recurrent_logits, flattened_shape)
            inputs_flattened = tf.reshape(self.inputs, flattened_shape)

            # split each pixel into its own tensor (pixel value
            conv2d_pixels = tf.split(1, image_height * image_width, conv2d_recurrent_logits_flattened)
            input_pixels = tf.split(1, image_height * image_width, inputs_flattened)

            # squeeze out one-dimensional things
            predicted_pixels = [tf.squeeze(pixel, squeeze_dims=[1]) for pixel in conv2d_pixels]
            target_pixels = [tf.squeeze(pixel, squeeze_dims=[1]) for pixel in input_pixels]

            # normalize predictions via 256-way softmax
            softmaxed_predictions = [tf.nn.softmax(pixel) for pixel in predicted_pixels]

            # at this point, we have image_width*image_height pixel tensors,
            # each of which is a num_channels*COLOR_RANGE matrix
            # so that each channel has its softmax distribution over possible values


            split_predicitions_by_channel = tf.split(1, num_channels, softmaxed_predictions)
            split_actual_by_channel = tf.split(1, num_channels, target_pixels)

            def loss_per_pixel(predicted_pixel, target_pixel):

                return tf.nn.sampled_softmax_loss(predicted_pixel, tf.zeros_like(predicted_pixel),
                                                  predicted_pixel, target_pixel, 1, COLOR_RANGE)

            losses = [loss_per_pixel(predicted_pixel, target_pixel) for predicted_pixel, target_pixel in
                      zip(predicted_pixels, target_pixels)]
        else:
            assert(num_channels == 1, "Convnet only support single-channel images")

            self.output = tf.nn.sigmoid(conv2d_recurrent_logits)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                conv2d_recurrent_logits, self.inputs, name='loss'
            ))

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
        gradients_and_vars = optimizer.compute_gradients(self.loss)

        self.optimize = optimizer.apply_gradients(gradients_and_vars)

    def predict(self, images):
        return self.sess.run(self.output, feed_dict={self.inputs: images})

    def test(self, images, perform_update=False):
        # TODO change to feedforward or something like that
        if perform_update:
            _, cost = self.sess.run([self.optimize, self.loss],
                    feed_dict = {self.inputs: images})
        else:
            cost = self.sess.run(self.loss, feed_dict={self.inputs: images})
        return cost

    def generate_image(self, num_images=100, starting_pos=[0, 0], starting_image=None):
        """
        Generate an Image from a starting image
        :param num_images: The number of images that you want to generate
        :param starting_pos: The starting position [x, y]
        :param starting_image: The iamage
        :return:
        """
        if starting_image is not None:
            samples = starting_image.copy()
        else:
            samples = np.zeros((num_images, self.image_height, self.image_width, self.num_channels), dtype='float32')
        for i in range(starting_pos[1], self.image_height):
            for j in range(starting_pos[0], self.image_width):
                for k in range(self.num_channels):
                    next_sample = binarize(self.predict(samples))
                    samples[:, i, j, k] = next_sample[:, i, j, k]
        return samples


