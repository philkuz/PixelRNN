import tensorflow as tf
from utils import *
import numpy as np
from ops import conv2d

USE_RESIDUALS = False

NUM_HIDDEN_UNITS = 6
INPUT_RECURRENT_LENGTH = 4
OUTPUT_RECURRENT_LENGTH = 4

class Network:
    def __init__(self, sess, image_height, image_width, num_channels):

        input_shape = [None, image_height, image_width, num_channels]
        self.inputs = tf.placeholder(tf.float32, input_shape)

        if USE_RESIDUALS:
            # add residuals here
        else:
            kernel_height, kernel_width = 7, 7
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
        self.output = last_input

