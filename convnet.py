import tensorflow as tf
from utils import *


USE_RESIDUALS = False
WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
NUM_HIDDEN_UNITS = 6
INPUT_RECURRENT_LENGTH = 4
OUTPUT_RECURRENT_LENGTH = 4

def conv2d(input, num_outputs, kernel_height, kernel_width, mask_type='A', scope'conv2d'):
    with tf.variable_scope(scope):
        batch_size, image_height, image_width, num_channels = input.get_shape().as_list()

        center_height = kernel_height // 2
        center_width = kernel_width // 2

        # initialize kernel weights
        weights_shape = [kernel_height, kernel_width, num_channels, num_outputs]
        weights = tf.get_variable("weights", weights_shape, tf.float32, WEIGHT_INITIALIZER, None)

        # pre-convolution mask
        mask_shape = (kernel_height, kernel_width, num_channels, num_outputs)
        mask = np.ones(mask_shape, dtype=np.float32)
        mask[center_height, center_width + 1, :, :] = 0.0
        mask[center_height + 1, :, :, :] = 0.0

        # in type A, we do not allow a connection to the current focus of the kernel
        # which is its center pixel
        if mask_type == 'a':
            mask[center_height, center_width, :, :] = 0.0

        # apply the mask
        weights *= tf.constant(mask, dtype=tf.float32)
        # store the weights variable
        tf.add_to_collection('conv2d_weights_mask_%s' % mask_type, weights)

    stride_shape = [1, 1, 1, 1]
    outputs = tf.nn.conv2d(input, weights, stride_shape, padding='SAME', name='conv2d_outputs')
    tf.add_to_collection('conv2d_outputs', outputs)




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
        



