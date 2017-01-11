import tensorflow as tf
import numpy as np

WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()

def conv2d(input, num_outputs, kernel_height, kernel_width, mask_type='A', scope='conv2d'):
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