import tensorflow as tf
import numpy as np
from utils import get_shape
WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
from lstm_cell import DiagonalLSTMCell

def conv1d(input, num_outputs, kernel_size, scope='conv1d'):
    with tf.variable_scope(scope):
        batch_size, image_height, image_width, num_channels = get_shape(input)
        kernel_height, kernel_width = kernel_size, 1
        # initialize kernel weights
        weights_shape = [kernel_height, kernel_width, num_channels, num_outputs]
        weights = tf.get_variable("weights", weights_shape, tf.float32, WEIGHT_INITIALIZER, None)

    stride_shape = [1, 1, 1, 1]
    outputs = tf.nn.conv2d(input, weights, stride_shape, padding='SAME', name='conv1d_outputs')
    return outputs

def conv2d(input, num_outputs, kernel_height, kernel_width, mask_type='A', scope='conv2d'):
    with tf.variable_scope(scope):
        batch_size, image_height, image_width, num_channels = get_shape(input)

        center_height = kernel_height / 2
        center_width = kernel_width / 2

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
def skew(inputs, scope="skew"):
    with tf.name_scope(scope):
        batch, width, height, channel = get_shape(inputs)
        new_width = width + height
        skewed_rows = []# inputs = tf.zeros([batch, width * 2 - 1 , height, channel])
        rows = tf.unpack(tf.transpose(inputs, [2, 0, 3, 1 ])) # [height, batch, channel, width]

        for i, row in enumerate(rows):
            squeezed_row = tf.squeeze(row, [0]) # [batch, channel, width]
            reshaped_row = tf.reshape(squeezed_row, [-1, width]) # [batch * channel, width]
            padded_row = tf.pad(reshaped_row, (0, 0), (i, height - 1 - i))

            unsqueezed_row = tf.reshape(padded_row, [-1, channel, new_width])  # [batch, channel, width*2-1]
            new_row = tf.transpose(unsqueezed_row, [0, 2, 1])  # [batch, width*2-1, channel]

            assert get_shape(new_row) == [batch, new_width, channel], "wrong shape of skewed row"
            skewed_rows.append(new_row)
        skewed_inputs = tf.pack(skewed_rows, axis=1, name="skewed_inputs")
        assert get_shape(skewed_inputs) == [None, height, new_width, channel], "wrong shape of skewed input"

    return skewed_inputs

def unskew(skewed_outputs, scope="unskew"):
    pass



def diagonal_lstm(inputs, hidden_dims, scope='diagonal_lstm'):
    with tf.variable_scope(scope):
        tf.add_to_collection('lstm_inputs', inputs)

        skewed_inputs = skew(inputs, scope="skewed_i")
        tf.add_to_collection('skewed_lstm_inputs', skewed_inputs)

        # input-to-state (K_is * x_i) : 1x1 convolution. generate 4h x n x n tensor.
        input_to_state = conv2d(skewed_inputs, hidden_dims * 4, 1, 1, mask_type="B", scope="i_to_s")
        column_wise_inputs = tf.transpose(
            input_to_state, [0, 2, 1, 3]) # [batch, width, height, hidden_dims * 4]

        tf.add_to_collection('skewed_conv_inputs', input_to_state)
        tf.add_to_collection('column_wise_inputs', column_wise_inputs)

        batch, width, height, channel = get_shape(column_wise_inputs)
        rnn_inputs = tf.reshape(column_wise_inputs,
            [-1, width, height * channel]) # [batch, max_time, height * hidden_dims * 4]

        tf.add_to_collection('rnn_inputs', rnn_inputs)

        rnn_input_list = [tf.squeeze(rnn_input, squeeze_dims=[1])
            for rnn_input in tf.split(split_dim=1, num_split=width, value=rnn_inputs)]

        cell = DiagonalLSTMCell(hidden_dims, height, channel)

        output_list, state_list = tf.nn.rnn(cell,
          inputs=rnn_input_list, dtype=tf.float32) # width * [batch, height * hidden_dims]

        packed_outputs = tf.pack(output_list, 1) # [batch, width, height * hidden_dims]
        width_first_outputs = tf.reshape(packed_outputs,
          [-1, width, height, hidden_dims]) # [batch, width, height, hidden_dims]

        skewed_outputs = tf.transpose(width_first_outputs, [0, 2, 1, 3])
        tf.add_to_collection('skewed_outputs', skewed_outputs)

        outputs = unskew(skewed_outputs)
        tf.add_to_collection('unskewed_outputs', outputs)

        return outputs
