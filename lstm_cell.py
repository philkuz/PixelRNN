# The MIT License (MIT)
#
# Copyright (c) 2016 Taehoon Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Class for the Diagonal LSTM Cell"""
import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from ops import conv1d


class DiagonalLSTMCell(rnn_cell.RNNCell):
  def __init__(self, hidden_dims, height, channel):
    self._num_unit_shards = 1
    self._forget_bias = 1.

    self._height = height
    self._channel = channel

    self._hidden_dims = hidden_dims
    self._num_units = self._hidden_dims * self._height
    self._state_size = self._num_units * 2
    self._output_size = self._num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, i_to_s, state, scope="DiagonalBiLSTMCell"):
    c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
    h_prev = tf.slice(state, [0, self._num_units], [-1, self._num_units]) # [batch, height * hidden_dims]

    # i_to_s : [batch, 4 * height * hidden_dims]
    input_size = i_to_s.get_shape().with_rank(2)[1]

    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    with tf.variable_scope(scope):
      # input-to-state (K_ss * h_{i-1}) : 2x1 convolution. generate 4h x n x n tensor.
      conv1d_inputs = tf.reshape(h_prev,
          [-1, self._height, 1, self._hidden_dims], name='conv1d_inputs') # [batch, height, 1, hidden_dims]

      tf.add_to_collection('i_to_s', i_to_s)
      tf.add_to_collection('conv1d_inputs', conv1d_inputs)

      conv_s_to_s = conv1d(conv1d_inputs,
          4 * self._hidden_dims, 2, scope='s_to_s') # [batch, height, 1, hidden_dims * 4]
      s_to_s = tf.reshape(conv_s_to_s,
          [-1, self._height * self._hidden_dims * 4]) # [batch, height * hidden_dims * 4]

      tf.add_to_collection('conv_s_to_s', conv_s_to_s)
      tf.add_to_collection('s_to_s', s_to_s)

      lstm_matrix = tf.sigmoid(s_to_s + i_to_s)

      # i = input_gate, g = new_input, f = forget_gate, o = output_gate
      i, g, f, o = tf.split(1, 4, lstm_matrix)

      c = f * c_prev + i * g
      h = tf.mul(o, tf.tanh(c), name='hid')


    new_state = tf.concat(1, [c, h])
    return h, new_state