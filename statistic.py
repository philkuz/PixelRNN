"""
The MIT License (MIT)

Copyright (c) 2016 Taehoon Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import numpy as np
import tensorflow as tf
from logging import getLogger

class Statistic(object):
    def __init__(self, sess, data, model_dir, variables, test_step, max_to_keep=20):
        self.sess = sess
        self.test_step = test_step
        self.reset()

        with tf.variable_scope('t'):
          self.t_op = tf.Variable(0, trainable=False, name='t')
          self.t_add_op = self.t_op.assign_add(1)

        self.model_dir = model_dir
        self.saver = tf.train.Saver(variables + [self.t_op], max_to_keep=max_to_keep)
        self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['train_l', 'test_l']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.scalar_summary('%s/%s' % (data, tag), self.summary_placeholders[tag])

    def reset(self):
        pass

    def on_step(self, train_l, test_l):
        self.t = self.t_add_op.eval(session=self.sess)

        self.inject_summary({'train_l': train_l, 'test_l': test_l}, self.t)

        self.save_model(self.t)
        self.reset()

    def get_t(self):
        return self.t_op.eval(session=self.sess)

    def inject_summary(self, tag_dict, t):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
          self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, t)

    def save_model(self, t):

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.saver.save(self.sess, self.model_dir, global_step=t)

    def load_model(self):
        tf.initialize_all_variables().run()

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.model_dir, ckpt_name)
            self.saver.restore(self.sess, fname)

        self.t = self.t_add_op.eval(session=self.sess)