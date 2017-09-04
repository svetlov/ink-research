# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf


def cross_entropy_builder(predicted_y, expected_y):
    return tf.reduce_mean(
        -tf.reduce_sum(
            expected_y * tf.log(tf.clip_by_value(predicted_y, 1e-10, 1.0)),
            reduction_indices=[1]
        )
    )


class NeuralNetworkWithOneHiddenLayer(object):
    def __init__(self, num_features, num_hidden, num_labels, optimizer, cost_builder):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_labels = num_labels

        self.optimizer = optimizer

        with tf.name_scope('input'):
            self.x = tf.placeholder('float', shape=[None, num_features], name='x')
            self.expected_y = tf.placeholder('float', shape=[None, num_labels], name='y')

        with tf.name_scope('hidden'):
            self._W_1 = tf.Variable(tf.truncated_normal([num_features, self.num_hidden]), name='W')
            self._b_1 = tf.Variable(tf.truncated_normal([self.num_hidden]), name='b')

            self.h1 = tf.nn.relu(tf.matmul(self.x, self._W_1) + self._b_1)

        with tf.name_scope('output'):
            self._W_2 = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels]), name='W')
            self._b_2 = tf.Variable(tf.truncated_normal([num_labels]), name='b')

            self.predicted_y = tf.nn.softmax(tf.matmul(self.h1, self._W_2) + self._b_2, name="y")
            predicted_label = tf.argmax(self.predicted_y, 1, name='predicted_label')

        self.cost = cost_builder(self.predicted_y, self.expected_y)
        self.gvs = self.optimizer.compute_gradients(self.cost)
        self.train_step = self.optimizer.apply_gradients(self.gvs)
