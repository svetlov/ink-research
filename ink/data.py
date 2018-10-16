# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import math
import numpy as np


def read_data(path):
    data = np.loadtxt(path, delimiter=",")
    return data[:, :-1], data[:, -1].astype(int)


def simple_data_provider(batch_size, features, labels):
    num_batches = int(math.floor(features.shape[0] / batch_size))
    for batch_idx in range(num_batches):
        low = batch_idx * batch_size
        high = low + batch_size
        yield [features[low:high], labels[low:high]]


def random_batches_data_provider(batch_size, features, labels):
    num_batches = int(math.floor(features.shape[0] / batch_size))
    indices = np.arange(num_batches * batch_size)
    np.random.shuffle(indices)
    for batch_idx in range(num_batches):
        low = batch_idx * batch_size
        high = low + batch_size
        batch_indices = indices[low:high]
        yield [features[batch_indices], labels[batch_indices]]
