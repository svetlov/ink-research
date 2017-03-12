#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from sklearn.cluster import KMeans


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--result')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = np.loadtxt(args.data, delimiter=',')

    results = []
    for num_clusters in xrange(3, 10):
        results.append(KMeans(num_clusters).fit_predict(data))

    with open(args.result, 'w') as wh:
        for result_line in zip(*results):
            print(",".join([str(value) for value in result_line]), file=wh)

