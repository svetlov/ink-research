#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import cPickle


import numpy as np
import tensorflow as tf

from .data import *
from .node import *
from .model import *
from .util import *


def apply_model(graph_path, variables_path, data_path, output_path):
    data = np.loadtxt(data_path, delimiter=',')[:, :-1]
    saver = tf.train.import_meta_graph(graph_path)
    session = tf.Session()
    saver.restore(session, variables_path)
    result = session.run('output/predicted_label:0', feed_dict={'input/x:0': data})
    with open(output_path, 'w') as wh:
        for label in result:
            print(label, file=wh)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='mode', dest='command')
    train = subparsers.add_parser('train')
    train.add_argument("--train-data", required=True)
    train.add_argument("--validation-data", required=True)
    train.add_argument("--num-hidden-neurons", default=3, type=int)
    train.add_argument("--batch-size", default=1, type=int)
    train.add_argument("--learning-rate", required=True, type=float)
    train.add_argument("--momentum", default=0.9, type=float)
    train.add_argument("--unite-threshold", default=0.2, type=float)
    train.add_argument("--unite-start", type=int, required=True)
    train.add_argument("--unite-timeout", type=int, required=True)
    apply = subparsers.add_parser('apply')
    apply.add_argument("--graph", required=True)
    apply.add_argument("--variables", required=True)
    apply.add_argument("--data", required=True)
    apply.add_argument('--output', required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    if args.command == 'train':
        trn_features, trn_labels = read_data(args.train_data)
        vld_features, vld_labels = read_data(args.validation_data)

        graph = tf.Graph()
        with graph.as_default():

            trn_node_data = UniteNodeData(
                trn_features,
                trn_labels,
                max_greater_than_threshold(0.2))

            model = NeuralNetworkWithOneHiddenLayer(
                trn_node_data.num_features,
                args.num_hidden_neurons,
                trn_node_data.num_labels,
                optimizer=tf.train.MomentumOptimizer(
                    args.learning_rate,
                    args.momentum,
                    use_nesterov=False),
                cost_builder=cross_entropy_builder
            )

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                # TODO add all required parameters
                train(
                    session,
                    model,
                    trn_node_data,
                    vld_features,
                    vld_labels,
                    unite_start=args.unite_start,
                    unite_timeout=args.unite_timeout,
                    batch_size=args.batch_size)
    elif args.command == 'apply':
        apply_model(args.graph, args.variables, args.data, args.output)
    else:
        raise RuntimeError("unsupported command")


if __name__ == "__main__":
    main()
