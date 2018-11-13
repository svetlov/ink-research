#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf

from .apply import *
from .train import *


def apply_model(graph_path, variables_path, data_path, output_path):
    data = np.loadtxt(data_path, delimiter=',')[:, :-1]
    saver = tf.train.import_meta_graph(graph_path)
    session = tf.Session()
    saver.restore(session, variables_path)
    result = session.run('output/predicted_label:0', feed_dict={'input/x:0': data})
    with open(output_path, 'w') as wh:
        for label in result:
            print(label, file=wh)


def get_model_directory(path):
    return path.rstrip("/") + "/"


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='mode', dest='command')

    train = subparsers.add_parser('train')
    train.add_argument(
        "--save-to-directory",
        type=get_model_directory,
        required=True,
        help="path to store train model")
    train.add_argument("--train-data", help="tsv file containing train features and labels", required=True)
    train.add_argument("--validation-data", help="tsv file containing validation features and labels", required=True)
    train.add_argument("--num-hidden-neurons", default=3, type=int)
    train.add_argument("--batch-size", default=1, type=int)
    train.add_argument("--learning-rate", required=True, type=float)
    train.add_argument("--momentum", default=0.9, type=float)
    train.add_argument(
        "--unite-threshold",
        default=0.2,
        type=float,
        help="minimal actiavtion value for sample to vote")
    train.add_argument(
        "--unite-start",
        type=int,
        required=True,
        help="epoch number at which uniting procedure will be performed first time")
    train.add_argument(
        "--unite-timeout",
        type=int,
        required=True,
        help="number of epochs between uniting procudures")
    train.add_argument(
        "--part-split-threshold",
        type=float,
        default=0.2,
        help="minimal (num samples in group)/(num samples in class) ratio")
    train.add_argument(
        "--wait-best-error-time",
        type=int,
        required=True,
        help="number of epochs without new local minimum to stop training")
    train.add_argument(
        "--retrain-num-units",
        type=int,
        default=None,
        help='number of units used to retrain node')
    train.add_argument(
        "--loss-function",
        choices=['mse', 'crossentropy'],
        default='crossentropy',
        help='Loss function to train')

    predict = subparsers.add_parser('predict')
    predict.add_argument("--load-from-directory", type=get_model_directory, required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--ignore-last-column-in-data", action='store_true')
    predict.add_argument('--output', required=True)

    predict = subparsers.add_parser('calculate-accuracy')
    predict.add_argument("--load-from-directory", type=get_model_directory, required=True)
    predict.add_argument("--data", required=True)

    return parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    args = parse_args()
    if args.command == 'train' or args.command == 'train-root-node':
        recursive = (args.command == 'train')

        trn_features, trn_labels = read_data(args.train_data)
        vld_features, vld_labels = read_data(args.validation_data)

        def model_builder(num_labels, num_units=args.num_hidden_neurons):
            cost_builder = {
                'mse': mse_builder,
                'crossentropy': cross_entropy_builder
            }[args.loss_function]

            model = NeuralNetworkWithOneHiddenLayer(
                trn_features.shape[1],
                num_units,
                num_labels,
                optimizer=tf.train.MomentumOptimizer(
                    args.learning_rate,
                    args.momentum,
                    use_nesterov=False),
                cost_builder=cost_builder)
            return model

        unite_parameters = UniteParameters(
            get_winner=max_greater_than_threshold(args.unite_threshold),
            unite_start=args.unite_start,
            unite_timeout=args.unite_timeout,
            part_split_threshold=args.part_split_threshold,
        )

        train_tree(
            recursive=recursive,
            train_data=(trn_features, trn_labels),
            validation_data=(vld_features, vld_labels),
            save_path=args.save_to_directory,
            batch_size=args.batch_size,
            model_builder=model_builder,
            wait_best_error_time=args.wait_best_error_time,
            retrain_num_units=args.retrain_num_units,
            unite_parameters=unite_parameters,
        )
    elif args.command == 'predict':
        data = np.loadtxt(args.data, delimiter=",")
        if args.ignore_last_column_in_data:
            data = data[:, :-1]
        with tf.Graph().as_default():
            session = tf.Session()
            saver = tf.train.import_meta_graph(args.load_from_directory + "/model.meta")
            saver.restore(session, tf.train.latest_checkpoint(args.load_from_directory))
            result = session.run("output/y:0", feed_dict={"input/x:0": data})
            with open(args.output, 'w') as wh:
                np.savetxt(wh, result, fmt="%.6e", delimiter=",")
    elif args.command == "calculate-accuracy":
        apply_tree(args, recursive=True)
    else:
        raise RuntimeError("unsupported command")


if __name__ == "__main__":
    main()
