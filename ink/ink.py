#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle
import os
import json

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
    if args.command == 'train':
        train_tree(args, recursive=True)
    elif args.command == "train-root-node":
        train_tree(args, recursive=False)
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
