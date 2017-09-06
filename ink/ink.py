#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle
import os
import json
import random

from collections import defaultdict


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


def split_node_data(node_data):
    output_to_node_labels = defaultdict(list)
    for node_label_idx, outputs in node_data.node_label_to_active_outputs.items():
        for output_idx in outputs:
            output_to_node_labels[output_idx].append(node_label_idx)

    new_nodes = {}
    for output_idx, node_labels_in_output in output_to_node_labels.items():
        if len(node_labels_in_output) <= 1:
            continue
        # here we want to create new node data based on labels in output
        sample_indices = np.where(node_data.mapped_trn_labels == output_idx)[0]
        features = node_data.features[sample_indices]
        labels = node_data.dataset_labels[sample_indices]

        new_node_data = UniteNodeData(
            features,
            labels,
            node_data._dataset_label_to_num_samples)
        new_nodes[output_idx] = new_node_data
    return new_nodes


def train_one_node(model_builder, trn_node_data, vld_features, vld_labels, save_path, get_winner, args):
    print("\n===== Training new node with classes {} ===== \n".format(
        trn_node_data.dataset_label_to_node_label.keys()))
    print("\nDataset label to node label:")
    for dataset_label, node_label in trn_node_data.dataset_label_to_node_label.items():
        print("\t{} -> {}".format(dataset_label, node_label))
    print("Num samples in trn {}:".format(trn_node_data.mapped_node_labels.shape[0]))
    print("Num samples in vld {}".format(vld_labels.shape[0]))
    print("\n")

    mapping, remapped_vld_labels = remap_labels(vld_labels, trn_node_data.dataset_label_to_node_label)
    graph = tf.Graph()
    with graph.as_default():
        model = model_builder(trn_node_data.num_labels)
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        info = train_one_node_impl(
            session=session,
            save_path=save_path,
            model=model,
            trn_node_data=trn_node_data,
            vld_features=vld_features,
            vld_labels=remapped_vld_labels,
            get_winner=get_winner,
            unite_start=args.unite_start,
            unite_timeout=args.unite_timeout,
            part_split_threshold=args.part_split_threshold,
            batch_size=args.batch_size,
            wait_best_error_time=args.wait_best_error_time)

        vld_outputs = session.run(model.predicted_y, {model.x: vld_features})
        vld_output_winners = get_winner(vld_outputs)
        session.close()

    new_nodes = split_node_data(trn_node_data)
    new_node_pathes = {output_idx: args.save_to_directory + str(output_idx) for output_idx in new_nodes.keys()}
    info["output_neuron_to_model_path"] = new_node_pathes
    json.dump(
        info,
        open(save_path + "node_info.json", "w"),
        sort_keys=True,
        indent=4)
    print("\n")

    for output_idx, new_node_data in new_nodes.items():
        dataset_labels = new_node_data.dataset_label_to_node_label.keys()
        samples_to_keep = []
        for iSample in xrange(vld_labels.shape[0]):
            if (vld_labels[iSample] in dataset_labels
                    and (
                        vld_output_winners[iSample] == output_idx
                        or
                        vld_output_winners[iSample] == -1
                    )):
                samples_to_keep.append(iSample)

        new_vld_features = vld_features[samples_to_keep]
        new_vld_labels = vld_labels[samples_to_keep]
        train_one_node(
            model_builder,
            new_node_data,
            new_vld_features,
            new_vld_labels,
            new_node_pathes[output_idx],
            get_winner,
            args)


def train_tree(args):
    trn_features, trn_labels = read_data(args.train_data)
    vld_features, vld_labels = read_data(args.validation_data)

    def model_builder(num_labels):
        model = NeuralNetworkWithOneHiddenLayer(
            trn_features.shape[1],
            args.num_hidden_neurons,
            num_labels,
            optimizer=tf.train.MomentumOptimizer(
                args.learning_rate,
                args.momentum,
                use_nesterov=False),
            cost_builder=cross_entropy_builder)
        return model

    dataset_label_to_num_samples = {}
    for label_idx in xrange(get_num_labels(trn_labels)):
        dataset_label_to_num_samples[label_idx] = np.where(trn_labels == label_idx)[0].size

    trn_node_data = UniteNodeData(
        trn_features,
        trn_labels,
        dataset_label_to_num_samples)

    train_one_node(
        model_builder,
        trn_node_data,
        vld_features,
        vld_labels,
        args.save_to_directory,
        max_greater_than_threshold(args.unite_threshold),
        args)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    args = parse_args()
    if args.command == 'train':
        train_tree(args)
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
        features, labels = read_data(args.data)
        info = json.load(open(args.load_from_directory + "node_info.json"))
        label_to_active_outputs = info["node_label_to_active_output_neurons"]
        label_to_active_outputs = {int(k): v for k, v in label_to_active_outputs.items()}
        with tf.Graph().as_default():
            session = tf.Session()
            saver = tf.train.import_meta_graph(args.load_from_directory + "/model.meta")
            saver.restore(session, tf.train.latest_checkpoint(args.load_from_directory))
            result = session.run("output/y:0", feed_dict={"input/x:0": features})
            accuracy = calculate_accuracy(result, labels, label_to_active_outputs)
            print("\n\nAccuracy is: {}%\n\n".format(accuracy))
    else:
        raise RuntimeError("unsupported command")


if __name__ == "__main__":
    main()
