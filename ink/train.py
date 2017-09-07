#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import json
from collections import defaultdict

from .data import *
from .model import *
from .node import max_greater_than_threshold, UniteNodeData, train_one_node_impl
from .util import get_num_labels, remap_labels


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


def train_one_node(recursive, model_builder, trn_node_data, vld_features, vld_labels, save_path, get_winner, args):
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
    new_node_pathes = {output_idx: (save_path + str(output_idx) + "/") for output_idx in new_nodes.keys()}
    info["output_neuron_to_model_path"] = new_node_pathes
    info["recursive"] = recursive
    json.dump(
        info,
        open(save_path + "node_info.json", "w"),
        sort_keys=True,
        indent=4)

    if recursive:
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
                recursive,
                model_builder,
                new_node_data,
                new_vld_features,
                new_vld_labels,
                new_node_pathes[output_idx],
                get_winner,
                args)


def train_tree(args, recursive):
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
        recursive,
        model_builder,
        trn_node_data,
        vld_features,
        vld_labels,
        args.save_to_directory,
        max_greater_than_threshold(args.unite_threshold),
        args)
