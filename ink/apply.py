#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
import tensorflow as tf

from .data import read_data
from .node import max_greater_than_threshold
from .util import as_int_keys


def apply_node(path, recursive, get_winner, features, labels):
    info = json.load(open(path + "node_info.json"))
    dataset_label_to_node_label = as_int_keys(info["dataset_label_to_node_label"])
    node_label_to_active_outputs = as_int_keys(info["node_label_to_active_output_neurons"])

    sample_indices, node_labels = [], []
    for iSample, dataset_label in enumerate(labels):
        if dataset_label in dataset_label_to_node_label:
            sample_indices.append(iSample)
            node_labels.append(dataset_label_to_node_label[dataset_label])
    node_features = features[sample_indices]
    node_dataset_labels = labels[sample_indices]  # useful in recursive case

    with tf.Graph().as_default():
        session = tf.Session()
        saver = tf.train.import_meta_graph(path + "/model.meta")
        checkpoint = tf.train.latest_checkpoint(path)
        assert checkpoint is not None
        saver.restore(session, checkpoint)
        outputs = session.run("output/y:0", feed_dict={"input/x:0": node_features})
        session.close()

    descendants = {k: [] for k, v in as_int_keys(info["output_neuron_to_model_path"]).items()}
    num_correctly_classified, num_rejected, num_incorrecly_classified = 0, 0, 0
    for iSample, winner in enumerate(get_winner(outputs)):
        if winner == -1:
            num_rejected += 1
        elif winner in node_label_to_active_outputs[node_labels[iSample]]:
            if winner not in descendants:
                num_correctly_classified += 1
            else:
                descendants[winner].append(iSample)
        else:
            num_incorrecly_classified += 1

    if recursive:
        for descendant, samples in descendants.items():
            if samples:
                descendant_path = path + str(descendant) + "/"
                descendants_features = node_features[samples]
                descendants_labels = node_dataset_labels[samples]
                d_correctly_classified, d_rejected, d_incorectly_classifed = apply_node(
                    descendant_path,
                    recursive,
                    get_winner,
                    descendants_features,
                    descendants_labels)
                print("Node at path {}".format(descendant_path))
                print("\t\tCorrectly: {}\n\t\tRejected: {}\n\t\tIncorrectly: {}".format(
                    d_correctly_classified, d_rejected, d_incorectly_classifed))
                num_correctly_classified += d_correctly_classified
                num_rejected += d_rejected
                num_incorrecly_classified += d_incorectly_classifed
    else:
        for descendant, samples in descendants.items():
            num_correctly_classified += len(samples)

    return num_correctly_classified, num_rejected, num_incorrecly_classified


def apply_tree(args, recursive):
    features, labels = read_data(args.data)
    assert len(features.shape) == 2
    assert len(labels.shape) == 1
    assert features.shape[0] == labels.shape[0]

    result = apply_node(args.load_from_directory, recursive, max_greater_than_threshold(0.2), features, labels)
    num_correctly_classified, num_rejected, num_incorrectly_classified = result

    print("\n===== Final result on tree =====")
    print("\t\tCorrectly: {}\n\t\tRejected: {}\n\t\tIncorrectly: {}".format(
        num_correctly_classified, num_rejected, num_incorrectly_classified))
    print("\n\t\tAccuracy: {}\n".format(1. * num_correctly_classified / labels.shape[0]))
