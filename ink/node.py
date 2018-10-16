# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys
import os

from copy import deepcopy
# from itertools import izip
# from itertools import izip

import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

from .data import random_batches_data_provider
from .util import (
    get_num_labels,
    remap_labels,
    to_one_hot
)


NO_WINNER_LABEL = -1


def max_greater_than_threshold(threshold):
    def max_greater_than_threshold_impl(outputs):
        positions = outputs.argmax(axis=1)
        values = outputs[np.arange(outputs.shape[0]), positions]
        positions[np.where(values < threshold)] = NO_WINNER_LABEL
        return positions
    return max_greater_than_threshold_impl


class UniteNodeData(object):
    def __init__(self, features, labels, dataset_label_to_num_samples):
        """Create node builder.

        Model will be used for trainining of trn_features/trn_labels, and training will continue until given
        number of epochs since best error on vld_features/vld_labels.

        features should be 2D numpy matrix with (num samples, num features) shape.
        labels should be 1D numpy array with corresponding label idx.
        trn labels must consist of the same labels as the vld labels.

        num rows in trn_features size should be equal to num rows in trn_labels.
        num rows in vld_features size should be equal to num rows in vld_labels.
        """
        assert len(features.shape) == 2
        assert len(labels.shape) == 1
        assert features.shape[0] == labels.shape[0]

        self.features = features
        self.dataset_labels = labels
        self.dataset_label_to_node_label, self._node_labels = remap_labels(labels)
        self.mapped_node_labels = self._node_labels.copy()

        self.num_samples = self.features.shape[0]
        self.num_features = self.features.shape[1]
        self.num_labels = get_num_labels(self._node_labels)

        # we need to pass it from constructor and convert to node_label_num_samples
        self._dataset_label_to_num_samples = dataset_label_to_num_samples
        self.node_label_num_samples = np.zeros([self.num_labels], dtype=int)
        for dataset_label_idx, num_samples_for_label in dataset_label_to_num_samples.items():
            if dataset_label_idx in self.dataset_label_to_node_label:
                node_label_idx = self.dataset_label_to_node_label[dataset_label_idx]
                self.node_label_num_samples[node_label_idx] = num_samples_for_label

        self.node_label_to_active_outputs = {
            label_idx: [label_idx]
            for label_idx in range(self.num_labels)
        }

    def try_unite_outputs(self, outputs, winners, part_split_threshold):
        assert winners.shape == self._node_labels.shape

        votes = np.zeros([self.num_labels, self.num_labels], dtype=np.float32)
        for node_label_idx, winner_label_idx in zip(self._node_labels, winners):
            if winner_label_idx == NO_WINNER_LABEL:
                continue
            votes[node_label_idx][winner_label_idx] += 1
        votes /= self.node_label_num_samples

        new_node_label_to_active_outputs = {}
        for node_label_idx in range(self.num_labels):
            active_outputs = []

            node_label_votes = votes[node_label_idx]
            for winner_label_idx in range(self.num_labels):
                if node_label_votes[winner_label_idx] >= part_split_threshold:
                    active_outputs.append(winner_label_idx)
            if len(active_outputs) == 0:
                active_outputs.append(node_label_idx)

            new_node_label_to_active_outputs[node_label_idx] = active_outputs

        all_active_outputs = set([
            output
            for active_outputs in new_node_label_to_active_outputs.values()
            for output in active_outputs
        ])
        if len(all_active_outputs) == 1 or \
                self.node_label_to_active_outputs == new_node_label_to_active_outputs:
            return False

        self.node_label_to_active_outputs = new_node_label_to_active_outputs

        # remap data labels according to new_label_to_active_outputs
        tuples = zip(range(self.num_samples), self._node_labels, winners, outputs)
        for sample_idx, node_label_idx, winner_idx, sample_output in tuples:
            if winner_idx in self.node_label_to_active_outputs[node_label_idx]:
                self.mapped_node_labels[sample_idx] = winner_idx
            else:
                active_outputs = self.node_label_to_active_outputs[node_label_idx]
                active_outputs_values = [
                    sample_output[output_idx]
                    for output_idx in active_outputs
                ]
                self.mapped_node_labels[sample_idx] = \
                    active_outputs[np.argmax(active_outputs_values)]
        return True


def calculate_accuracy(outputs, expected_labels, label_to_active_outputs):
    correctly_classified_samples = 0
    winners = np.argmax(outputs, axis=1)
    for winner, expected_label in zip(winners, expected_labels):
        accepted_labels = label_to_active_outputs[expected_label]
        if winner in accepted_labels:
            correctly_classified_samples += 1
    return 1.0 * correctly_classified_samples / outputs.shape[0]


def create_predictor(session, predict_op, features_op, get_winner):
    def predictor(features):
        predicted = session.run(predict_op, feed_dict={features_op: features})
        return get_winner(predicted)
    return predictor


def visualize(predictor, X, y, title, path):
    assert X.shape[1] == 2
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    x0_grid, x1_grid = np.meshgrid(
        np.arange(x0_min, x0_max, 0.1),
        np.arange(x1_min, x1_max, 0.1)
    )
    grid = np.stack([x0_grid.ravel(), x1_grid.ravel()]).T.reshape([-1, 2])
    predicted = predictor(grid).reshape(x0_grid.shape)

    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
    axarr.contourf(x0_grid, x1_grid, predicted, alpha=0.2, cmap=cm.tab20)
    axarr.scatter(X[:, 0], X[:, 1], c=y, s=10, edgecolor=None)
    axarr.set_title(title)
    f.savefig(os.path.join(path, "{}.png".format(title)), dpi=300)


def train_one_node_impl(
    session,
    save_path,
    model,
    trn_node_data,
    vld_features,
    vld_labels,
    get_winner,
    unite_start=0,
    unite_timeout=1,
    part_split_threshold=0.2,
    batch_size=1,
    wait_best_error_time=50
):
    assert len(vld_features.shape) == 2
    assert len(vld_labels.shape) == 1
    assert vld_features.shape[0] == vld_labels.shape[0]
    assert vld_features.shape[1] == trn_node_data.num_features, \
        ": {} != {}".format(vld_features.shape[1], trn_node_data.num_features)

    saver = tf.train.Saver()

    num_epochs = -1
    best_epoch = -1
    best_accuracy = sys.float_info.min
    best_node_label_to_active_outputs = deepcopy(trn_node_data.node_label_to_active_outputs)

    since_last_unite = 0
    while num_epochs - best_epoch < wait_best_error_time:
        num_epochs += 1
        since_last_unite += 1

        trn_one_hot_labels = to_one_hot(trn_node_data.mapped_node_labels, trn_node_data.num_labels)
        batches = random_batches_data_provider(batch_size, trn_node_data.features, trn_one_hot_labels)

        for x_data, y_data in batches:
            session.run(model.train_step, {model.x: x_data, model.expected_y: y_data})
        if num_epochs > unite_start and since_last_unite > unite_timeout:
            outputs = session.run(model.predicted_y, feed_dict={model.x: trn_node_data.features})
            if trn_node_data.try_unite_outputs(outputs, get_winner(outputs), part_split_threshold):
                print("New mapping:")
                for label, active_outputs in trn_node_data.node_label_to_active_outputs.items():
                    print("\t{} -> {}".format(label, ",".join(map(str, sorted(active_outputs)))))

        h1, trn_outputs = session.run([model.h1, model.predicted_y], {model.x: trn_node_data.features})
        trn_accuracy = calculate_accuracy(
            trn_outputs,
            trn_node_data._node_labels,
            trn_node_data.node_label_to_active_outputs)

        vld_outputs = session.run(model.predicted_y, {model.x: vld_features})
        vld_accuracy = calculate_accuracy(vld_outputs, vld_labels, trn_node_data.node_label_to_active_outputs)

        is_new_best_epoch = vld_accuracy > best_accuracy
        if is_new_best_epoch:
            best_epoch = num_epochs
            best_accuracy = vld_accuracy
            best_mapped_labels = deepcopy(trn_node_data.mapped_node_labels)
            best_node_label_to_active_outputs = deepcopy(trn_node_data.node_label_to_active_outputs)

            saver.save(session, save_path + "model")

        print(
            "{} Accuracy at epoch {}:\t\t{}\t\t{}".format(
                '*' if is_new_best_epoch else ' ',
                num_epochs,
                trn_accuracy,
                vld_accuracy))
    trn_node_data.mapped_trn_labels = best_mapped_labels
    trn_node_data.node_label_to_active_outputs = best_node_label_to_active_outputs

    if trn_node_data.features.shape[1] == 2:
        predictor = create_predictor(session, model.predicted_y, model.x, get_winner)
        visualize(predictor, trn_node_data.features, trn_node_data.mapped_node_labels, title='mapped', path=save_path)
        visualize(predictor, trn_node_data.features, trn_node_data._node_labels, title='original', path=save_path)

    return {
        "best_epoch": best_epoch,
        "accuracy": best_accuracy,
        "dataset_label_to_node_label": trn_node_data.dataset_label_to_node_label,
        "node_label_to_active_output_neurons": best_node_label_to_active_outputs
    }
