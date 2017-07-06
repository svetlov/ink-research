# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys

from copy import deepcopy
from itertools import izip

import numpy as np

from .data import random_batches_data_provider
from .util import (
    _check_correct_and_get_num_labels,
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
    def __init__(self, features, labels, get_winner):
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

        self._original_trn_features = features
        self._original_trn_labels = labels
        self._mapped_trn_labels = labels.copy()

        self._num_trn_samples = self._original_trn_features.shape[0]
        self.num_features = self._original_trn_features.shape[1]
        self.num_labels = _check_correct_and_get_num_labels(self._original_trn_labels)

        self._label_trn_num_samples = np.zeros([self.num_labels], dtype=int)
        for label_idx in xrange(self.num_labels):
            self._label_trn_num_samples[label_idx] = \
                np.where(self._original_trn_labels == label_idx)[0].size

        self.label_to_active_outputs = {
            label_idx: [label_idx]
            for label_idx in xrange(self.num_labels)
        }

        self.get_winner = get_winner

    def try_unite_outputs(self, outputs, part_split_threshold):
        winners = self.get_winner(outputs)
        assert winners.shape == self._original_trn_labels.shape

        votes = np.zeros([self.num_labels, self.num_labels], dtype=np.float32)
        for original_label_idx, winner_label_idx in izip(self._original_trn_labels, winners):
            if winner_label_idx == NO_WINNER_LABEL:
                continue
            votes[original_label_idx][winner_label_idx] += 1
        votes = votes / self._label_trn_num_samples

        new_label_to_active_outputs = {}
        for original_label_idx in xrange(self.num_labels):
            active_outputs = []

            original_label_votes = votes[original_label_idx]
            for winner_label_idx in xrange(self.num_labels):
                if original_label_votes[winner_label_idx] >= part_split_threshold:
                    active_outputs.append(winner_label_idx)
            if len(active_outputs) == 0:
                active_outputs.append(original_label_idx)

            new_label_to_active_outputs[original_label_idx] = active_outputs

        all_active_outputs = set([
            output
            for active_outputs in new_label_to_active_outputs.values()
            for output in active_outputs
        ])
        if len(all_active_outputs) == 1 or \
                self.label_to_active_outputs == new_label_to_active_outputs:
            return False

        self.label_to_active_outputs = new_label_to_active_outputs

        # remap data labels according to new_label_to_active_outputs
        tuples = izip(xrange(self._num_trn_samples), self._original_trn_labels, winners, outputs)
        for sample_idx, label_idx, winner_idx, sample_output in tuples:
            if winner_idx in self.label_to_active_outputs[label_idx]:
                self._mapped_trn_labels[sample_idx] = winner_idx
            else:
                active_outputs = self.label_to_active_outputs[label_idx]
                active_outputs_values = [
                    sample_output[output_idx]
                    for output_idx in active_outputs
                ]
                self._mapped_trn_labels[sample_idx] = \
                    active_outputs[np.argmax(active_outputs_values)]
        return True

    def build(self, unite_start, unite_timeout):
        assert self.num_epochs == 0
        with self.model.graph.as_default():
            session = tf.Session()
            session.run(tf.initialize_all_variables())

            since_last_unite = 0
            while self.num_epochs - self.best_epoch < 60:
                self.num_epochs += 1
                since_last_unite += 1
                trn_features, trn_labels = _balance_labels_and_shuffle(self._original_trn_features, self._mapped_trn_labels)
                # trn_features, trn_labels = self._original_trn_features, self._mapped_trn_labels
                trn_one_hot_labels = to_one_hot(trn_labels, self.num_labels)

                batches = random_batches_data_provider(self.model.batch_size, trn_features, trn_one_hot_labels)
                for x_data, y_data in batches:
                    session.run(self.model.train_step, {self.model.x: x_data, self.model.expected_y: y_data})
                if self.num_epochs > unite_start and since_last_unite > unite_timeout:
                    self.try_unite_outputs(session)

                trn_one_hot_labels = to_one_hot(self._mapped_trn_labels, self.num_labels)
                vld_one_hot_labels = to_one_hot(self._mapped_vld_labels, self.num_labels)
                trn_accuracy, h1, outputs = session.run([self.model.accuracy, self.model.h1, self.model.predicted_y], {
                        self.model.x: self._original_trn_features,
                        self.model.expected_y: trn_one_hot_labels
                    })
                self.features_and_hidden_outputs = np.hstack([self.features_and_hidden_outputs, h1, outputs])
                with open('./log_trn_hidden', 'w') as wh:
                    for line in self.features_and_hidden_outputs:
                        wh.write("\t".join([str(value) for value in line]))
                        wh.write("\n")

                vld_accuracy = session.run(self.model.accuracy, {
                        self.model.x: self._original_vld_features,
                        self.model.expected_y: vld_one_hot_labels
                    })
                is_new_best_epoch = self.check_new_best_accuracy(vld_accuracy)
                print("{}Accuracy at epoch {}:\t\t{}\t\t{}".format(
                    '* ' if is_new_best_epoch else '', self.num_epochs, trn_accuracy, vld_accuracy))

                if is_new_best_epoch:
                    # saver = tf.train.Saver(tf.all_trainable_variables())
                    saver = tf.train.Saver()
                    saver.save(session, 'best-epoch.ckpt')
                    saver.export_meta_graph('best-epoch.meta')

                    cPickle.dump(self.best_mapping, open('best-mapping.pkl', 'wb'))

                if self.num_epochs % 20 == 0:
                    print("Mapping at epoch {}".format(self.num_epochs))
                    for label, neuron in self.mapping.iteritems():
                        print("\t{} -> {}".format(label, neuron))

            print("all work is done, best epoch is {}".format(self.best_epoch))
            print("Mapping at best epoch:")
            for label, neuron in self.best_mapping.iteritems():
                print("\t{} -> {}".format(label, neuron))


def calculate_accuracy(outputs, expected_labels, label_to_active_outputs):
    correctly_classified_samples = 0
    winners = np.argmax(outputs, axis=1)
    for winner, expected_label in izip(winners, expected_labels):
        accepted_labels = label_to_active_outputs[expected_label]
        if winner in accepted_labels:
            correctly_classified_samples += 1
    return 1.0 * correctly_classified_samples / outputs.shape[0]


def train(
    session,
    model,
    trn_node_data,
    vld_features,
    vld_labels,
    unite_start = 0,
    unite_timeout = 1,
    part_split_threshold = 0.2,
    batch_size = 1,
    wait_best_epoch_time = 50
):
    assert len(vld_features.shape) == 2
    assert len(vld_labels.shape) == 1
    assert vld_features.shape[0] == vld_labels.shape[0]
    assert vld_features.shape[1] == trn_node_data.num_features, \
        ": {} != {}".format(vld_features.shape[1], trn_node_data.num_features)

    num_epochs = -1
    best_epoch = -1
    best_accuracy = sys.float_info.min
    best_label_to_active_outputs = deepcopy(trn_node_data.label_to_active_outputs)

    since_last_unite = 0
    while num_epochs - best_epoch < wait_best_epoch_time:
        num_epochs += 1
        since_last_unite += 1

        trn_features = trn_node_data._original_trn_features
        trn_labels = trn_node_data._mapped_trn_labels
        trn_one_hot_labels = to_one_hot(trn_labels, trn_node_data.num_labels)

        batches = random_batches_data_provider(batch_size, trn_features, trn_one_hot_labels)
        for x_data, y_data in batches:
            session.run(model.train_step, {model.x: x_data, model.expected_y: y_data})
        if num_epochs > unite_start and since_last_unite > unite_timeout:
            outputs = model.predicted_y.eval({model.x: trn_node_data._original_trn_features})
            if trn_node_data.try_unite_outputs(outputs, part_split_threshold):
                print("New mapping:")
                for label, active_outputs in trn_node_data.label_to_active_outputs.items():
                    print("{} -> {}".format(label, ",".join(map(str, sorted(active_outputs)))))

        trn_labels = trn_node_data._mapped_trn_labels
        h1, trn_outputs = session.run([model.h1, model.predicted_y], {
            model.x: trn_node_data._original_trn_features
        })
        trn_accuracy = calculate_accuracy(trn_outputs, trn_labels, trn_node_data.label_to_active_outputs)

        vld_outputs = session.run(model.predicted_y, {
            model.x: vld_features,
        })
        vld_accuracy = calculate_accuracy(vld_outputs, vld_labels, trn_node_data.label_to_active_outputs)

        is_new_best_epoch = vld_accuracy > best_accuracy
        if is_new_best_epoch:
            best_epoch = num_epochs
            best_accuracy = vld_accuracy
            best_label_to_active_outputs = deepcopy(trn_node_data.label_to_active_outputs)

        print(
            "{} Accuracy at epoch {}:\t\t{}\t\t{}".format(
                '*' if is_new_best_epoch else ' ',
                num_epochs,
                trn_accuracy,
                vld_accuracy))
