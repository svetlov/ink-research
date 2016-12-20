#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import argparse
import random
import cPickle

from itertools import izip

import numpy as np
import tensorflow as tf

NO_WINNER_LABEL = -1


def _check_correct_and_get_num_labels(labels):
    """Check that labels forms continious sequence from 0 to max label idx.

    :param labels: numpy array of shape [nrows]
    :return: Number of unique labels
    """
    assert len(labels.shape) == 1

    uniq_labels = set(labels)
    print(sorted(uniq_labels), range(len(uniq_labels)))
    assert sorted(uniq_labels) == range(len(uniq_labels))

    return len(uniq_labels)


def _to_one_hot(labels, n_columns=None):
    """Create one hot representation of labels"""
    n_columns = _check_correct_and_get_num_labels(labels) if n_columns is None else n_columns
    n_rows = labels.shape[0]

    binary = np.zeros([n_rows, n_columns], dtype='float')
    binary[np.arange(n_rows), labels] = 1.0
    return binary


def _filter_features_and_labels_by_keys(features, labels, good_labels):
    """Filter labels and corresponding features by array of good labels.

    :param features: numpy array of shape [nrows, nfeatures]
    :param labels: numpy array of shape [nrows]
    :param good_labels: array of labels to be presented in return values
    :return: features and labels with good labels, in the same order as in input features/labels.
    """
    assert len(features.shape) == 2 and len(labels.shape) == 1 and features.shape[0] == labels.shape[0]

    good_indices = []
    for label in good_labels:
        sample_indices = np.where(labels == label)[0]
        assert sample_indices.shape[0] > 0, "no samples for label {}".format(label)
        good_indices.append(sample_indices)
    good_indices = sorted(np.hstack(good_indices))
    return features[good_indices], labels[good_indices]


def _map_labels(labels, mapping):
    """Map each label in labels to new label according to mapping dictionary.

    :param labels: numpy array of shape [nrows]
    :param mapping: dictionary of {old_label: new_label}
    :return: mapped labels
    """
    assert len(labels.shape) == 1

    new_labels = labels.copy()
    for old_label, new_label in mapping.iteritems():
        new_labels[labels == old_label] = new_label
    return new_labels


def _balance_labels_and_shuffle(features, labels):
    """Balance number of presense of each label in features/labels dataset.

    In case, when one class have less samples, that it should be in output dataset, bootstrapping method used.
    :param features: numpy array of shape [nrows, nfeatures]
    :param labels: numpy array of shape [nrows]
    :return: shuffled features and labels with balanced labels
    """
    assert len(features.shape) == 2 and len(labels.shape) == 1 and features.shape[0] == labels.shape[0]

    unique_labels = set(labels)
    n_unique_labels = len(unique_labels)
    n_samples = features.shape[0]

    samples_per_label = int(n_samples / n_unique_labels)
    indices = np.arange(n_samples)

    new_features = np.empty(shape=features.shape, dtype=features.dtype)
    new_labels = np.empty(shape=[n_samples], dtype=labels.dtype)
    label_indices_dict = {}

    for label_idx, label in enumerate(unique_labels):
        ordered_label_indices = indices[labels == label]
        label_indices_dict[label] = ordered_label_indices
        label_indices = ordered_label_indices.copy()

        # use bootstrapping method to get more samples if required
        while label_indices.shape[0] < samples_per_label:
            label_indices = np.hstack([label_indices, ordered_label_indices])

        np.random.shuffle(label_indices)  # shuffle to get random samples, not the first ones
        label_indices = label_indices[:samples_per_label]

        from_, to_ = samples_per_label * label_idx, samples_per_label * (label_idx + 1)
        new_features[from_:to_] = features[label_indices]
        new_labels[from_:to_] = label

    # добиваем хвост примеров случайными
    unique_labels = list(unique_labels)
    for idx in xrange(samples_per_label * n_unique_labels, n_samples):
        label_idx = random.randint(0, len(unique_labels) - 1)
        label = unique_labels[label_idx]
        sample_idx = random.randint(0, label_indices_dict[label].shape[0] - 1)
        new_features[idx] = features[label_indices_dict[label][sample_idx]]
        new_labels[idx] = labels[label_indices_dict[label][sample_idx]]

    if np.any(new_labels > 10):
        print(new_labels)
        raise RuntimeError

    np.random.shuffle(indices)
    new_features = new_features[indices]
    new_labels = new_labels[indices]

    return new_features, new_labels


def cross_entropy_builder(predicted_y, expected_y):
    return tf.reduce_mean(
        -tf.reduce_sum(
            expected_y * tf.log(tf.clip_by_value(predicted_y, 1e-10, 1.0)),
            reduction_indices=[1]
        )
    )


def simple_data_provider(batch_size, features, labels):
    num_batches = int(math.floor(features.shape[0] / batch_size))
    for batch_idx in xrange(num_batches):
        low = batch_idx * batch_size
        high = low + batch_size
        yield [features[low:high], labels[low:high]]


def random_batches_data_provider(batch_size, features, labels):
    num_batches = int(math.floor(features.shape[0] / batch_size))
    indices = np.arange(num_batches * batch_size)
    np.random.shuffle(indices)
    for batch_idx in xrange(num_batches):
        low = batch_idx * batch_size
        high = low + batch_size
        batch_indices = indices[low:high]
        yield [features[batch_indices], labels[batch_indices]]


class NeuralNetworkWithOneHiddenLayer(object):
    def __init__(self, batch_size, num_hidden, optimizer_builder, cost_builder):
        self.batch_size = batch_size
        self.num_hidden = num_hidden

        self._optimizer_builder = optimizer_builder
        self._cost_builder = cost_builder

    def finalize(self, num_features, num_labels):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.x = tf.placeholder('float', shape=[None, num_features], name='x')
                self.expected_y = tf.placeholder('float', shape=[None, num_labels], name='y')

            with tf.name_scope('hidden'):
                self._W_1 = tf.Variable(tf.truncated_normal([num_features, self.num_hidden]), name='W')
                self._b_1 = tf.Variable(tf.truncated_normal([self.num_hidden]), name='b')

                h1 = tf.nn.relu(tf.matmul(self.x, self._W_1) + self._b_1)

            with tf.name_scope('output'):
                self._W_2 = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels]), name='W')
                self._b_2 = tf.Variable(tf.truncated_normal([num_labels]), name='b')

                self.predicted_y = tf.nn.softmax(tf.matmul(h1, self._W_2) + self._b_2)
                predicted_label = tf.argmax(self.predicted_y, 1, name='predicted_label')

            self.cost = self._cost_builder(self.predicted_y, self.expected_y)
            self.optimizer = self._optimizer_builder()
            self.gvs = self.optimizer.compute_gradients(self.cost)
            self.train_step = self.optimizer.apply_gradients(self.gvs)

            self.correct_prediction = tf.equal(predicted_label, tf.argmax(self.expected_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        return model


class TNodeBuilder(object):
    def __init__(self, model, trn_features, trn_labels, vld_features, vld_labels, mapping, get_winner):
        """Create node builder.

        Model will be used for trainining of trn_features/trn_labels, and training will continue until given
        number of epochs since best error on vld_features/vld_labels.

        features should be 2D numpy matrix with (num samples, num features) shape.
        labels should be 1D numpy array with corresponding label idx.
        trn labels must consist of the same labels as the vld labels.

        num rows in trn_features size should be equal to num rows in trn_labels.
        num rows in vld_features size should be equal to num rows in vld_labels.

        mapping should be dict, which map original label idx to expected output neuron idx. Also, it aims filtering
        goal, so only labels presented in mapping will be used for training.
        """
        assert len(trn_features.shape) == 2 and len(vld_features.shape) == 2
        assert len(trn_labels.shape) == 1 and len(vld_labels.shape) == 1
        assert trn_features.shape[1] == vld_features.shape[1]
        assert trn_features.shape[0] == trn_labels.shape[0]
        assert vld_features.shape[0] == vld_labels.shape[0]
        assert set(trn_labels) == set(vld_labels)

        features_used_in_training, labels_used_in_training = \
            _filter_features_and_labels_by_keys(trn_features, trn_labels, mapping)
        self._original_trn_features = features_used_in_training
        self._original_trn_labels = labels_used_in_training

        features_used_in_training, labels_used_in_training = \
            _filter_features_and_labels_by_keys(vld_features, vld_labels, mapping)
        self._original_vld_features = features_used_in_training
        self._original_vld_labels = labels_used_in_training

        self._original_labels_indices = {label_idx for label_idx in mapping.keys()}
        self._original_trn_label_to_num_samples = {
            label_idx: np.where(self._original_trn_labels == label_idx)[0].size
            for label_idx in self._original_labels_indices
        }
        self._original_vld_label_to_num_samples = {
            label_idx: np.where(self._original_vld_labels == label_idx)[0].size
            for label_idx in self._original_labels_indices
        }

        self._mapping = None
        self.mapping = mapping.copy()  # it's trigger generation of self._mapped_(trn|vld)_labels data members

        self.num_features = trn_features.shape[1]
        self.num_labels = len({output_neuron_idx for output_neuron_idx in self.mapping.values()})

        self.model = model.finalize(self.num_features, self.num_labels)
        self.get_winner = get_winner

        self.num_epochs = 0
        self.best_epoch = 0
        self.best_accuracy = None
        self.best_mapping = self.mapping.copy()

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, new_mapping):
        remap_required = False if self.mapping is not None else True

        if self.mapping is not None:
            # we should check that this mapping contains at least 2 active neurons
            num_active_neurons = len({neuron_idx for neuron_idx in new_mapping.values()})
            if num_active_neurons == 1:
                return

        for label_idx in self._original_labels_indices:
            if self.mapping is not None and new_mapping[label_idx] != self.mapping[label_idx]:
                print("remap label {} output from {} to {}".format(
                    label_idx, self.mapping[label_idx], new_mapping[label_idx]
                ))
                remap_required = True

        if remap_required:
            self._mapped_trn_labels = _map_labels(self._original_trn_labels, new_mapping)
            self._mapped_vld_labels = _map_labels(self._original_vld_labels, new_mapping)

        if self.mapping is None:
            # we should check that labels indices form sequence from 0 to num_labels
            mapped_trn_num_labels = _check_correct_and_get_num_labels(self._mapped_trn_labels)
            mapped_vld_num_labels = _check_correct_and_get_num_labels(self._mapped_vld_labels)
            assert mapped_trn_num_labels >= 2 and mapped_trn_num_labels == mapped_vld_num_labels

        self._mapping = new_mapping

    def try_unite_outputs(self, session):
        outputs = self.model.predicted_y.eval({self.model.x: self._original_trn_features}, session=session)

        votes = {label_idx: np.zeros([self.num_labels], dtype=int) for label_idx in self._original_labels_indices}
        winners = self.get_winner(outputs)
        assert winners.shape == self._original_trn_labels.shape

        for original_label_idx, winner_label_idx in izip(self._original_trn_labels, winners):
            if winner_label_idx == NO_WINNER_LABEL:
                continue
            votes[original_label_idx][winner_label_idx] += 1

        # create {original label idx -> output neuron idx with maximum votes for this original label idx}
        max_votes_neurons = {label_idx: votes[label_idx].argmax() for label_idx in self._original_labels_indices}
        # create {original label idx -> maximum votes for this original label idx}
        max_votes_counts = {label_idx: votes[label_idx].max() for label_idx in self._original_labels_indices}

        new_mapping = self.mapping.copy()
        for label_idx in self._original_labels_indices:
            if max_votes_counts[label_idx] > self._original_trn_label_to_num_samples[label_idx] * 0.5:
                new_mapping[label_idx] = max_votes_neurons[label_idx]

        self.mapping = new_mapping

    def check_new_best_accuracy(self, vld_accuracy):
        if self.best_accuracy is None or vld_accuracy > self.best_accuracy:
            self.best_accuracy = vld_accuracy
            self.best_epoch = self.num_epochs
            self.best_mapping = self.mapping.copy()
            return True
        return False

    def build(self):
        assert self.num_epochs == 0
        with self.model.graph.as_default():
            session = tf.Session()
            session.run(tf.initialize_all_variables())

            while self.num_epochs - self.best_epoch < 60:
                self.num_epochs += 1
                trn_features, trn_labels = _balance_labels_and_shuffle(self._original_trn_features, self._mapped_trn_labels)
                # trn_features, trn_labels = self._original_trn_features, self._mapped_trn_labels
                trn_one_hot_labels = _to_one_hot(trn_labels, self.num_labels)

                batches = random_batches_data_provider(self.model.batch_size, trn_features, trn_one_hot_labels)
                for x_data, y_data in batches:
                    session.run(self.model.train_step, {self.model.x: x_data, self.model.expected_y: y_data})
                self.try_unite_outputs(session)

                trn_one_hot_labels = _to_one_hot(self._mapped_trn_labels, self.num_labels)
                vld_one_hot_labels = _to_one_hot(self._mapped_vld_labels, self.num_labels)
                trn_accuracy = session.run(self.model.accuracy, {
                        self.model.x: self._original_trn_features,
                        self.model.expected_y: trn_one_hot_labels
                    })

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
                    saver.save(session, 'best-epoch.tf')
                    cPickle.dump(self.best_mapping, open('best-mapping.pkl', 'wb'))

                if self.num_epochs % 20 == 0:
                    print("Mapping at epoch {}".format(self.num_epochs))
                    for label, neuron in self.mapping.iteritems():
                        print("\t{} -> {}".format(label, neuron))

            print("all work is done, best epoch is {}".format(self.best_epoch))
            print("Mapping at best epoch:")
            for label, neuron in self.best_mapping.iteritems():
                print("\t{} -> {}".format(label, neuron))


def apply_model(graph_path, variables_path, data_path, output_path):
    data = np.loadtxt(data_path, delimiter=',')[:, :-1]
    saver = tf.train.import_meta_graph(graph_path)
    session = tf.Session()
    saver.restore(session, variables_path)
    result = session.run('output/predicted_label:0', feed_dict={'input/x:0': data})
    with open(output_path, 'w') as wh:
        for label in result:
            print(label, file=wh)


def read_train_validation(trn_path, vld_path):
    all_trn = np.loadtxt(trn_path, delimiter=',')
    all_vld = np.loadtxt(vld_path, delimiter=',')
    assert all_trn.shape[1] > 2 and all_trn.shape[1] == all_vld.shape[1]
    return all_trn[:, :-1], all_trn[:, -1].astype(int), all_vld[:, :-1], all_vld[:, -1].astype(int)


def max_greater_than_threshold(outputs):
    threshold = 0.2

    positions = outputs.argmax(axis=1)
    values = outputs[np.arange(outputs.shape[0]), positions]
    positions[np.where(values < threshold)] = NO_WINNER_LABEL
    return positions


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
    apply = subparsers.add_parser('apply')
    apply.add_argument("--graph", required=True)
    apply.add_argument("--variables", required=True)
    apply.add_argument("--data", required=True)
    apply.add_argument('--output', required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == 'train':
        trn_features, trn_labels, vld_features, vld_labels = read_train_validation(args.train_data, args.validation_data)

        model = NeuralNetworkWithOneHiddenLayer(
            batch_size=args.batch_size, num_hidden=args.num_hidden_neurons,
            optimizer_builder=lambda: tf.train.MomentumOptimizer(args.learning_rate, args.momentum, use_nesterov=False),
            cost_builder=cross_entropy_builder
        )

        n_unique_labels = set(trn_labels)

        node = TNodeBuilder(
            model,
            trn_features, trn_labels,
            vld_features, vld_labels,
            {i: i for i in xrange(len(n_unique_labels))},
            max_greater_than_threshold
        )
        node.build()
    elif args.command == 'apply':
        apply_model(args.graph, args.variables, args.data, args.output)
    else:
        raise RuntimeError("unsupported command")
