#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np


def get_num_labels(labels):
    """Check that labels forms continious sequence from 0 to max label idx.

    :param labels: numpy array of shape [nrows]
    :return: Number of unique labels
    """
    assert len(labels.shape) == 1
    return len(set(labels))


def to_one_hot(labels, n_columns=None):
    """Create one hot representation of labels"""
    n_columns = get_num_labels(labels) if n_columns is None else n_columns
    n_rows = labels.shape[0]

    binary = np.zeros([n_rows, n_columns], dtype='float')
    binary[np.arange(n_rows), labels] = 1.0
    return binary


def remap_labels(labels):
    assert len(labels.shape) == 1
    unique_labels = sorted(set(labels))

    mapping = {}
    for idx, label in enumerate(unique_labels):
        mapping[label] = idx

    remapped = np.zeros_like(labels)
    for sample_idx, label in enumerate(labels):
        remapped[sample_idx] = mapping[label]
    return mapping, remapped


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

    np.random.shuffle(indices)
    new_features = new_features[indices]
    new_labels = new_labels[indices]

    return new_features, new_labels
