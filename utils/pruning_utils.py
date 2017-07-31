import collections

import numpy as np

def prune_weights(weights, pruning_threshold):

    small_weights = np.abs(weights) < pruning_threshold
    weights[small_weights] = 0
    values = weights[weights != 0]
    indices = np.transpose(np.nonzero(weights))
    return values, indices

def get_sparse_values_indices(weights):

    values = weights[weights != 0]
    indices = np.transpose(np.nonzero(weights))
    return values, indices

def mask_for_big_values(weights, pruning_threshold):

    small_weights = np.abs(weights) < pruning_threshold
    return np.logical_not(small_weights)

def calculate_number_of_sparse_parameters(sparse_layers):

    total_count = 0

    for layer in sparse_layers:

        total_count += layer.values.nbytes // 4
        total_count += layer.indices.nbytes // 4
        total_count += layer.dense_shape.nbytes // 4
        total_count += layer.bias.nbytes // 4

    return total_count

class SparseLayer(collections.namedtuple('SparseLayer',
                                         ['values',
                                          'indices',
                                          'dense_shape',
                                          'bias'])):

    """An auxilary class to represent sparse layer"""
    pass