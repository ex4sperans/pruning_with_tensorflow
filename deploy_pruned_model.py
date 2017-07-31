import collections

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

test_data_provider = mnist.test

from networks import network_dense
from networks import network_sparse
from configs import ConfigNetworkDensePruned as config_pruned
from configs import ConfigNetworkSparse as config_sparse
from utils import plot_utils
from utils import pruning_utils

# create pruned classifier
classifier = network_dense.FullyConnectedClassifier(
                            input_size=config_pruned.input_size,
                            n_classes=config_pruned.n_classes,
                            layer_sizes=config_pruned.layer_sizes,
                            model_path=config_pruned.model_path,
                            dropout=config_pruned.dropout,
                            weight_decay=config_pruned.weight_decay,
                            activation_fn=config_pruned.activation_fn,
                            pruning_threshold=config_pruned.pruning_threshold)
# restore a model
classifier.load_model()

accuracy, loss = classifier.evaluate(data_provider=test_data_provider,
                                     batch_size=config_pruned.batch_size)
print('Accuracy on test with dense model (pruned): {accuracy}, loss on test: {loss}'.format(
                                                   accuracy=accuracy, loss=loss))

weight_matrices, biases = classifier.sess.run([classifier.weight_matricies,
                                               classifier.biases])
sparse_layers = []
# turn dense pruned weights into sparse indices and values
for weights, bias in zip(weight_matrices, biases):

    values, indices = pruning_utils.get_sparse_values_indices(weights)
    shape = np.array(weights.shape).astype(np.int64)
    sparse_layers.append(pruning_utils.SparseLayer(values=values.astype(np.float32),
                                                   indices=indices.astype(np.int16),
                                                   dense_shape=shape,
                                                   bias=bias))

# create sparse classifier
sparse_classifier = network_sparse.FullyConnectedClassifierSparse(
                            input_size=config_sparse.input_size,
                            n_classes=config_sparse.n_classes,
                            sparse_layers=sparse_layers,
                            model_path=config_sparse.model_path,
                            activation_fn=config_sparse.activation_fn)

# test sparse classifier
accuracy, loss = sparse_classifier.evaluate(data_provider=test_data_provider,
                                            batch_size=config_sparse.batch_size)
print('Accuracy on test with sparse model: {accuracy}, loss on test: {loss}'.format(
                                                   accuracy=accuracy, loss=loss))

# finally, save a sparse model
sparse_classifier.save_model()
