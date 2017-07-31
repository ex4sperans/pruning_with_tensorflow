import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

train_data_provider = mnist.train
validation_data_provider = mnist.validation
test_data_provider = mnist.test

from networks import network_dense
from configs import ConfigNetworkDense as config_dense
from configs import ConfigNetworkDensePruned as config_pruned
from utils import plot_utils
from utils import pruning_utils

# at first, create classifier
classifier = network_dense.FullyConnectedClassifier(
                            input_size=config_pruned.input_size,
                            n_classes=config_pruned.n_classes,
                            layer_sizes=config_pruned.layer_sizes,
                            model_path=config_pruned.model_path,
                            dropout=config_pruned.dropout,
                            weight_decay=config_pruned.weight_decay,
                            activation_fn=config_pruned.activation_fn,
                            pruning_threshold=config_pruned.pruning_threshold)

# collect tf variables and correspoding optimizer variables
with classifier.graph.as_default():
    weight_matrices_tf = classifier.weight_matrices
    optimizer_matrices_tf = [v 
                              for v in tf.global_variables()
                              for w in weight_matrices_tf
                              if w.name[:-2] in v.name
                              and 'optimizer' in v.name]

# load model previously trained model
# and get values of weights and optimizer variables
weights, optimizer_weights = (classifier
                             .load_model(config_dense.model_path)
                             .sess.run([weight_matrices_tf,
                                        optimizer_matrices_tf]))

# plot weights distribution before pruning
weights = classifier.sess.run(weight_matrices_tf)
plot_utils.plot_histogram(weights,
                          'weights_distribution_before_pruning',
                          include_zeros=False)

# for each pair (weight matrix + optimizer matrix)
# get a binary mask to get rid of small values. 
# Than, based on this mask change the values of 
# the weight matrix and the optimizer matrix  

for (weight_matrix,
     optimizer_matrix,
     tf_weight_matrix,
     tf_optimizer_matrix) in zip(
     weights,
     optimizer_weights,
     weight_matrices_tf,
     optimizer_matrices_tf):

    mask = pruning_utils.mask_for_big_values(weight_matrix,
                                             config_pruned.pruning_threshold)
    with classifier.graph.as_default():
        # update weights
        classifier.sess.run(tf_weight_matrix.assign(weight_matrix * mask))
        # and corresponding optimizer matrix
        classifier.sess.run(tf_optimizer_matrix.assign(optimizer_matrix * mask))

# now, lets look on weights distribution (zero values are excluded)
weights = classifier.sess.run(weight_matrices_tf)
plot_utils.plot_histogram(weights,
                          'weights_distribution_after_pruning',
                          include_zeros=False)

accuracy, loss = classifier.evaluate(data_provider=test_data_provider,
                                     batch_size=config_pruned.batch_size)
print('Accuracy on test before fine-tuning: {accuracy}, loss on test: {loss}'.format(
                                                    accuracy=accuracy, loss=loss))

# fine-tune classifier 
classifier.fit(n_epochs=config_pruned.n_epochs,
               batch_size=config_pruned.batch_size,
               learning_rate_schedule=config_pruned.learning_rate_schedule,
               train_data_provider=train_data_provider,
               validation_data_provider=validation_data_provider,
               test_data_provider=test_data_provider)

# plot weights distribution again to see the difference
weights = classifier.sess.run(weight_matrices_tf)
plot_utils.plot_histogram(weights,
                          'weights_distribution_after_fine_tuning',
                          include_zeros=False)

