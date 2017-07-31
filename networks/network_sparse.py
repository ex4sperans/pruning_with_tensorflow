from typing import Union

import tensorflow as tf
from tqdm import tqdm

from networks.network_dense import FullyConnectedClassifier
from utils import tensorflow_utils
from utils import pruning_utils

class FullyConnectedClassifierSparse(FullyConnectedClassifier):

    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 sparse_layers: list,
                 model_path: str,
                 activation_fn=tf.nn.relu,
                 scope='FullyConnectedClassifierSparse',
                 verbose=True):

        self.input_size = input_size
        self.n_classes = n_classes
        self.sparse_layers = sparse_layers
        self.model_path = model_path
        self.activation_fn = activation_fn
        self.scope = scope
        self.verbose = verbose

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.scope):

                self._create_placeholders()

                self.logits = self._build_network(inputs=self.inputs,
                                                  sparse_layers=self.sparse_layers,
                                                  activation_fn=self.activation_fn)

                self.loss = self._create_loss(logits=self.logits,
                                              labels=self.labels)

                self._create_metrics(logits=self.logits,
                                     labels=self.labels,
                                     loss=self.loss)

                self.saver = self._create_saver(tf.global_variables())
                self.init_variables(tf.global_variables())

                if self.verbose:
                    print('\nSuccessfully created graph for {model}.'.format(
                                                            model=self.scope))
                    print('Number of parameters (four bytes == 1 parameter): {}.\n'.format(
                        pruning_utils.calculate_number_of_sparse_parameters(
                                                            self.sparse_layers)))

    def _create_placeholders(self):
    
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=(None, self.input_size),
                                     name='inputs')
    
        self.labels = tf.placeholder(dtype=tf.int64,
                                     shape=None,
                                     name='labels')

        # for compatibility with dense model
        self.keep_prob = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='keep_prob')

    def _build_network(self,
                       inputs: tf.Tensor,
                       sparse_layers: list,
                       activation_fn: callable) -> tf.Tensor:
    
        with tf.variable_scope('network'):
    
            net = inputs
    
            self.weight_tensors = []

            bias_initializer = tf.constant_initializer(0.1)

            for i, layer in enumerate(sparse_layers):
    
                with tf.variable_scope('layer_{layer}'.format(layer=i+1)):

                    # create variables based on sparse values                    
                    with tf.variable_scope('sparse'):

                        indicies = tf.get_variable(name='indicies',
                                                   initializer=layer.indices,
                                                   dtype=tf.int16)

                        values = tf.get_variable(name='values',
                                                 initializer=layer.values,
                                                 dtype=tf.float32)

                        dense_shape = tf.get_variable(name='dense_shape',
                                                      initializer=layer.dense_shape,
                                                      dtype=tf.int64)

                    # create a weight tensor based on the created variables
                    weights = tf.sparse_to_dense(tf.cast(indicies, tf.int64),
                                                 dense_shape,
                                                 values)

                    self.weight_tensors.append(weights)
        
                    name = 'bias'
                    bias = tf.get_variable(name=name,
                                           initializer=layer.bias)
    
                    net = tf.matmul(net, weights) + bias
    
                    if i < len(sparse_layers) - 1:
                        net = activation_fn(net)
    
            return net

    def _create_loss(self,
                     logits: tf.Tensor,
                     labels: tf.Tensor) -> tf.Tensor:
    
        with tf.variable_scope('loss'):
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=logits, labels=labels,
                                            name='classification_loss')
    
            classification_loss = tf.reduce_mean(classification_loss,
                                                 name='classification_loss_averaged')

            return classification_loss
