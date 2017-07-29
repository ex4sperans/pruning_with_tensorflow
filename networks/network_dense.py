from typing import Union

import tensorflow as tf


def get_second_dimension(tensor):
    return tensor.get_shape().as_list()[1]


class FullyConnectedClassifier:

    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 layer_sizes: list,
                 activation_fn=tf.nn.relu,
                 dropout=0.75,
                 momentum=0.9,
                 scope='FullyConnectedNetwork'):

        self. input_size = input_size
        self.n_classes = n_classes
        self.layer_sizes = layer_sizes + [n_classes]
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.momentum = momentum
        self.scope = scope

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.scope):

                self._create_placeholders()
                self.logits = self._build_network(inputs=self.inputs,
                                                  layer_sizes=self.layer_sizes,
                                                  activation_fn=self.activation_fn,
                                                  keep_prob=self.keep_prob)
                self.loss = self._create_loss(self.logits, self.labels)
                self.train_op = self._create_optimizer(self.loss,
                                                       learning_rate=self.learning_rate,
                                                       momentum=momentum)


    def _create_placeholders(self):
    
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=(None, self.input_size),
                                     name='inputs')
    
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=None,
                                     name='labels')
    
        self.keep_prob = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='keep_prob')
    
        self.learning_rate = tf.placeholder(dtype=tf.float32,
                                            shape=(),
                                            name='learning_rate')
    
    def _build_network(self,
                       inputs: tf.Tensor,
                       layer_sizes: list,
                       activation_fn: callable,
                       keep_prob: Union[tf.Tensor, float]) -> tf.Tensor:
    
        with tf.variable_scope('network'):
    
            net = inputs
    
            for i, layer_size in enumerate(layer_sizes):
    
                with tf.variable_scope('layer_{layer}'.format(layer=i+1)):
                    name = 'weights'
                    shape = (get_second_dimension(net), layer_size)
                    weights = tf.get_variable(name=name, shape=shape)
        
                    name = 'bias'
                    shape = [layer_size]
                    bias = tf.get_variable(name=name, shape=shape)
    
                    net = tf.matmul(net, weights) + bias
    
                    if not i < len(layer_sizes) - 1:
                        net = activation_fn(net)
                        net = tf.nn.dropout(net, keep_prob=keep_prob)
    
            return net
    
    def _create_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    
        with tf.variable_scope('loss'):
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=logits, labels=labels,
                                            name='classification_loss')
    
            classification_loss = tf.reduce_mean(classification_loss,
                                                 name='classification_loss_averaged')
    
            return classification_loss

    def _create_optimizer(self,
                          loss: tf.Tensor,
                          learning_rate: Union[tf.Tensor, float],
                          momentum: Union[tf.Tensor, float]) -> tf.Operation:

        with tf.variable_scope('optimizer'):

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   momentum=momentum,
                                                   name='optimizer')
            global_step = tf.Variable(0)
            train_op = optimizer.minimize(loss, global_step=global_step)

            return train_op


    def fit(self,
            n_epochs: int,
            learning_rate_schedule: callable,
            train_data_provider,
            validation_data_provider):

        for epoch in range(n_epochs):

            pass





    def predict():

        pass