from typing import Union

import tensorflow as tf
from tqdm import tqdm

from networks.network_base import BaseNetwork
from utils import tensorflow_utils

class FullyConnectedClassifier(BaseNetwork):

    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 layer_sizes: list,
                 activation_fn=tf.nn.relu,
                 dropout=0.25,
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
                self._create_metrics(logits=self.logits,
                                     labels=self.labels,
                                     loss=self.loss)
                self.init_variables(tf.global_variables())


    def _create_placeholders(self):
    
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=(None, self.input_size),
                                     name='inputs')
    
        self.labels = tf.placeholder(dtype=tf.int64,
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
                    shape = (tensorflow_utils.get_second_dimension(net), layer_size)
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
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

            return train_op

    def _create_metrics(self,
                        logits: tf.Tensor,
                        labels: tf.Tensor,
                        loss: tf.Tensor):

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def fit(self,
            n_epochs: int,
            batch_size: int,
            learning_rate_schedule: callable,
            train_data_provider,
            validation_data_provider,
            test_data_provider):

        n_iterations = train_data_provider.num_examples // batch_size

        for epoch in range(n_epochs):
            print('Starting epoch {epoch}.\n'.format(epoch=epoch+1))
            for iteration in tqdm(range(n_iterations), ncols=75):

                images, labels = train_data_provider.next_batch(batch_size)

                feed_dict = {self.inputs: images,
                             self.labels: labels,
                             self.learning_rate: learning_rate_schedule(epoch+1),
                             self.keep_prob: 1 - self.dropout} 

                self.sess.run(self.train_op, feed_dict=feed_dict)
    
            train_accuracy, train_loss = self.evaluate(train_data_provider,
                                                       batch_size=batch_size)
            validation_accuracy, validation_loss = self.evaluate(train_data_provider,
                                                                 batch_size=batch_size)

            print('\nEpoch {epoch} completed.'.format(epoch=epoch+1))
            print('Accuracy on train: {accuracy}, loss on train: {loss}'.format(
                                    accuracy=train_accuracy, loss=train_loss))
            print('Accuracy on validation: {accuracy}, loss on validation: {loss}'.format(
                                    accuracy=validation_accuracy, loss=validation_loss))

    def evaluate(self, data_provider, batch_size: int):

        fetches = [self.accuracy, self.loss]

        n_iterations = data_provider.num_examples // batch_size

        average_accuracy = 0
        average_loss = 0

        for iteration in range(n_iterations):

            images, labels = data_provider.next_batch(batch_size)

            feed_dict = {self.inputs: images,
                         self.labels: labels,
                         self.keep_prob: 1.0} 

            accuracy, loss = self.sess.run(fetches, feed_dict=feed_dict)
            
            average_accuracy += accuracy / n_iterations
            average_loss += loss / n_iterations

        return average_accuracy, average_loss

    def predict():

        pass