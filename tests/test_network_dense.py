import unittest

import tensorflow as tf

from networks import network_dense



class TestNetwork(unittest.TestCase):

    def setUp(self):

        input_size = 20
        n_classes = 5
        layer_sizes = [5, 10]

        self.network = network_dense.FullyConnectedClassifier(input_size=input_size,
                                                              n_classes=n_classes,
                                                              layer_sizes=layer_sizes)

    def test_logits_shape(self):

        self.assertEqual(self.network.logits.get_shape().as_list(), [None, 5])

    def test_loss_shape(self):

        self.assertEqual(self.network.loss.get_shape().as_list(), [])

    def test_train_operation(self):

        self.assertIsInstance(self.network.train_op, tf.Operation)


