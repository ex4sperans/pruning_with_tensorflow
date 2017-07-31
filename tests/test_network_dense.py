import unittest
import shutil

import tensorflow as tf

from networks import network_dense

class TestNetworkGraph(unittest.TestCase):

    def test_shapes(self):

        input_size = 20
        n_classes = 5
        layer_sizes = [5, 10]

        network = network_dense.FullyConnectedClassifier(input_size=input_size,
                                                         n_classes=n_classes,
                                                         layer_sizes=layer_sizes,
                                                         model_path='temp',
                                                         verbose=False)

        self.assertEqual(network.logits.get_shape().as_list(), [None, 5])
        self.assertEqual(network.loss.get_shape().as_list(), [])
        self.assertIsInstance(network.train_op, tf.Operation)

        shapes = [[20, 5], [5, 10], [10, 5]]
        for v, shape in zip(network.weight_matrices, shapes):
            self.assertEqual(v.get_shape().as_list(), shape)



class TestNetworkSaveRestore(unittest.TestCase):

    def test_save_restore(self):

        input_size = 20
        n_classes = 5
        layer_sizes = [5, 10]

        network = network_dense.FullyConnectedClassifier(input_size=input_size,
                                                         n_classes=n_classes,
                                                         layer_sizes=layer_sizes,
                                                         model_path='temp',
                                                         verbose=False)

        variables_original = network.sess.run(network.weight_matrices)
        network.save_model()
        network.load_model()
        variables_restored = network.sess.run(network.weight_matrices)

        for original, restored in zip(variables_original, variables_restored):
            self.assertTrue((original == restored).all())

    def tearDown(self):
        shutil.rmtree('temp')

