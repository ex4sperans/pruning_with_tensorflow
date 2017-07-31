import unittest
import shutil

import tensorflow as tf
import numpy as np

from networks import network_sparse
from utils.pruning_utils import SparseLayer

class TestNetworkGraph(unittest.TestCase):

    def test_shapes(self):

        input_size = 20
        n_classes = 5
        sparse_layers = [SparseLayer(
                         values=np.array([1, 2, 3]).astype(np.float32),
                         indices=np.array([[0, 1], [1, 2], [3, 4]]).astype(np.int16),
                         dense_shape=np.array([5, 10]).astype(np.int64),
                         bias=np.array([1, 1, 1, 1, 1]).astype(np.float32)),
                         SparseLayer(
                         values=np.array([1, 2, 5]).astype(np.float32),
                         indices=np.array([[0, 2], [1, 2], [3, 4]]).astype(np.int16),
                         dense_shape=np.array([10, 5]).astype(np.int64),
                         bias=np.array([0, 0, 0, 0, 0]).astype(np.float32))]

        network = network_sparse.FullyConnectedClassifierSparse(
                                                input_size=input_size,
                                                n_classes=n_classes,
                                                sparse_layers=sparse_layers,
                                                model_path='temp',
                                                verbose=False)

        self.assertEqual(network.logits.get_shape().as_list(), [None, 5])
        self.assertEqual(network.loss.get_shape().as_list(), [])


class TestNetworkSaveRestore(unittest.TestCase):

    def test_save_restore(self):

        input_size = 20
        n_classes = 5
        sparse_layers = [SparseLayer(
                         values=np.array([1, 2, 3]).astype(np.float32),
                         indices=np.array([[0, 1], [1, 2], [3, 4]]).astype(np.int16),
                         dense_shape=np.array([5, 10]).astype(np.int64),
                         bias=np.array([1, 1, 1, 1, 1]).astype(np.float32)),
                         SparseLayer(
                         values=np.array([1, 2, 5]).astype(np.float32),
                         indices=np.array([[0, 2], [1, 2], [3, 4]]).astype(np.int16),
                         dense_shape=np.array([10, 5]).astype(np.int64),
                         bias=np.array([0, 0, 0, 0, 0]).astype(np.float32))]

        network = network_sparse.FullyConnectedClassifierSparse(
                                                input_size=input_size,
                                                n_classes=n_classes,
                                                sparse_layers=sparse_layers,
                                                model_path='temp',
                                                verbose=False)

        variables_original = network.sess.run(network.weight_tensors)
        network.save_model()
        network.load_model()
        variables_restored = network.sess.run(network.weight_tensors)

        for original, restored in zip(variables_original, variables_restored):
            self.assertTrue((original == restored).all())

    def tearDown(self):
        shutil.rmtree('temp')

