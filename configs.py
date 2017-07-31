import tensorflow as tf

class ConfigNetworkDense:

    input_size = 28 * 28
    n_classes = 10
    layer_sizes = [512, 512]
    dropout = 0.5
    weight_decay = 0.0005
    activation_fn = tf.nn.relu
    model_path = 'saved_models/network_dense'

    n_epochs = 25
    batch_size = 100

    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 10:
            return 1e-2
        elif epoch < 20:
            return 1e-3
        else:
            return 1e-4

class ConfigNetworkDensePruned:

    input_size = 28 * 28
    n_classes = 10
    layer_sizes = [512, 512]
    dropout = 0
    weight_decay = 0.0001
    activation_fn = tf.nn.relu
    model_path = 'saved_models/network_dense_pruned'
    pruning_threshold = 0.03

    n_epochs = 20
    batch_size = 100

    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 10:
            return 1e-3
        else:
            return 1e-4

class ConfigNetworkSparse:

    input_size = 28 * 28
    n_classes = 10
    activation_fn = tf.nn.relu
    model_path = 'saved_models/network_sparse'
    
    batch_size = 100