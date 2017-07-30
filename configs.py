import tensorflow as tf

class ConfigNetworkDense:

    input_size = 28 * 28
    n_classes = 10
    layer_sizes = [512]
    dropout = 0.25
    weight_decay = 0.0005
    activation_fn = tf.nn.relu
    model_path = 'saved_models/network_dense'

    n_epochs = 2
    batch_size = 100

    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 20:
            return 1e-2
        elif epoch < 30:
            return 1e-3
        else:
            return 1e-4