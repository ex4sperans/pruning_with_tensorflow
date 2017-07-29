class ConfigNetworkDense:

    input_size = 28 * 28
    n_classes = 10
    layer_sizes = [512]

    n_epochs = 40
    batch_size = 100

    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 20:
            return 1e-3
        elif epoch < 30:
            return 1e-4
        else:
            return 1e-5