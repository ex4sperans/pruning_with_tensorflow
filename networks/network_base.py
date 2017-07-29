import tensorflow as tf

class BaseNetwork:

    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config, graph=self.graph)
        return self._sess

    def init_variables(self, var_list):

        self.sess.run(tf.variables_initializer(var_list))