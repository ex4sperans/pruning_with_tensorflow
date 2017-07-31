import os

import tensorflow as tf
import numpy as np

class BaseNetwork:

    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            # to save GPU resources
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config, graph=self.graph)
        return self._sess

    def init_variables(self, var_list):

        self.sess.run(tf.variables_initializer(var_list))

    def number_of_parameters(self, var_list):
        return sum(np.prod(v.get_shape().as_list()) for v in var_list)

    def save_model(self, path=None, sess=None, global_step=None, verbose=True):
        save_dir = path or self.model_path
        os.makedirs(save_dir, exist_ok=True)
        self.saver.save(sess or self.sess,
                        os.path.join(save_dir, 'model.ckpt'),
                        global_step=global_step)
        return self

    def load_model(self, path=None, sess=None, verbose=True):
        path = path or self.model_path
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt is None:
            raise FileNotFoundError('Can`t load a model. '\
            'Checkpoint does not exist.')    
        restore_path = ckpt.model_checkpoint_path
        self.saver.restore(sess or self.sess, restore_path)

        return self