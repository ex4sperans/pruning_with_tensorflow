import tensorflow as tf

def get_second_dimension(tensor):
    return tensor.get_shape().as_list()[1]