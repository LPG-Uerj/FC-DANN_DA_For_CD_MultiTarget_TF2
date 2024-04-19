import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer

@tf.custom_gradient
def GradientReversalOperator(x,l):
    y = tf.identity(x)
    def grad(dy):
        return -1. * dy * l, 0. * dy
    return y, grad

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, inputs):
        input,l = inputs
        return GradientReversalOperator(input,l)

class ReshapeTensor(Layer):

    def __init__(self, shape, **kwargs):
        super(ReshapeTensor, self).__init__(**kwargs)        
        self.shape = shape

    def call(self, inputs):
        #reshaped = tf.compat.v1.image.resize(inputs, self.shape)
        reshaped = tf.image.resize(inputs, self.shape, method=tf.image.ResizeMethod.BILINEAR)
        return reshaped

    def get_config(self):
        config = super(ReshapeTensor, self).get_config()
        config.update({'shape': self.shape})
        return config

    def from_config(cls, config):
        return cls(**config)


class ExpandDimensions(Layer):

    def __init__(self, axis = -1, **kwargs):
        super(ExpandDimensions, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        expanded = tf.expand_dims(inputs, axis = self.axis)
        return expanded

    def get_config(self):
        config = super(ExpandDimensions, self).get_config()
        config.update({'axis': self.axis})
        return config

    def from_config(cls, config):
        return cls(**config)