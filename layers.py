import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer
import tf_slim as slim


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
    
class SlimSeparableConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride=(1, 1), padding='same', activation_fn = None, scope=None, **kwargs):
        super(SlimSeparableConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_fn = activation_fn
        self.scope = scope

    def build(self, input_shape):
        #with tf.name_scope(self.scope):
        self.separable_conv2d = slim.layers.separable_conv2d

    def call(self, inputs):
        #with tf.name_scope(self.scope):
        outputs = self.separable_conv2d(inputs, self.filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, activation_fn=self.activation_fn,scope=self.scope, reuse=False)
        return outputs
    

class SlimConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = 1, stride=(1, 1), padding='same', rate = 1, activation_fn = tf.nn.relu, normalizer_fn=slim.layers.batch_norm, scope=None, **kwargs):
        super(SlimConv2D, self).__init__(**kwargs)

        batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'updates_collections': slim.ops.GraphKeys.UPDATE_OPS,
        }

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rate=rate
        self.weights_regularizer=slim.regularizers.l2_regularizer(0.0001)
        self.weights_initializer=slim.initializers.variance_scaling_initializer()
        self.normalizer_fn=normalizer_fn
        self.normalizer_params=batch_norm_params
        self.activation_fn = activation_fn
        self.scope = scope

    def build(self, input_shape):
        #with tf.name_scope(self.scope):
        self.conv2d = slim.layers.conv2d

    def call(self, inputs):
        #with tf.name_scope(self.scope):
        outputs = self.conv2d(inputs, self.filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,rate=self.rate, weights_regularizer=self.weights_regularizer,weights_initializer=self.weights_initializer,normalizer_fn=self.normalizer_fn,normalizer_params=self.normalizer_params, activation_fn=self.activation_fn,scope = self.scope, reuse=False)
        return outputs