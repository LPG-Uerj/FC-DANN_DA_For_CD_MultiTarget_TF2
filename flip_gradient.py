from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.compat.v1.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y

flip_gradient = FlipGradientBuilder()

#def flip_gradient(x, l=1.0):
#	positive_path = tf.stop_gradient(x * tf.cast(1 + l, tf.float32))
#	negative_path = -x * tf.cast(l, tf.float32)
#	return positive_path + negative_path
