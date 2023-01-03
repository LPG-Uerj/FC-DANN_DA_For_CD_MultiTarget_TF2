'''
import tensorflow as tf

@tf.custom_gradient
def GradientReversalOperator(x, l):
	def grad(dy):
		return -1 * dy * l
	return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        self.num_calls = 0

    def __call__(self, x, l=1.0):        
        self.num_calls += 1
        return GradientReversalOperator(x, l)

flip_gradient = GradientReversalLayer()
'''