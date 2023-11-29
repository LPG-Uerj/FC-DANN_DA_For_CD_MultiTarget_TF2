import numpy as np
import tensorflow as tf
from keras.losses import Loss

class WeightedCrossEntropyC(Loss):
    def __init__(self, class_weights=None, mask=None):
        super(WeightedCrossEntropyC, self).__init__()
        self.class_weights = class_weights
        self.mask = mask        

    def call(self, y_true, y_pred):        
        temp = -y_true * tf.math.log(y_pred + 1e-3)
        temp_weighted = tf.math.multiply(self.class_weights,temp)
        loss = tf.math.reduce_sum(temp_weighted, 3)
        #classifier_loss = tf.reduce_sum(self.mask * loss) / tf.reduce_sum(self.mask)
        classifier_loss = tf.math.divide_no_nan(tf.reduce_sum(self.mask * loss), tf.reduce_sum(self.mask))
        return classifier_loss


class DiscriminatorCrossEntropy(Loss):
    def __init__(self):
        super(DiscriminatorCrossEntropy, self).__init__()

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_pred, labels = y_true))


class BinaryCrossentropy(Loss):

    def __init__(self, **kwargs):
        super(BinaryCrossentropy, self).__init__(**kwargs)

    def call(self, y_true, y_pred):

        loss = tf.math.multiply(y_true, y_pred)
        loss = tf.math.reduce_sum(loss, axis = -1)
        loss = tf.math.log(loss)
        loss = tf.math.multiply(-1., loss)

        return loss