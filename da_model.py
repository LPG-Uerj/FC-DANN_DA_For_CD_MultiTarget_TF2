import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from layers import GradientReversalLayer

class DomainAdaptationModel(Model):

    def __init__(self, input_shape, main_network_model, domain_discriminator_model):    
        super(DomainAdaptationModel, self).__init__()
        self.main_network = main_network_model
        self.gradient_reversal_layer = GradientReversalLayer()  
        self.domain_discriminator = domain_discriminator_model

        self.inputs = [Input(shape = input_shape), Input(shape = (1,))]
        self.outputs = self.call(self.inputs)

        self.build()
    
    def call(self, inputs):
        input_img, l = inputs

        segmentation_output, feature_output = self.main_network(input_img)
        discriminator_input = self.gradient_reversal_layer([feature_output, l])
        discriminator_output,discriminator_logits = self.domain_discriminator(discriminator_input)

        return segmentation_output, discriminator_output, discriminator_logits

    def get_config(self):
        config = super(DomainAdaptationModel, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)

    def build(self):
        super(DomainAdaptationModel, self).build(self.inputs.shape if tf.is_tensor(self.inputs) else self.inputs)
        self.call(self.inputs)