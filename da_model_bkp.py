import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input

class DomainAdaptationModel(Model):

    def __init__(self, main_network_model, domain_discriminator_model, input_shape: tuple = (256, 256, 1)):
        super(DomainAdaptationModel, self).__init__()
        self.main_network = main_network_model
        self.domain_discriminator = domain_discriminator_model

        self.inputs = [Input(shape = input_shape), Input(shape = (1,))]
        self.outputs = self.call(self.inputs)

        self.build()

    def call(self, inputs):
        input_img, l = inputs

        segmentation_output, atrous_output = self.main_network(input_img)        
        discriminator_output = self.domain_discriminator(atrous_output)

        return segmentation_output, discriminator_output
    
    def build(self):
        super(DomainAdaptationModel, self).build(self.inputs.shape if tf.is_tensor(self.inputs) else self.inputs)
        self.call(self.inputs)