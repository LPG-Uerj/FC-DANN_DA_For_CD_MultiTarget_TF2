import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from layers import GradientReversalLayer
from discriminators import DomainDiscriminatorFullyConnected

class DomainAdaptationModel(tf.keras.Model):

    def __init__(self, input_shape, encoder_model, decoder_model, domain_discriminator_model):
        super(DomainAdaptationModel, self).__init__()

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.grl = GradientReversalLayer()
        self.domain_discriminator = DomainDiscriminatorFullyConnected()
        self.main_network = MainNetwork(encoder_model,decoder_model)

        self.inputs = [Input(shape = input_shape), Input(shape = ())]
        self.outputs = self.call(self.inputs)

        self.build(input_shape)

    def call(self, inputs):
        input_img, l = inputs

        encoder_output, feature_output = self.encoder_model(input_img)
        segmentation_output = self.decoder_model([encoder_output,feature_output])
        discriminator_input = self.grl([encoder_output,l])
        discriminator_logits,_ = self.domain_discriminator(discriminator_input)
        
        return segmentation_output, discriminator_logits

    def get_config(self):
        config = super(DomainAdaptationModel, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)



class MainNetwork(Model):

    def __init__(self, encoder_model,decoder_model):    
        super(MainNetwork, self).__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
    
    def call(self, inputs):
        input_img = inputs
        encoder_output,low_level_features = self.encoder_model(input_img)
        decoder_output = self.decoder_model([encoder_output,low_level_features])

        return decoder_output

    def get_config(self):
        config = super(MainNetwork, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)