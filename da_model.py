import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from layers import GradientReversalLayer

class DomainAdaptationModel(tf.keras.Model):

    def __init__(self, encoder_model,decoder_model, domain_discriminator_model):
        super(DomainAdaptationModel, self).__init__()

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.grl = GradientReversalLayer()
        self.domain_discriminator = domain_discriminator_model
        self.main_network = MainNetwork(encoder_model,decoder_model)
        self.discriminator_network = DomainRegressorNetwork(encoder_model,domain_discriminator_model)

    def call(self, inputs):
        input_img, l = inputs

        encoder_output, feature_output = self.encoder_model(input_img)

        discriminator_input = self.grl([encoder_output,l])
        discriminator_output,discriminator_logits = self.domain_discriminator(discriminator_input)

        segmentation_output = self.decoder_model([encoder_output,feature_output])

        return segmentation_output, discriminator_output, discriminator_logits    

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


class DomainRegressorNetwork(Model):

    def __init__(self, encoder_model,domain_discriminator):    
        super(DomainRegressorNetwork, self).__init__()
        self.encoder_model = encoder_model
        self.grl = GradientReversalLayer()
        self.domain_discriminator_model = domain_discriminator
    
    def call(self, inputs):
        input_img, l = inputs
        encoder_output,_ = self.encoder_model(input_img)
        discriminator_input = self.grl([encoder_output,l])
        discriminator_output,discriminator_logits = self.domain_discriminator_model(discriminator_input)
        
        return discriminator_output, discriminator_logits

    def get_config(self):
        config = super(DomainRegressorNetwork, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)
