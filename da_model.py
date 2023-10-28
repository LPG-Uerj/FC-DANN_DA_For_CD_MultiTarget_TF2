import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
#from layers import GradientReversalLayer

class DomainAdaptationModel(Model):

    def __init__(self, input_shape, encoder_model,decoder_model, domain_discriminator_model):
        super(DomainAdaptationModel, self).__init__()

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.domain_discriminator = domain_discriminator_model
        self.main_network = MainNetwork(input_shape,encoder_model,decoder_model)
        self.inputs = [Input(shape = input_shape), Input(shape = ())]
        self.outputs = self.call(self.inputs)

        self.build()
    
    def call(self, inputs):
        input_img = inputs

        encoder_output, feature_output = self.encoder_model(input_img)
        segmentation_output = self.decoder_model([encoder_output,feature_output])
        discriminator_output,discriminator_logits = self.domain_discriminator(encoder_output)

        return segmentation_output, discriminator_output, discriminator_logits    

    def get_config(self):
        config = super(DomainAdaptationModel, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)

    def build(self):
        super(DomainAdaptationModel, self).build(self.inputs.shape if tf.is_tensor(self.inputs) else self.inputs)
        self.call(self.inputs)

class MainNetwork(Model):

    def __init__(self, input_shape, encoder_model,decoder_model):    
        super(MainNetwork, self).__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model        

        self.inputs = [Input(shape = input_shape)]
        self.outputs = self.call(self.inputs)

        self.build()
    
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

    def build(self):
        super(MainNetwork, self).build(self.inputs.shape if tf.is_tensor(self.inputs) else self.inputs)
        self.call(self.inputs)

class DomainRegressorNetwork(Model):

    def __init__(self, input_shape, encoder_model,domain_discriminator):    
        super(DomainRegressorNetwork, self).__init__()
        self.encoder_model = encoder_model
        self.domain_discriminator_model = domain_discriminator                

        self.inputs = [Input(shape = input_shape), Input(shape = ())]
        self.outputs = self.call(self.inputs)

        self.build()
    
    def call(self, inputs):
        input_img, l = inputs
        encoder_output,_ = self.encoder_model(input_img)
        discriminator_output,discriminator_logits = self.domain_discriminator_model(encoder_output)
        
        return discriminator_output, discriminator_logits

    def get_config(self):
        config = super(DomainRegressorNetwork, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)

    def build(self):
        super(DomainRegressorNetwork, self).build(self.inputs.shape if tf.is_tensor(self.inputs) else self.inputs)
        self.call(self.inputs)
