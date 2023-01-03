import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from layers import GradientReversalLayer

class DomainAdaptationModel():

    def __init__(self, main_network_model, gradient_reversal_layer, domain_discriminator_model, classifier_loss, domainregressor_loss, training_optimizer):        
        self.main_network = main_network_model
        self.gradient_reversal_layer = gradient_reversal_layer
        self.domain_discriminator = domain_discriminator_model
        self.classifier_loss = classifier_loss
        self.domainregressor_loss = domainregressor_loss
        self.training_optimizer = training_optimizer
    
    def build(self):
        self.compile_model(self.main_network)
        self.compile_model(self.domain_discriminator)

    def compile_model(self, model, show_summary: bool = True):
        model.compile(optimizer = self.training_optimizer)        
        if show_summary:
            model.summary()