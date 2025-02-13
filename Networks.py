from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#-----para resunet-----
import os
import sys
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
#from tensorflow.contrib import layers as layers_lib
import tf_slim as slim
#--------------------

from BaseModels import *

class Unet():
    def __init__(self, args):
        super(Unet, self).__init__()
        self.args = args

    def build_Unet_Arch(self, input_data, name="Unet_Arch"):
        with tf.compat.v1.variable_scope(name):
            # Encoder definition
            o_c1 = self.general_conv2d(input_data, self.args.base_number_of_features, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_1')
            o_mp1 = tf.compat.v1.layers.max_pooling2d(
                o_c1, 2, 2, name=name + '_maxpooling_1')
            o_c2 = self.general_conv2d(o_mp1, self.args.base_number_of_features * 2, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_2')
            o_mp2 = tf.compat.v1.layers.max_pooling2d(
                o_c2, 2, 2, name=name + '_maxpooling_2')
            o_c3 = self.general_conv2d(o_mp2, self.args.base_number_of_features * 4, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_3')
            o_mp3 = tf.compat.v1.layers.max_pooling2d(
                o_c3, 2, 2, name=name + '_maxpooling_3')
            o_c4 = self.general_conv2d(o_mp3, self.args.base_number_of_features * 8, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_4')
            o_mp4 = tf.compat.v1.layers.max_pooling2d(
                o_c4, 2, 2, name=name + '_maxpooling_4')
            o_c5 = self.general_conv2d(o_mp4, self.args.base_number_of_features * 16, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_5')
            o_mp5 = tf.compat.v1.layers.max_pooling2d(
                o_c5, 2, 2, name=name + '_maxpooling_5')
            o_c6 = self.general_conv2d(o_mp5, self.args.base_number_of_features * 16, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_6')

            # Decoder definition
            o_d1 = self.general_deconv2d(o_c6, self.args.base_number_of_features * 8, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_1')
            o_me1 = tf.concat([o_d1, o_c5], 3)  # Skip connection
            o_d2 = self.general_deconv2d(o_me1, self.args.base_number_of_features * 4, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_2')
            o_me2 = tf.concat([o_d2, o_c4], 3)  # Skip connection
            o_d3 = self.general_deconv2d(o_me2, self.args.base_number_of_features * 2, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_3')
            o_me3 = tf.concat([o_d3, o_c3], 3)  # Skip connection
            o_d4 = self.general_deconv2d(o_me3, self.args.base_number_of_features, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_4')
            o_me4 = tf.concat([o_d4, o_c2], 3)  # Skip connection

            o_d5 = self.general_deconv2d(o_me4, self.args.base_number_of_features, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_5')
            o_me5 = tf.concat([o_d5, o_c1], 3)  # Skip connection
            logits = tf.compat.v1.layers.conv2d(
                o_me5, self.args.num_classes, 1, 1, 'SAME', activation=None)
            prediction = tf.nn.softmax(logits, name=name + '_softmax')

            #RETURN DO BOTTLENECK MODIFICADO: ORIGINAL ERA O_C5
            return logits, prediction, o_c6

    def build_Unet_Encoder(self, X, name="Unet_Encoder_Arch"):
        Layers = []
        with tf.compat.v1.variable_scope(name):
            for i in range(self.args.encoder_blocks + 1):
                if i == 0:
                    Layers.append(X)
                else:
                    Layers.append(self.general_conv2d(Layers[-1], self.args.base_number_of_features * (2**(i)), 3, stride=1,
                                                      padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_' + str(i)))
                    Layers.append(tf.compat.v1.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_' + str(i)))

            return Layers

    def build_Unet_Decoder(self, X, Encoder_Layers, name = "Unet_Decoder_Arch"):

        Layers = []
        with tf.compat.v1.variable_scope(name):
            for i in range(self.args.encoder_blocks + 1):
                if i == 0:
                    Layers.append(X)
                    Layers.append(tf.compat.v1.layers.dropout(Layers[-1], 0.2, name = name + 'dropout_' + str(i)))
                else:
                    Layers.append(self.general_deconv2d(Layers[-1], self.args.base_number_of_features * (2**(self.args.encoder_blocks - i)), 3, stride=2,
                                                 padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_' + str(i)))
                    if self.args.skip_connections:
                        d_shape = Layers[-1].get_shape().as_list()[1:]
                        for layer in Encoder_Layers:
                            e_shape = layer.get_shape().as_list()[1:]
                            if d_shape == e_shape:
                                Layers.append(tf.concat([Layers[-1], layer], 3))  # Skip connection
                                break

            Layers.append(tf.compat.v1.layers.conv2d(Layers[-1], self.args.num_classes, 1, 1, 'SAME', activation=None))
            Layers.append(tf.nn.softmax(Layers[-1], name=name + '_softmax'))

            return Layers

    def general_conv2d(self, input_data, filters=64,  kernel_size=7, stride=1, stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="conv2d"):
        with tf.compat.v1.variable_scope(name):
            conv = tf.compat.v1.layers.conv2d(
                input_data, filters, kernel_size, stride, padding, activation=None, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

            if do_norm:
                conv = tf.compat.v1.layers.batch_normalization(conv, momentum=0.9)

            if activation_function == "relu":
                conv = tf.nn.relu(conv, name='relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name='elu')

            return conv

    def general_deconv2d(self, input_data, filters=64, kernel_size=7, stride=1, stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="deconv2d"):
        with tf.compat.v1.variable_scope(name):
            deconv = tf.compat.v1.layers.conv2d_transpose(input_data, filters, kernel_size, (stride, stride), padding, activation=None, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

            if do_norm:
                deconv = tf.compat.v1.layers.batch_normalization(deconv, momentum=0.9)

            if activation_function == "relu":
                deconv = tf.nn.relu(deconv, name='relu')
            if activation_function == "leakyrelu":
                deconv = tf.nn.leaky_relu(deconv, alpha=relu_factor)
            if activation_function == "elu":
                deconv = tf.nn.elu(deconv, name='elu')

            return deconv

class Domain_Regressors():
    def __init__(self, args):
        super(Domain_Regressors, self).__init__()
        self.args = args
    #=============================GABRIEL: DOMAIN_CLASSIFIER=============================
    def build_Domain_Classifier_Arch(self, input_data, name = "Domain_Classifier_Arch"):
        Layers = []
        with tf.compat.v1.variable_scope(name):
            #Domain Classifier Definition: 2x (Fully_Connected_1024_units + ReLu) + Fully_Connected_1_unit + Logistic

            Layers.append(tf.compat.v1.layers.flatten(input_data))
            Layers.append(self.general_dense(Layers[-1], units=1024, activation_function="relu", name=name + '_dense1'))
            Layers.append(self.general_dense(Layers[-1], units=1024, activation_function="relu", name=name + '_dense2'))
            #SERÁ AQUI O ERRO??? ACTIVATION NONE NA VERDADE USA UMA ATIVAÇÃO LINEAR
            Layers.append(tf.compat.v1.layers.dense(Layers[-1], units=self.args.num_domains, activation=None))
            #ALÉM DISSO, AQUI EU USO UMA SOFTMAX, MAS EM MODELS, OUTRA SOFTMAX É USADA EM CIMA DESSE LOGITS (essa saída não é usada em models)
            Layers.append(tf.nn.softmax(Layers[-1], name=name + '_softmax'))

            return Layers


    def build_Dense_Domain_Classifier(self, input_data, name = "Domain_Classifier_Arch"):
        Layers = []
        num_filters = input_data.get_shape().as_list()[3]
        with tf.compat.v1.variable_scope(name):
            for i in range(3):
                if i == 0:
                    Layers.append(self.general_conv2d(input_data, num_filters/(2**i), 3, stride=1, padding='SAME', activation_function='leakyrelu', do_norm=True, name=name + '_conv2d_' + str(i)))
                else:
                    Layers.append(self.general_conv2d(Layers[-1], num_filters/(2**i), 3, stride=1, padding='SAME', activation_function='leakyrelu', do_norm=True, name=name + '_conv2d_' + str(i)))

            Layers.append(self.general_conv2d(Layers[-1], self.args.num_domains, 1, stride=1, padding='SAME', activation_function='None', do_norm=False, name=name + '_conv2d_' + str(i + 1)))
            Layers.append(tf.nn.softmax(Layers[-1], name=name + '_softmax'))

            return Layers

    def general_conv2d(self, input_data, filters=64,  kernel_size=7, stride=1, stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="conv2d"):
        with tf.compat.v1.variable_scope(name):
            conv = tf.compat.v1.layers.conv2d(
                input_data, filters, kernel_size, stride, padding, activation=None, kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal)

            if do_norm:
                conv = tf.compat.v1.layers.batch_normalization(conv, momentum=0.9)

            if activation_function == "relu":
                conv = tf.nn.relu(conv, name='relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name='elu')

            return conv

    def general_dense(self, input_data, units=1024, activation_function="relu", use_bias=True, kernel_initializer=None,
                      bias_initializer=tf.compat.v1.zeros_initializer(), kernel_regularizer=None,bias_regularizer=None, activity_regularizer=None,
                      kernel_constraint=None, bias_constraint=None, trainable=True, name='dense'):

        with tf.compat.v1.variable_scope(name):
            dense = tf.compat.v1.layers.dense(input_data, units, activation=None)

            if activation_function == "relu":
                dense = tf.nn.relu(dense, name='relu')
            if activation_function == "leakyrelu":
                dense = tf.nn.leaky_relu(dense, alpha=relu_factor)
            if activation_function == "elu":
                dense = tf.nn.elu(dense, name='elu')

            return dense

class SegNet():
    def __init__(self, args):
        super(SegNet, self).__init__()
        self.args = args

    def n_enc_block(self, inputs, n, k, name):
        h = inputs
        with tf.compat.v1.variable_scope(name):
            for i in range(n):
                h = tf.compat.v1.layers.conv2d(h, k, 3, 1, padding="SAME", kernel_initializer = tf.compat.v1.initializers.random_uniform(), activation=None)
                h = tf.compat.v1.layers.batch_normalization(h, momentum=0.9)
                h = tf.nn.relu(h, name='relu')

            h = tf.compat.v1.layers.max_pooling2d(h, 2, 2, name=name + '_maxpooling_' + str(i))
        return h

    def n_dec_block(self, inputs, n, k, name):
        h = inputs
        with tf.compat.v1.variable_scope(name):
            h = tf.keras.layers.UpSampling2D((2,2), name='unpool')(h)
            for i in range(n):
                if i == n:
                    h = tf.compat.v1.layers.conv2d(h, k/2, 3, 1, padding="SAME", activation=None)
                else:
                    h = tf.compat.v1.layers.conv2d(h, k, 3, 1, padding="SAME", activation=None)

                h = tf.compat.v1.layers.batch_normalization(h, momentum=0.9)
                h = tf.nn.relu(h, name='relu')
        return h

    def Build_SegNet_Encoder(self, input_data, ns, ks, name='SegNet_encoder'):
        Layers = []
        with tf.compat.v1.variable_scope(name):
            for i in range(self.args.encoder_blocks + 1):
                if i == 0:
                    Layers.append(input_data)
                else:
                    Layers.append(self.n_enc_block(Layers[-1], n = ns[i - 1], k = ks[i - 1], name = 'eblock_' + str(i)))

            return Layers

    def Build_SegNet_Decoder(self, input_data, ns, ks, name = 'SegNet_decoder'):
        Layers = []
        with tf.compat.v1.variable_scope(name):
            for i in range(self.args.encoder_blocks + 1):
                if i == 0:
                    Layers.append(input_data)
                else:
                    Layers.append(self.n_dec_block(Layers[-1], n = ns[i - 1], k = ks[i - 1], name = 'dblock_' + str(i)))

            Layers.append(tf.compat.v1.layers.conv2d(Layers[-1], self.args.num_classes, 1, 1, 'SAME', activation=None))
            Layers.append(tf.nn.softmax(Layers[-1], name=name + '_softmax'))
            return Layers

class DeepLabV3Plus():
    def __init__(self, args):
        super(DeepLabV3Plus, self).__init__()
        self.args = args

    def build_DeepLab_Encoder(self, X, name = "DeepLab_Encoder_Arch"):

        """
        Generator for DeepLab v3 plus models.

        Args:
        num_classes: The number of possible classes for image classification.
        aspp_rates: The ASPP rates. default value is 6,12,18
        base_architecture: The architecture of base Resnet building block.
        pre_trained_model: The path to the directory that contains pre-trained models.
        batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
        data_format: The input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
            Only 'channels_last' is supported currently.

        Returns:
        The model function that takes in `inputs` and `is_training` and
        returns the output tensor of the DeepLab v3 model.
        """
        print('-------------------------------------')
        print('Initializing DeepLab V3+ Architecture')
        print('-------------------------------------')
        print('Input data shape:',X.shape)

        if self.args.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            X = tf.transpose(X, [0, 3, 1, 2])

        print('Building backbone architecture...')

        if 'ResNetV1' in self.args.backbone:
            backbone = ResNetV1(self.args)
            Encoder_Layers = backbone.build_Encoder_Layers(X, name = name)
            low_Level_Features =  tf.compat.v1.layers.conv2d(Encoder_Layers[6], 48, 1, 1, padding = 'SAME', activation=None, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            Encoder_Layers.append(self.atrous_Spatial_Pyramid_Pooling(Encoder_Layers[-1], self.args.aspp_rates, self.args.bn_decay, True))

            return Encoder_Layers, low_Level_Features

        if 'ResNetV2' in self.args.backbone:
            backbone = ResNetV2(self.args)
            Encoder_Layers = backbone.build_Encoder_Layers(X, name = name)
            low_Level_Features =  tf.compat.v1.layers.conv2d(Encoder_Layers[6], 48, 1, 1, padding = 'SAME', activation=None, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            Encoder_Layers.append(self.atrous_Spatial_Pyramid_Pooling(Encoder_Layers[-1], self.args.aspp_rates, self.args.bn_decay, True))

            return Encoder_Layers, low_Level_Features

        elif self.args.backbone == 'mobile_net':
            backbone = MobileNet(self.args)
            Encoder_Layers = backbone.build_Encoder_Layers(X, name = name)
            Encoder_Layers.append(self.atrous_Spatial_Pyramid_Pooling(Encoder_Layers[-1], self.args.aspp_rates, self.args.bn_decay, True))
            return Encoder_Layers, None

        elif self.args.backbone == 'xception':
            backbone = Xception(self.args)
            Encoder_Layers = backbone.build_Encoder_Layers(X, name = name)
            low_Level_Features =  tf.compat.v1.layers.conv2d(Encoder_Layers[14], 48, 1, 1, padding = 'SAME', activation=None, kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal)
            Encoder_Layers.append(self.atrous_Spatial_Pyramid_Pooling(Encoder_Layers[-1], self.args.aspp_rates, self.args.bn_decay, True))
            return Encoder_Layers, low_Level_Features

    def build_DeepLab_Decoder(self, X, low_Level_Features, name = "DeepLab_Decoder_Arch"):

        Layers = []
        Layers.append(X)
        with tf.compat.v1.variable_scope(name):
            if low_Level_Features is not None:
                with tf.compat.v1.variable_scope("upsampling_logits"):
                    low_level_features_size = low_Level_Features.get_shape().as_list()[1:3]
                    Layers.append(tf.image.resize(Layers[-1], low_level_features_size, name = "upsample_1", method=tf.image.ResizeMethod.BILINEAR))
                    Layers.append(tf.concat([Layers[-1], low_Level_Features],axis = 3, name = "concat_1"))

            Layers.append(slim.conv2d(Layers[-1], 256, 3, stride = 1, scope = 'convd_3x3_1'))
            Layers.append(slim.conv2d(Layers[-1], 256, 3, stride = 1, scope = 'convd_3x3_2'))
            Layers.append(slim.conv2d(Layers[-1], self.args.num_classes,  1, stride = 1, activation_fn = None, normalizer_fn = None, scope = 'convd_3x3_3'))
                
            inputs_size = [self.args.patches_dimension, self.args.patches_dimension]
            Layers.append(tf.image.resize(Layers[-1], inputs_size, name='upsample_2', method=tf.image.ResizeMethod.BILINEAR))
            Layers.append(tf.nn.softmax(Layers[-1], name = name + '_softmax'))

            return Layers

    def atrous_Spatial_Pyramid_Pooling(self, inputs, aspp_rates, batch_norm_decay, is_training, depth=256):
        """Atrous Spatial Pyramid Pooling.

        Args:
        inputs: A tensor of size [batch, height, width, channels].
        aspp_rates: The ASPP rates for atrous convolution.
        batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
        is_training: A boolean denoting whether the input is for training.
        depth: The depth of the ResNet unit output.

        Returns:
        The atrous spatial pyramid pooling output.
        """
        with tf.compat.v1.variable_scope("aspp"):

            atrous_rates = aspp_rates or [6, 12, 18]

            with slim.arg_scope([slim.layers.batch_norm],decay=batch_norm_decay):
                # with arg_scope([layers.batch_norm], is_training=is_training):
                    inputs_size = tf.shape(inputs)[1:3]

                    batch_norm_params = {
                        'decay': 0.997,
                        'epsilon': 1e-5,
                        'scale': True,
                        'updates_collections': slim.ops.GraphKeys.UPDATE_OPS,
                    }
                    # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
                    # the rates are doubled when output stride = 8.
                    with tf.compat.v1.variable_scope("atrous_pyramid"):
                        conv_1x1 = slim.layers.conv2d(inputs, depth, 1, stride=1, activation_fn = tf.nn.relu,normalizer_fn=slim.layers.batch_norm,weights_regularizer=slim.regularizers.l2_regularizer(0.0001),weights_initializer=slim.initializers.variance_scaling_initializer(),normalizer_params=batch_norm_params, scope="conv_1x1")
                        conv_3x3_1 = slim.layers.conv2d(inputs, depth, 3, stride=1, rate=atrous_rates[0], activation_fn = tf.nn.relu,normalizer_fn=slim.layers.batch_norm,weights_regularizer=slim.regularizers.l2_regularizer(0.0001),weights_initializer=slim.initializers.variance_scaling_initializer(),normalizer_params=batch_norm_params, scope='conv_3x3_1')
                        conv_3x3_2 = slim.layers.conv2d(inputs, depth, 3, stride=1, rate=atrous_rates[1], activation_fn = tf.nn.relu,normalizer_fn=slim.layers.batch_norm,weights_regularizer=slim.regularizers.l2_regularizer(0.0001),weights_initializer=slim.initializers.variance_scaling_initializer(),normalizer_params=batch_norm_params, scope='conv_3x3_2')
                        conv_3x3_3 = slim.layers.conv2d(inputs, depth, 3, stride=1, rate=atrous_rates[2], activation_fn = tf.nn.relu,normalizer_fn=slim.layers.batch_norm,weights_regularizer=slim.regularizers.l2_regularizer(0.0001),weights_initializer=slim.initializers.variance_scaling_initializer(),normalizer_params=batch_norm_params, scope='conv_3x3_3')

                    # (b) the image-level features
                    with tf.compat.v1.variable_scope("image_level_features"):
                        # global average pooling
                        image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
                        # 1x1 convolution with 256 filters( and batch normalization)
                        image_level_features = slim.layers.conv2d(image_level_features, depth, 1, stride=1, activation_fn = tf.nn.relu, normalizer_fn=slim.layers.batch_norm,weights_regularizer=slim.regularizers.l2_regularizer(0.0001),weights_initializer=slim.initializers.variance_scaling_initializer(),normalizer_params=batch_norm_params, scope='conv_1x1')
                        
                        # bilinearly upsample features
                        image_level_features = tf.image.resize(image_level_features, inputs_size, method=tf.image.ResizeMethod.BILINEAR, name='upsample')

                    
                    net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
                    net = slim.layers.conv2d(net, depth, 1, stride=1, activation_fn = tf.nn.relu, normalizer_fn=slim.layers.batch_norm,weights_regularizer=slim.regularizers.l2_regularizer(0.0001),weights_initializer=slim.initializers.variance_scaling_initializer(),normalizer_params=batch_norm_params, scope='conv_1x1_concat')
                    

                    return net