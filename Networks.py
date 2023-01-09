import tensorflow as tf
from layers import ReshapeTensor
from tensorflow.keras.models import Model

#--------------------

from BaseModels import *

class DeepLabV3Plus():
    def __init__(self, args):
        super(DeepLabV3Plus, self).__init__()
        self.args = args

    def build_DeepLab_Encoder(self, input_block, name = "DeepLab_Encoder_Arch"):

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
        #print('Input data shape:',X.shape)

        '''
        if self.args.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            X = tf.transpose(X, [0, 3, 1, 2])
        '''

        print('Building backbone architecture...')

        '''
        if 'ResNetV1' in self.args.backbone:
            backbone = ResNetV1(self.args)
            Encoder_Layers = backbone.build_Encoder_Layers(X, name = name)
            low_Level_Features =  tf.keras.layers.Conv2D(Encoder_Layers[6], 48, 1, 1, padding = 'SAME', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            Encoder_Layers.append(self.atrous_Spatial_Pyramid_Pooling(Encoder_Layers[-1], self.args.aspp_rates, self.args.bn_decay, True))

            return Encoder_Layers, low_Level_Features

        if 'ResNetV2' in self.args.backbone:
            backbone = ResNetV2(self.args)
            Encoder_Layers = backbone.build_Encoder_Layers(X, name = name)
            low_Level_Features =  tf.keras.layers.Conv2D(Encoder_Layers[6], 48, 1, 1, padding = 'SAME', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            Encoder_Layers.append(self.atrous_Spatial_Pyramid_Pooling(Encoder_Layers[-1], self.args.aspp_rates, self.args.bn_decay, True))

            return Encoder_Layers, low_Level_Features

        elif self.args.backbone == 'mobile_net':
            backbone = MobileNet(self.args)
            Encoder_Layers = backbone.build_Encoder_Layers(X, name = name)
            Encoder_Layers.append(self.atrous_Spatial_Pyramid_Pooling(Encoder_Layers[-1], self.args.aspp_rates, self.args.bn_decay, True))
            return Encoder_Layers, None
        '''
        if self.args.backbone == 'xception':
            backbone = Xception(self.args)
            Encoder_Layers, low_layer = backbone.build_Encoder_Layers(input_block, name = name)
            low_Level_Features =  tf.keras.layers.Conv2D(48, 1, 1, padding = 'SAME', activation=None, kernel_initializer='glorot_uniform')(low_layer)
            Encoder_Layers = atrous_layer = self.atrous_Spatial_Pyramid_Pooling(Encoder_Layers, self.args.aspp_rates, self.args.bn_decay, True)
            return Encoder_Layers, low_Level_Features, atrous_layer

    def build_DeepLab_Decoder(self, encoder_layer, low_Level_Features, name = "DeepLab_Decoder_Arch"):
        #layer = tf.keras.layers.Input(shape = X.shape)
        layer = encoder_layer

        if low_Level_Features is not None:                
            low_level_features_size = low_Level_Features.shape[1:3]
            layer = ReshapeTensor(low_level_features_size)(layer)
            layer = tf.keras.layers.Concatenate(axis=3)([layer,low_Level_Features]) 

        layer = tf.keras.layers.Conv2D(256,1, strides = 1)(layer)
        layer = tf.keras.layers.Conv2D(256,1, strides = 1)(layer)

        layer = tf.keras.layers.Conv2D(int(self.args.num_classes),1)(layer)
        inputs_size = [self.args.patches_dimension, self.args.patches_dimension]
        logits = ReshapeTensor(inputs_size)(layer)
        layer = tf.keras.layers.Softmax()(logits)

        return layer

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
        atrous_rates = aspp_rates or [6, 12, 18]

        #dims = inputs.shape
        inputs_size = inputs.shape[1:3]

        #print("Input shape: " + str(inputs.shape))

        #print("Atrous rates:" + str(atrous_rates))

        depth_padding = 'same'

        conv_1x1 = tf.keras.layers.Conv2D(depth, 1, strides=1, padding=depth_padding, dilation_rate=1, use_bias = False)(inputs)
        conv_3x3_1 = tf.keras.layers.Conv2D(depth, 3, strides=1, padding=depth_padding, dilation_rate=atrous_rates[0], use_bias = False)(inputs)
        conv_3x3_2 = tf.keras.layers.Conv2D(depth, 3, strides=1, padding=depth_padding, dilation_rate=atrous_rates[1], use_bias = False)(inputs)
        conv_3x3_3 = tf.keras.layers.Conv2D(depth, 3, strides=1, padding=depth_padding, dilation_rate=atrous_rates[2], use_bias = False)(inputs)

        #print("conv_1x1:" + str(conv_1x1.shape))
        #print("conv_3x3_1:" + str(conv_3x3_1.shape))
        #print("conv_3x3_2:" + str(conv_3x3_2.shape))
        #print("conv_3x3_3:" + str(conv_3x3_3.shape))        

        image_level_features = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(inputs)

        image_level_features = tf.keras.layers.Conv2D(depth, 1, strides=1)(image_level_features)
        
        #bilinearly upsample features
        image_level_features = ReshapeTensor(inputs_size)(image_level_features)

        net = tf.keras.layers.Concatenate(axis=3)([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features])        
        net = tf.keras.layers.Conv2D(depth, 1, strides=1)(net)

        return net