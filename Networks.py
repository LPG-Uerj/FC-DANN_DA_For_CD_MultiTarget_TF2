import tensorflow as tf
from layers import ReshapeTensor, SlimConv2D

from BaseModels import *

class DeepLabV3Plus():
    def __init__(self, args):
        super(DeepLabV3Plus, self).__init__()
        self.args = args

    def build_DeepLab_Encoder(self, input_block):

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

        if self.args.backbone == 'xception':
            backbone = Xception(self.args)
            Encoder_Layers, low_layer = backbone.build_Encoder_Layers(input_block)
            
            #Layer that feedforward domain regressor model
            Encoder_Layers = self.atrous_Spatial_Pyramid_Pooling(Encoder_Layers, self.args.aspp_rates)

            return Encoder_Layers, low_layer
        else:
            raise Exception(f'Backbone not implemented {self.args.backbone}')

    def build_DeepLab_Decoder(self, features_shape, skip_connect_shape):
        with tf.name_scope("DeepLab_Decoder"):
            encoder_layer = tf.keras.layers.Input(shape = features_shape)
            low_Level_Features = tf.keras.layers.Input(shape = skip_connect_shape)
            
            if self.args.skip_connections:
                print('Skip_connections enabled')
                skip_connection = tf.keras.layers.Conv2D(48, 1, strides=1, padding = 'same', activation=None)(low_Level_Features)
                
                low_level_features_size = skip_connection.shape[1:3]
                layer = ReshapeTensor(low_level_features_size)(encoder_layer)

                layer = tf.keras.layers.Concatenate(axis=3)([layer,skip_connection])
                
                #layer = SlimConv2D(256, 3, stride = 1, padding='same', scope = 'convd_3x3_1')(layer)
                layer = self.general_conv2d(layer, filters=256, kernel_size=3, strides = 1, padding='same', do_norm=True, activation_function="relu")
            else:
                print('Skip_connections disabled')
                #layer = SlimConv2D(256, 3, stride = 1, padding='same', scope = 'convd_3x3_1')(encoder_layer)
                layer = self.general_conv2d(encoder_layer, filters=256, kernel_size=3, strides = 1, padding='same', do_norm=True, activation_function="relu")
            
            #layer = SlimConv2D(256, 3, stride = 1, padding='same', scope = 'convd_3x3_2')(layer)
            layer = self.general_conv2d(layer, filters=256, kernel_size=3, strides = 1, padding='same', do_norm=True, activation_function='relu')
            
            #layer = SlimConv2D(int(self.args.num_classes), 1, stride = 1, padding='same', activation_fn = None, normalizer_fn = None, scope = 'convd_3x3_3')(layer)
            layer = self.general_conv2d(layer, filters=int(self.args.num_classes), kernel_size=1, padding='same', do_norm=False, activation_function="none")


            inputs_size = [self.args.patches_dimension, self.args.patches_dimension]
            logits = ReshapeTensor(inputs_size)(layer)
            layer = tf.keras.layers.Softmax()(logits)
            
            decoder_model = tf.keras.Model(inputs = [encoder_layer,low_Level_Features], outputs = layer, name = 'deeplabv3plus_decoder')

            return decoder_model

    
    def atrous_Spatial_Pyramid_Pooling(self, inputs, aspp_rates, depth=256):
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

        inputs_size = inputs.shape[1:3]

        depth_padding = 'same'

        with tf.name_scope("atrous_pyramid"):

            conv_1x1 = self.regularized_conv2d(inputs, filters=depth, kernel_size=1, strides=1, padding=depth_padding, dilation_rate=1, activation_function='relu')
            conv_3x3_1 = self.regularized_conv2d(inputs, filters=depth, kernel_size=3, strides=1, padding=depth_padding, dilation_rate=atrous_rates[0], activation_function='relu')
            conv_3x3_2 = self.regularized_conv2d(inputs, filters=depth, kernel_size=3, strides=1, padding=depth_padding, dilation_rate=atrous_rates[1], activation_function='relu')
            conv_3x3_3 = self.regularized_conv2d(inputs,  filters=depth, kernel_size=3, strides=1, padding=depth_padding, dilation_rate=atrous_rates[2], activation_function='relu')

            #conv_1x1 = SlimConv2D(depth, 1, stride=1, scope="conv_1x1")(inputs)
            #conv_3x3_1 = SlimConv2D(depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')(inputs)
            #conv_3x3_2 = SlimConv2D(depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')(inputs)
            #conv_3x3_3 = SlimConv2D(depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')(inputs)
         
        with tf.name_scope("image_level_features"):
            image_level_features = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(inputs)
            #image_level_features = SlimConv2D(depth, 1, stride=1, padding=depth_padding, scope='conv_image_level_features')(image_level_features)
            image_level_features = self.regularized_conv2d(image_level_features, filters=depth, kernel_size=1, strides=1, padding=depth_padding, activation_function='relu')

            #bilinearly upsample features
            image_level_features = ReshapeTensor(inputs_size)(image_level_features)

        net = tf.keras.layers.Concatenate(axis=3)([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features])
        #net = SlimConv2D(depth, 1, stride=1, padding=depth_padding, scope='conv_1x1_concat')(net)
        net = self.regularized_conv2d(net, filters=depth, kernel_size=1, strides=1, padding=depth_padding, activation_function='relu')

        return net
    
    def regularized_conv2d(self, input_data, filters=256,  kernel_size=1, strides=1, padding='same', activation_function='relu', dilation_rate=1 , do_norm=True, relu_factor=0, name="conv2d"):
        
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0,mode='fan_in',distribution='truncated_normal')
        conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=None, dilation_rate=dilation_rate,kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001), kernel_initializer=initializer)(input_data)
        
        if do_norm:
            conv = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-5)(conv)
        if activation_function.casefold() == "leakyrelu":
            conv = tf.keras.layers.LeakyReLU(alpha=relu_factor)(conv)
        elif activation_function.casefold() != "none":
            conv = tf.keras.layers.Activation(activation_function)(conv)
        return conv
    
    
    def general_conv2d(self, input_data, filters=256,  kernel_size=1, strides=1, conv_type = 'conv', padding='same', activation_function='relu', dilation_rate=1 , do_norm=True, relu_factor=0, name="conv2d"):
        if conv_type == 'conv':
            conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=None, dilation_rate=dilation_rate)(input_data)
        elif conv_type == 'dep_conv':
            conv = tf.keras.layers.SeparableConv2D(filters, kernel_size, strides, padding, activation = None)(input_data)
        if do_norm:
            conv = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-5)(conv)
            #conv = tf.keras.layers.BatchNormalization()(conv)
        if activation_function.casefold() == "leakyrelu":
            conv = tf.keras.layers.LeakyReLU(alpha=relu_factor)(conv)
        elif activation_function.casefold() != "none":
            conv = tf.keras.layers.Activation(activation_function)(conv)
        return conv