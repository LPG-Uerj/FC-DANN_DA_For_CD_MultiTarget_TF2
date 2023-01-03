import tensorflow as tf


class Xception():
    def __init__(self, args):
        self.args = args

    def build_Encoder_Layers(self, input_block, name = 'xception'):

        conv = self.general_conv2d(input_block, 32, 3, strides=2, conv_type = 'conv', padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_1')
        conv = self.general_conv2d(conv, 64, 3, strides=1, conv_type = 'conv',padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_2')
        tensor = conv

        conv = self.general_conv2d(conv, 128, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_3')
        conv = self.general_conv2d(conv, 128, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_4')
        conv = tf.keras.layers.MaxPool2D(2,2)(conv)        

        tensor = self.general_conv2d(tensor, 128, 1, strides=2, conv_type = 'conv',padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_5')
        conv = tf.keras.layers.Add()([tensor, conv])
        conv = tf.keras.layers.Activation(tf.nn.relu)(conv)

        conv = self.general_conv2d(conv, 256, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_6')
        conv = self.general_conv2d(conv, 256, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_7')
        conv = tf.keras.layers.MaxPool2D(2,2)(conv)

        tensor = self.general_conv2d(tensor, 256, 1, strides=2, conv_type = 'conv',padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_8')
        conv = tf.keras.layers.Add()([tensor, conv])
        conv = tf.keras.layers.Activation(tf.nn.relu)(conv)

        conv = self.general_conv2d(conv, 728, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_9')
        conv = low_layer = self.general_conv2d(conv, 728, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_10')
        #Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_2'))

        tensor = self.general_conv2d(tensor, 728, 1, strides=1, conv_type = 'conv',padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_11')
        conv = tf.keras.layers.Add()([tensor, conv])
        conv = tf.keras.layers.Activation(tf.nn.relu)(conv)

        #Middle flow
        for i in range(8):
            tensor = conv
            conv = self.general_conv2d(conv, 728, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_12_1' + str(i))
            conv = self.general_conv2d(conv, 728, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_12_2' + str(i))
            conv = self.general_conv2d(conv, 728, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_12_3' + str(i))
            conv = tf.keras.layers.Add()([tensor, conv])
        
        #Exit flow
        conv = tf.keras.layers.Activation(tf.nn.relu)(conv)
        
        tensor = self.general_conv2d(conv, 1024, 1, strides=1, conv_type = 'conv', padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_13')

        conv = self.general_conv2d(conv, 728, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_14')
        conv = self.general_conv2d(conv, 1024, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_15')
        #Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_3'))

        conv = tf.keras.layers.Add()([tensor, conv])

        conv = self.general_conv2d(conv, 1536, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_16')
        conv = self.general_conv2d(conv, 2048, 3, strides=1, conv_type = 'dep_conv',padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_17')

        return conv, low_layer

    def general_conv2d(self, input_data, filters=64,  kernel_size=7, strides=1, conv_type = 'conv', stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="conv2d"):
        if conv_type == 'conv':                
            conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=None, kernel_initializer='glorot_uniform')(input_data)
        elif conv_type == 'dep_conv':                
            conv = tf.keras.layers.SeparableConv2D(filters, kernel_size, strides, padding, activation = None, depthwise_initializer = 'glorot_uniform')(input_data)
        if do_norm:                
            conv = tf.keras.layers.BatchNormalization(momentum=0.9)(conv)
        if activation_function.casefold() == "leakyrelu":                
            conv = tf.keras.layers.LeakyReLU(alpha=relu_factor)(conv)
        elif activation_function.casefold() != "none":
            conv = tf.keras.layers.Activation(activation_function)(conv)
        return conv