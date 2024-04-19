import tensorflow as tf


class DomainDiscriminatorFullyConnected(tf.keras.Model):

    def __init__(self, units: int = 1024, num_targets: int = 2, **kwargs):
        super(DomainDiscriminatorFullyConnected, self).__init__(**kwargs)

        self.flat = tf.keras.layers.Flatten()
        
        self.dense_1 = tf.keras.layers.Dense(units = units)
        self.activ_1 = tf.keras.layers.Activation('relu')
        
        self.dense_2 = tf.keras.layers.Dense(units = units)
        self.activ_2 = tf.keras.layers.Activation('relu')
                
        self.dense_3 = tf.keras.layers.Dense(units = num_targets)     
        self.proba = tf.keras.layers.Activation('softmax')    
        
    def call(self, x):

        x = self.flat(x)
        
        x = self.dense_1(x)
        x = self.activ_1(x)

        x = self.dense_2(x)
        x = self.activ_2(x)

        x = self.dense_3(x)
        probs = self.proba(x)

        return x, probs

    def get_config(self):
        config = super(DomainDiscriminatorFullyConnected, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)

def Domain_Regressor_FullyConnected(input_shape, units: int = 1024, num_targets: int = 2):
    print("Domain_Regressor_FullyConnected -  output neurons: " + str(num_targets))

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Flatten()(inputs)
    
    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Dense(units=num_targets, activation=None)(x)
    
    #output = tf.keras.layers.Softmax()(x)

    return tf.keras.Model(inputs = inputs, outputs = x, name = 'discriminator')




    
def Domain_Regressor_Convolutional(input_shape, num_targets: int):
    print("Domain_Regressor_Convolutional - input_shape: " + str(input_shape))
    num_filters = input_shape[2]
    inputs = tf.keras.Input(shape=input_shape)
    
    layers = inputs    
    for i in range(3):            
        layers = general_conv2d(layers, num_filters/(2**i), 3, strides=1, padding='SAME', activation_function='leakyrelu', do_norm=True)
    layers = general_conv2d(layers, num_targets, 1, strides=1, padding='SAME', activation_function='None', do_norm=False)
    output = tf.keras.layers.Softmax()(layers)
    model = tf.keras.Model(inputs = inputs, outputs = [output, layers], name = 'Domain_Regressor_FullyConnected')
    return model

def general_conv2d(input_data, filters,  kernel_size, strides=1, activation_function="relu", padding="valid", do_norm=True, relu_factor=0, name="conv2d"):        
    conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=None, kernel_initializer='glorot_uniform')(input_data)
    if do_norm:
        conv = tf.keras.layers.BatchNormalization(momentum=0.9)(conv)            
    if activation_function.casefold() == "leakyrelu":                
        conv = tf.keras.layers.LeakyReLU(alpha=relu_factor)(conv)
    elif activation_function.casefold() != "none":
        conv = tf.keras.layers.Activation(activation_function)(conv)        
    return conv 