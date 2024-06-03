# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 11:23:13 2017
Updated on Nov 14 2017
@author: Zain
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Masking

import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Activation, MaxPooling2D, Dropout, Permute, Reshape, Masking, Dense, MultiHeadAttention, LayerNormalization, Add, Input, Flatten)
from tensorflow.keras import Model


def CRNN2D(X_shape, nb_classes):
    '''
    Model used for evaluation in paper. Inspired by K. Choi model in:
    https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_crnn.py
    '''

    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))

    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0]))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1]))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    model.add(Masking(mask_value=0.0))  # Masking layer to handle varying sequence lengths

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))

    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


# Example usage
# X_shape = (None, frequency, time, channels) -> replace with actual input shape
# nb_classes = number of classes -> replace with actual number of classes
# model = CRNNTransformer(X_shape, nb_classes)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# +

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])
    
    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def CRNN2DTransformer(X_shape, nb_classes):
    '''
    Model with Transformer layers replacing GRU.
    '''
    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area
    head_size = 64
    num_heads = 4
    ff_dim = 128
    dropout = 0.1

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Input layer
    inputs = Input(shape=input_shape)
    x = BatchNormalization(axis=frequency_axis)(inputs)

    # First convolution layer specifies shape
    x = Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same', data_format="channels_last")(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0])(x)
    x = Dropout(0.1)(x)

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        x = Conv2D(nb_filters[layer + 1], kernel_size=kernel_size, padding='same')(x)
        x = Activation(activation)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = MaxPooling2D(pool_size=pool_size[layer + 1], strides=pool_size[layer + 1])(x)
        x = Dropout(0.1)(x)

    # Reshaping input for transformer layer
    x = Permute((time_axis, frequency_axis, channel_axis))(x)
    resize_shape = x.shape[2] * x.shape[3]
    x = Reshape((x.shape[1], resize_shape))(x)

    # Transformer encoder layers
    for _ in range(2):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = Flatten()(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(nb_classes, activation="softmax")(x)

    # Create model
    model = Model(inputs, outputs)
    return model



# -

def JustTransformer(X_shape, nb_classes):
    '''
    Model with only Transformer layers.
    '''
    head_size = 64
    num_heads = 4
    ff_dim = 128
    dropout = 0.1

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Input layer
    inputs = Input(shape=input_shape)
    x = BatchNormalization(axis=frequency_axis)(inputs)

    # Reshaping input for transformer layer
    x = Permute((time_axis, frequency_axis, channel_axis))(x)
    resize_shape = x.shape[2] * x.shape[3]
    x = Reshape((x.shape[1], resize_shape))(x)

    # Transformer encoder layers
    for _ in range(4):  # Increased to 4 layers for deeper transformation
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = Flatten()(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(nb_classes, activation="softmax")(x)

    # Create model
    model = Model(inputs, outputs)
    return model



""
'''
Models below this point were only pre-tested and were not presented in the paper
'''



""
def CRNN2DLarger(X_shape, nb_classes):
    '''
    Making the previous model larger and deeper
    '''
    nb_layers = 5  # number of convolutional layers
    nb_filters = [64, 128, 256, 512, 512]
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (2, 2), (2, 2), (4, 1),
                 (4, 1)]  # # size of pooling area
    # pool_size = [(4,2), (4,2), (4,1), (2,1)] this worked well

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model
    model = Sequential()
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))

    # First convolution layer
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(
        axis=channel_axis))  # Improves overfitting/underfitting
    model.add(MaxPooling2D(pool_size=pool_size[0],
                           strides=pool_size[0]))  # Max pooling
    model.add(Dropout(0.1))  # 0.2

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1]))  # Max pooling
        model.add(Dropout(0.1))  # 0.2

    # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


def CRNN2DVGG(X_shape, nb_classes):
    '''
    Based on VGG-16 Architecture
    '''
    nb_layers = 5  # number of convolutional layers
    nb_filters = [64, 128, 256, 512, 512]
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (2, 2), (2, 2), (4, 1),
                 (4, 1)]  # # size of pooling area
    # pool_size = [(4,2), (4,2), (4,1), (2,1)] this worked well

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model
    model = Sequential()
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))

    # First convolution layer
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(
        axis=channel_axis))  # Improves overfitting/underfitting

    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(
        axis=channel_axis))  # Improves overfitting/underfitting

    model.add(MaxPooling2D(pool_size=pool_size[0],
                           strides=pool_size[0]))  # Max pooling
    model.add(Dropout(0.1))  # 0.2

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting

        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting

        if nb_filters[layer + 1] != 128:
            model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                             padding='same'))
            model.add(Activation(activation))
            model.add(BatchNormalization(
                axis=channel_axis))  # Improves overfitting/underfitting

        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1]))  # Max pooling
        model.add(Dropout(0.1))  # 0.2

    # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    model.add(Masking(mask_value=0.0))  # Masking layer to handle varying sequence lengths

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


# +
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Dropout, Permute, Reshape, Masking, GRU, Dense
from tensorflow.keras.models import Model

def residual_block(x, nb_filters, kernel_size, activation='relu'):
    # Shortcut path
    shortcut = x
    
    # Residual path
    x = Conv2D(nb_filters, kernel_size=kernel_size, padding='same', data_format="channels_last")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(activation)(x)
    
    x = Conv2D(nb_filters, kernel_size=kernel_size, padding='same', data_format="channels_last")(x)
    x = BatchNormalization(axis=-1)(x)
    
    # Add shortcut to the residual path
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    
    return x

def CRNN2DResNet(X_shape, nb_classes):
    '''
    Based on ResNet Architecture
    '''
    nb_filters = [64, 128, 256, 512]
    kernel_size = (3, 3)  # Convolution kernel size
    activation = 'elu'  # Activation function to use after each layer
    pool_size = [(2, 2), (2, 2), (2, 2), (2, 2)]  # Pooling size
    
    # Shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    
    # Create the input layer
    inputs = Input(shape=input_shape)
    x = BatchNormalization(axis=-1)(inputs)
    
    # First convolutional layer
    x = Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same', data_format="channels_last")(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis=-1)(x)
    
    # Add residual blocks
    for i, filters in enumerate(nb_filters):
        for _ in range(2):  # Two residual blocks per layer
            x = residual_block(x, filters, kernel_size, activation)
        
        # Add MaxPooling after each set of residual blocks
        x = MaxPooling2D(pool_size=pool_size[i], strides=pool_size[i])(x)
        x = Dropout(0.1)(x)
    
    # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channels)
    x = Permute((2, 1, 3))(x)
    resize_shape = x.shape[2] * x.shape[3]
    x = Reshape((x.shape[1], resize_shape))(x)
    
    x = Masking(mask_value=0.0)(x)  # Masking layer to handle varying sequence lengths
    
    # Recurrent layers
    x = GRU(32, return_sequences=True)(x)
    x = GRU(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(nb_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs, outputs)
    
    return model



# -

def CRNN1D(X_shape, nb_classes):
    '''
    Based on 1D convolution
    '''

    nb_layers = 3  # number of convolutional layers
    kernel_size = 5  # convolution kernel size
    activation = 'relu'  # activation function to use after each layer
    pool_size = 2  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model
    model = Sequential()

    model.add(Permute((time_axis, frequency_axis, channel_axis),
                      input_shape=input_shape))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # First convolution layer
    model.add(Conv1D(64, kernel_size))
    model.add(Activation(activation))
    model.add(
        MaxPooling1D(pool_size=pool_size, strides=pool_size))  # Max pooling
    # model.add(Dropout(0.2))

    # Add more convolutional layers
    for _ in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv1D(128, kernel_size))
        model.add(Activation(activation))
        model.add(MaxPooling1D(pool_size=pool_size,
                               strides=pool_size))  # Max pooling

    model.add(Masking(mask_value=0.0))  # Masking layer to handle varying sequence lengths

    model.add(GRU(64, return_sequences=True))
    model.add(GRU(64, return_sequences=False))

    model.add(Dense(nb_classes))  # note sure about this
    model.add(Activation('softmax'))

    # Output layer
    return model


def RNN(X_shape, nb_classes):
    '''
    Implementing only the RNN
    '''
    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model
    model = Sequential()

    model.add(Permute((time_axis, frequency_axis, channel_axis),
                      input_shape=input_shape))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))

    model.add(Dense(nb_classes))  # note sure about this
    model.add(Activation('softmax'))

    # Output layer
    return model
