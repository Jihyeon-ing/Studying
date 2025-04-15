'''
keras version of LightTS model (unofficial)

Original GitHub: https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py
Paper link: https://arxiv.org/abs/2207.01186
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras.layers import *

def IEBlock(x, hidden_d):
    # shape of x: (#, H, W)
    num_node = x.shape[2]  # W
    x_permute = Permute((2, 1))(x)   # (#, W, H)
    temporal_proj = Dense(hidden_d)(x_permute)    # (#, W, F)
    temporal_proj = Activation('LeakyReLU')(temporal_proj)
    temporal_proj = Dense(hidden_d//4)(temporal_proj)    # (#, W, F')
    temporal_proj = Permute((2, 1))(temporal_proj)       # (#, F', W)

    channel_proj = Dense(num_node)(temporal_proj)    # (#, F', W)
    channel_proj = Permute((2, 1))(channel_proj)    # (#, W, F')
    output_proj = Dense(hidden_d)(channel_proj)     # (#, W, F)

    return Permute((2, 1))(output_proj)    # (#, F, W)

def light_ts(input_len, n_features, output_len, out_features, chunk_size):
    inputs = Input(shape=(input_len, n_features))
    num_chunks = input_len // chunk_size
    hidden_d = 32
    # continuous sampling
    x1 = Reshape((num_chunks, chunk_size, n_features))(inputs)   # (batch, T/C, C, N)
    x1 = Permute((3, 2, 1))(x1)    # (batch, N, C, T/C)
    x1 = Lambda(lambda t: tf.reshape(t, (-1, tf.shape(t)[2], tf.shape(t)[3])))(x1)   # (batch*N, C, T/C)
    x1 = IEBlock(x1, hidden_d)     # (batch*N, F, T/C)
    x1 = Dense(1)(x1)    # (batch*N, F, 1)

    # interval sampling
    x2 = Reshape((chunk_size, num_chunks, n_features))(inputs)   # (batch, C, T/C, N)
    x2 = Permute((3, 1, 2))(x2)    # (batch, N, C, T/C) 
    x2 = Lambda(lambda t: tf.reshape(t, (-1, tf.shape(t)[2], tf.shape(t)[3])))(x2)  # (batch*N, C, T/C)
    x2 = IEBlock(x2, hidden_d)    # (batch*N, F, T/C)
    x2 = Dense(1)(x2)   # (batch*N, F, 1)

    # concat
    x3 = Concatenate(axis=1)([x1, x2]) # (batch*N, 2F, 1)
    x3 = Lambda(lambda t: tf.reshape(t, (-1, n_features, tf.shape(t)[1])))(x3)   # (batch, N, 2F)
    x3 = Permute((2, 1))(x3)    # (batch, 2F, N)
    x3 = IEBlock(x3, output_len)     # (batch, output_len, N)
    out = Dense(out_features)(x3)

    model = Model(inputs=inputs, outputs=out)
    model.summary()
    return model
