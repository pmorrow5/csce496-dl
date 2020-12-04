# Helper Functions
import numpy as np
import tensorflow as tf
import os

# found from this stack overlow post: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
# x, y = shuffler(x, y)
def shuffler(images, label):
    randomize = np.arange(len(images))
    np.random.shuffle(randomize)
    images = images[randomize]
    label = label[randomize]
    return images, label

# function to split data off
# x, y = split_data(y, .1)
# Split off 10% from y and put it into x while remvoing
# the 10% split off into x from y
def split_data(data, proportion):
    size = data.shape[0]
    split_idx = int(proportion * size)
    return data[:split_idx], data[split_idx:]


def autoencoder_network(x, code_size=100):
    #"""This network assumes [?, 32, 32, 3] shaped input"""
    encoder_16 = downscale_block(x, 1) # [None, 16, 16, 3]
    encoder_8 = downscale_block(encoder_16, 2) # [None, 8, 8, 3]
    encoder_4 = downscale_block(encoder_8, 3) # [None, 4, 4, 3]
    flatten_dim = np.prod(encoder_4.get_shape().as_list()[1:])
    flat = tf.reshape(encoder_4, [-1, flatten_dim]) # [None, 4, 4, 3]
    code = tf.layers.dense(flat, code_size, activation=tf.nn.relu, name="code")
    hidden_decoder = tf.layers.dense(code, flatten_dim, activation=tf.nn.elu)
    decoder_4 = tf.reshape(hidden_decoder, [-1, 4, 4, 3])
    decoder_8 = upscale_block(decoder_4)
    decoder_16 = upscale_block(decoder_8)
    output = upscale_block(decoder_16) # [None, 32, 32, 3]

    return code, output

def upscale_block(x, scale=2):
    #"""transpose convolution upscale"""
    return tf.layers.conv2d_transpose(x, 3, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, i, scale=2):
    return tf.layers.conv2d(x, 3, 3, strides=scale, padding='same', name="encoder" + str(i))