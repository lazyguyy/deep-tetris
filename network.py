import tensorflow as tf
import numpy as np
import tetris

def padded_conv(layer, kernel_size, activation):


batch_size = 64
num_actions = 4
width, height = 10, 20

dtype = tf.float64

inputs    = tf.placeholder(shape=(None, width, height), dtype=dtype)
qualities = tf.placeholder(shape=(None, num_actions), dtype=dtype)

