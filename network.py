import tensorflow as tf
import numpy as np
import tetris

batch_size = 64
width, height = 10, 20
num_rotations, num_shifts = 4, width

dtype = tf.float64


# simple topdown network for now
inputs    = tf.placeholder(shape=(None, width, height), dtype=dtype)
qualities = tf.placeholder(shape=(None, num_actions), dtype=dtype)

