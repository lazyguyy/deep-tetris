import tensorflow as tf
import numpy as np
import tetris


batch_size = 64
num_tiles = 7
WIDTH, HEIGHT = 10, 20

dtype = tf.float64

def make_network(depths, tile_id, activation=None, bias=True):
    one_hot_tile_id = tf.one_hot(tile_id, num_tiles)

    input_layer = tf.concat([one_hot_tile_id, depths], axis=-1)
    output_rotation = tf.layers.dense(input_layer, 4, activation=activation, use_bias=bias)
    output_column = tf.layers.dense(input_layer, WIDTH, activation=activation, use_bias=bias)

    return output_rotation, output_column

def main():
    depths = tf.placeholder(shape=(None, WIDTH), dtype=dtype)
    tile_id = tf.placeholder(shape=(None,), dtype=dtype)

    output_rotation, output_column = make_network(depths, tile_id, activation=tf.nn.leaky_relu)

    loss = ...
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    with tf.Session() as sess:
        ...

if __name__ == '__main__':
    main()
