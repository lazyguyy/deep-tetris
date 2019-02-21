import tensorflow as tf
import numpy as np
import tetris
import tqdm


NUM_TILES = len(tetris.BASIC_TILES)
NUM_EPISODES = ...
BATCH_SIZE = 64
WIDTH, HEIGHT = 10, 20
EMA_FACTOR = 0.999
RANDOM_MOVE_PROBABILITY = 0.1

dtype = tf.float64

def make_network(depths, tile_id, activation=None, bias=True):
    one_hot_tile_id = tf.one_hot(tile_id, NUM_TILES)

    input_layer = tf.concat([one_hot_tile_id, depths], axis=-1)
    output_rotation = tf.layers.dense(input_layer, 4, activation=activation, use_bias=bias)
    output_column = tf.layers.dense(input_layer, WIDTH, activation=activation, use_bias=bias)

    return output_rotation, output_column

def make_loss(output_rotation, output_column, modified_rotation, modified_column):
    rotation_loss = tf.reduce_mean(tf.square(output_rotation - modified_rotation))
    column_loss = tf.reduce_mean(tf.square(output_column - modified_column))
    return rotation_loss + column_loss


def main():
    depths = tf.placeholder(shape=(None, WIDTH), dtype=dtype)
    tile_id = tf.placeholder(shape=(None,), dtype=dtype)
    modified_rotation = tf.placeholder(shape=(None,), dtype=dtype)
    modified_column = tf.placeholder(shape=(None,), dtype=dtype)

    output_rotation, output_column = make_network(depths, tile_id, activation=tf.nn.leaky_relu)

    loss = make_loss(output_rotation, output_column, modified_rotation, modified_column)
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    episodes = tqdm.tqdm(range(NUM_EPISODES))
    for episode in episodes:
        with tf.Session() as sess:

            # init boards here

            while ...:
                rotation_quality, column_quality = sess.run((output_rotation, output_column), feed_dict={
                    depths: ..., tile_id: ...})

                rot = np.argmax(rotation_quality)
                col = np.argmax(column_quality)

                if np.random.uniform(0, 1) < RANDOM_MOVE_PROBABILITY:
                    rot = np.random.randint(4)
                    col = np.random.randint(WIDTH)

                # execute action here

                next_rotation_quality, next_column_quality = sess.run((output_rotation, output_column), feed_dict={
                    depths: ..., tile_id: ...})

                rotation_quality[:, rot] = reward + EMA_FACTOR * np.max(next_rotation_quality)
                column_quality[:, col] = reward + EMA_FACTOR * np.max(next_column_quality)

                sess.run(optimizer, feed_dict={
                    modified_rotation: rotation_quality,
                    modified_column: column_quality,
                    depths: ...,
                    tile_id: ...})

                # update local state here


if __name__ == '__main__':
    main()
