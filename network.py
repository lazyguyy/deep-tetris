import tensorflow as tf
import numpy as np
import tetris
import tqdm

from keyboard_player import render_board


NUM_TILES = len(tetris.TILES)
NUM_EPISODES = 2**5
BATCH_SIZE = 2**5
LOSSES_PER_EPISODE = 2 * BATCH_SIZE
PENALTY_PER_LOSS = 1
WIDTH, HEIGHT = 10, 20
EMA_FACTOR = 0.999
RANDOM_MOVE_PROBABILITY = 0.1

dtype = tf.float64

def make_network(depths, tile_id, activation=None, bias=True):
    one_hot_tile_id = tf.one_hot(tile_id, NUM_TILES, dtype=dtype)

    input_layer = tf.concat([one_hot_tile_id, depths], axis=-1)
    output_rotation = tf.layers.dense(input_layer, 4, activation=activation, use_bias=bias)
    output_column = tf.layers.dense(input_layer, WIDTH, activation=activation, use_bias=bias)

    return output_rotation, output_column

def make_loss(output_rotation, output_column, modified_rotation, modified_column):
    rotation_loss = tf.reduce_mean(tf.square(output_rotation - modified_rotation))
    column_loss = tf.reduce_mean(tf.square(output_column - modified_column))
    return rotation_loss + column_loss


class depths_network:

    def __init__(self):
        self.depths = tf.placeholder(shape=(None, WIDTH), dtype=dtype, name="depths")
        self.tile_id = tf.placeholder(shape=(None,), dtype=tf.int32, name="tile_ids")
        self.modified_rotation = tf.placeholder(shape=(None,), dtype=dtype, name="modified_rotations")
        self.modified_column = tf.placeholder(shape=(None,), dtype=dtype, name="modified_columns")

        self.output_rotation, self.output_column = make_network(self.depths, self.tile_id, activation=tf.nn.leaky_relu)

        self.loss = make_loss(self.output_rotation, self.output_column, self.modified_rotation, self.modified_column)
        self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

    def train(self, episodes=NUM_EPISODES):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for episode in range(episodes):

                game = tetris.tetris_batch(BATCH_SIZE)

                lost_games = 0
                while lost_games < LOSSES_PER_EPISODE:
                    render_board(game.get_boards()[0])
                    old_depths = np.copy(game.depths)
                    old_tile_ids = np.copy(game.tiles)
                    rotation_quality, column_quality = sess.run((self.output_rotation, self.output_column), feed_dict={
                        self.depths: old_depths,
                        self.tile_id: old_tile_ids
                        })

                    rot = np.argmax(rotation_quality, axis=-1)
                    col = np.argmax(column_quality, axis=-1)

                    if np.random.uniform(0, 1) < RANDOM_MOVE_PROBABILITY:
                        rot = np.random.randint(4)
                        col = np.random.randint(WIDTH)

                    reward, lost = game.drop_in(col, rot)

                    next_rotation_quality, next_column_quality = sess.run((self.output_rotation, self.output_column), feed_dict={
                        self.depths: game.depths,
                        self.tile_id: game.tiles
                        })

                    lost_game_penalty = np.where(lost, np.zeros(BATCH_SIZE), -PENALTY_PER_LOSS * np.ones(BATCH_SIZE))

                    rotation_quality[:, rot] = lost_game_penalty + reward + EMA_FACTOR * np.max(next_rotation_quality)
                    column_quality[:, col] = lost_game_penalty + reward + EMA_FACTOR * np.max(next_column_quality)

                    sess.run(self.optimizer, feed_dict={
                        self.modified_rotation: rotation_quality,
                        self.modified_column: column_quality,
                        self.depths: old_depths,
                        self.tile_id: old_tile_ids
                        })

                    lost_games += np.sum(lost)

    def next_move(self):
        pass

if __name__ == "__main__":
    network = depths_network()
    network.train()