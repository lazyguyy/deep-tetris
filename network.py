import tensorflow as tf
import numpy as np
import tetris
import tqdm

from keyboard_player import render_board


NUM_TILES = len(tetris.TILES)
NUM_EPISODES = 2**9
BATCH_SIZE = 2**8
LOSSES_PER_EPISODE = 2**4 * BATCH_SIZE
PENALTY_PER_LOSS = 30
WIDTH, HEIGHT = 10, 20
EMA_FACTOR = 0.999
RANDOM_MOVE_PROBABILITY = 0.995
DECAY = 0.999

dtype = tf.float64

def make_network(depths, tile_id):
    one_hot_tile_id = tf.one_hot(tile_id, NUM_TILES, dtype=dtype)

    depths = tf.layers.batch_normalization(depths, training=True)
    input_layer = tf.concat([one_hot_tile_id, depths], axis=-1)
    # input_layer = tf.layers.dense(input_layer, 64, activation=tf.nn.relu, use_bias=bias)

    output = tf.layers.dense(input_layer, 4 * WIDTH, use_bias=False)

    output = tf.nn.softmax(output)

    return output

def make_loss(output, modified_output):
    loss = tf.reduce_mean(tf.square(output - modified_output))
    return loss


class depths_network:

    def __init__(self):
        self.depths = tf.placeholder(shape=(None, WIDTH), dtype=dtype, name="depths")
        self.tile_ids = tf.placeholder(shape=(None,), dtype=tf.int32, name="tile_ids")
        self.modified_output = tf.placeholder(shape=(None, WIDTH * 4), dtype=dtype, name="modified_output")

        self.output = make_network(self.depths, self.tile_ids)

        self.loss = make_loss(self.output, self.modified_output)
        self.optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(self.loss)

    def train(self, num_episodes=NUM_EPISODES):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            placed_tiles = 0
            for episode in range(num_episodes):
                game = tetris.tetris_batch(BATCH_SIZE, rows=20)

                lost_games = 0
                # with tqdm.tqdm(total=LOSSES_PER_EPISODE) as pbar:
                while lost_games < LOSSES_PER_EPISODE:
                    random_threshold = RANDOM_MOVE_PROBABILITY  * DECAY ** (placed_tiles // 10)
                    placed_tiles += 1
                    render_board(game.get_boards()[:4])
                    print(game.score[:4])
                    old_depths = np.copy(game.depths)
                    old_tile_ids = np.copy(game.tiles)
                    move = sess.run(self.output, feed_dict={
                        self.depths: old_depths,
                        self.tile_ids: old_tile_ids
                        })
                    print("Random bound, iterations: ", max(0.1, round(random_threshold, 5)), placed_tiles)
                    print("Depths:", old_depths[0])
                    print("Tile Ids:", old_tile_ids[0])
                    print("Output:", np.round(np.sum(move[0].reshape(WIDTH, 4), axis=-1), 5))

                    best_index = np.argmax(move, axis=-1)
                    col, rot = np.unravel_index(best_index, (WIDTH, 4))

                    if np.random.uniform(0, 1) < max(0.1, random_threshold):
                        rot = np.random.choice(4, BATCH_SIZE, True)
                        col = np.random.choice(WIDTH, BATCH_SIZE, True)

                    reward, lost = game.drop_in(col, rot)
                    # print(reward)

                    next_move = sess.run(self.output, feed_dict={
                        self.depths: game.depths,
                        self.tile_ids: game.tiles
                        })

                    lost_game_penalty = np.where(lost, np.zeros(BATCH_SIZE), -PENALTY_PER_LOSS * np.ones(BATCH_SIZE))

                    move[:, best_index] = lost_game_penalty + reward + EMA_FACTOR * np.max(next_move, axis=-1)

                    sess.run(self.optimizer, feed_dict={
                        self.modified_output: move,
                        self.depths: old_depths,
                        self.tile_ids: old_tile_ids
                        })

                    lost_games += np.sum(lost)

    def next_move(self, depths, tile_ids):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            rotation_quality, column_quality = sess.run((self.output_rotation, self.output_column), feed_dict={
                self.depths: depths,
                self.tile_ids: tile_ids
            })
        return np.argmax(rotation_quality, axis=-1), np.argmax(column_quality, axis=-1)

if __name__ == "__main__":
    network = depths_network()
    network.train()
