import tensorflow as tf
import ntetris as tetris

dtype = tf.float64

def _make_network(depths, tile_id):
    one_hot_tile_id = tf.one_hot(tile_id, tetris.NUM_TILES, dtype=dtype)
    normalized_depths = tf.layers.batch_normalization(depths, training=True)

    input_layer = tf.concat([one_hot_tile_id, normalized_depths], axis=-1)

    hidden_layer = tf.layers.dense(input_layer, 128, activation=tf.nn.relu, use_bias=True)
    normalize_hidden = tf.layers.batch_normalization(hidden_layer, training=True)

    output = tf.layers.dense(normalize_hidden, 4 * tetris.COLUMNS, use_bias=False)
    return output

def _make_loss(output, modified_output):
    loss = tf.reduce_mean(tf.square(output - modified_output))
    return loss


class depths_network:

    __slots__ = 'depths', 'tile_ids', 'feedback', 'output', 'loss', 'optimizer'

    def __init__(self):
        self.depths = tf.placeholder(shape=(None, tetris.COLUMNS), dtype=dtype, name="depths")
        self.tile_ids = tf.placeholder(shape=(None,), dtype=tf.int32, name="tile_ids")
        self.feedback = tf.placeholder(shape=(None, tetris.COLUMNS * 4), dtype=dtype, name="modified_output")

        self.output = _make_network(self.depths, self.tile_ids)

        self.loss = _make_loss(self.output, self.feedback)
        self.optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(self.loss)

