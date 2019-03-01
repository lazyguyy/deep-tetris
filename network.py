import tensorflow as tf
import ntetris as tetris

dtype = tf.float64

COLUMN_OFFSET = tetris.PADDING
DROPPABLE_COLUMNS = tetris.COLUMNS + COLUMN_OFFSET

# def _make_network(depths, tile_id):
#     one_hot_tile_id = tf.one_hot(tile_id, tetris.NUM_TILES, dtype=dtype)
#     # normalized_depths = tf.layers.batch_normalization(depths, training=True)
#     normalized_depths = depths / tetris.ROWS - 0.5

#     input_layer = tf.concat([one_hot_tile_id, normalized_depths], axis=-1)

#     hidden_layer = tf.layers.dense(input_layer, 128, activation=tf.nn.relu, use_bias=True)
#     normalize_hidden = tf.layers.batch_normalization(hidden_layer, training=True)
#     hidden_with_depths = tf.concat([normalize_hidden, depths / tetris.ROWS - 0.5], axis=-1)
#     hidden_with_depths = tf.layers.dense(hidden_with_depths, 128, activation=tf.nn.relu, use_bias=True)

#     output = tf.layers.dense(hidden_with_depths, 4 * tetris.COLUMNS, use_bias=False)
#     return output

def _make_depths_network(depths, tile_id):
    one_hot_tile_id = tf.one_hot(tile_id, tetris.NUM_TILES, dtype=dtype)
    normalized_depths = depths / tetris.ROWS - 0.5
    relative_depths = depths - tf.reduce_max(depths, axis=1, keep_dims=True)
    concat_depths = tf.concat([normalized_depths, relative_depths], axis=-1)

    hidden_layer = tf.layers.dense(concat_depths, 128, activation=tf.nn.relu, use_bias=True)
    hidden_layer = tf.concat([hidden_layer, concat_depths], axis=-1)
    hidden_layer = tf.layers.dense(hidden_layer, DROPPABLE_COLUMNS * 4, use_bias=True)

    return hidden_layer


def _make_conv_network(board):


    return ...


def _make_loss(output, modified_output):
    # output = (BATCH, 4 * Columns)
    loss = tf.reduce_mean(tf.square(output - modified_output))
    return loss

def _make_policy_loss(output, score, action):
    output = tf.nn.softmax(output)
    one_hot_action = tf.one_hot(action, 4 * DROPPABLE_COLUMNS)
    loss = -score * tf.log(tf.boolean_mask(output, one_hot_action) + 1e-5)
    return loss


class depths_network:

    __slots__ = 'depths', 'tile_ids', 'feedback', 'output', 'loss', 'optimizer', 'score', 'action', 'policy_loss'

    def __init__(self):
        self.depths = tf.placeholder(shape=(None, tetris.COLUMNS), dtype=dtype, name="depths")
        self.tile_ids = tf.placeholder(shape=(None,), dtype=tf.int32, name="tile_ids")
        self.feedback = tf.placeholder(shape=(None, DROPPABLE_COLUMNS * 4), dtype=dtype, name="modified_output")
        self.score = tf.placeholder(shape=(None,), dtype=dtype, name="score")
        self.action = tf.placeholder(shape=(None,), dtype=tf.int32, name="action")

        self.output = _make_depths_network(self.depths, self.tile_ids)

        self.loss = _make_loss(self.output, self.feedback)
        self.policy_loss = _make_policy_loss(self.output, self.score, self.action)
        # self.optimizer = tf.train.MomentumOptimizer(1e-2, 0.9).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.policy_loss)

