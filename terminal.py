import tensorflow as tf
import numpy as np
import network
import pickle
import time
import ntetris as tetris


from asciimatics.screen import Screen


BATCH_SIZE = 2**14
LOSSES_PER_EPISODE = 2**4 * BATCH_SIZE
PENALTY_PER_LOSS = -1
EMA_FACTOR = 0.999
RANDOM_MOVE_BASE_PROBABILITY = 1
MEASUREMENT_EMA = 0.7
RANDOM_MOVE_PROBABILITY_DECAY = 0.999
BOARD_SPACING = 1
LABEL_WIDTH = 20
CUTOFF = 8
GIVE_BONUS_POINTS = False
SAVE_PATH = "./model/model"
MAX_VALUE_WIDTH = 20

def render_boards(screen, boards, labels=None, cutoff=1, offset=0):
    def to_chr(num):
        return ' ' if not num else str(num)

    for y in range(boards.shape[1]):
        for x in range(min(boards.shape[0], cutoff)):
            line = '|' + ''.join(to_chr(n) for n in boards[x, y]) + '|'
            screen.print_at(
                line,
                x=x * (tetris.COLUMNS + BOARD_SPACING + 2),
                y=y + offset,
                colour=Screen.COLOUR_WHITE
            )
    if labels:
        for y, arr in enumerate(labels):
            for x in range(min(boards.shape[0], cutoff)):
                line = f'{arr[x]:>{tetris.COLUMNS}}'
                screen.print_at(
                    line,
                    x=x * (tetris.COLUMNS + BOARD_SPACING + 2) + 1,
                    y=y + tetris.ROWS + offset,
                    colour=Screen.COLOUR_WHITE
                )

def render_state(screen, state, offset=1):
    max_label_width = max((len(l) for l, _ in state), default=0)
    value_offset = max_label_width + 2
    value_width = min(screen.width - value_offset, MAX_VALUE_WIDTH)
    for i in range(len(state)):
        label, value = state[i]
        value = str(value).replace('\n', '')
        y = tetris.ROWS + offset + i
        screen.print_at(label, x=0, y=y)
        screen.print_at(f'{value:>{value_width}}', x=value_offset, y=y)

def render_progress(screen, current, total, offset=0):
    height, width = screen.dimensions
    bar_width = width - LABEL_WIDTH - 3
    label_width = LABEL_WIDTH

    if bar_width > 5:
        progress = current * bar_width // total
        bar = '|' + '=' * progress + ' ' * (bar_width - progress) + '| '
    else:
        label_width = width
        bar = ''

    label = f'{current}/{total}'
    screen.print_at(bar + f'{label:<{label_width}}', x=0, y=height - 1 - offset)


def train(screen):
    with tf.Session() as sess:
        model = network.depths_network()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        random_move_probability = RANDOM_MOVE_BASE_PROBABILITY
        give_bonus_points = GIVE_BONUS_POINTS
        inference_mode = False
        game = tetris.tetris_batch(BATCH_SIZE)

        def bonus_points():
            depths = (game.boards[..., tetris.PADDING:-tetris.PADDING] != 0).argmax(axis=1).min(axis=1)
            density_points = (1 + (depths * tetris.COLUMNS - (game.boards[:, :-tetris.PADDING, tetris.PADDING:-tetris.PADDING] == 0).sum(axis=(1, 2))) / (tetris.COLUMNS * tetris.ROWS))
            return depths * density_points


        screen.clear()

        lost_games = 0
        iterations = 0

        average_time = 0
        average_clears = 0
        average_losses = 0
        while ...:
            iterations += BATCH_SIZE
            start_time = time.time()

            if not inference_mode:
                random_move_probability = max(0.1, random_move_probability * RANDOM_MOVE_PROBABILITY_DECAY)

            old_depths = np.copy(game.depths)
            old_tile_ids = np.copy(game.tiles)
            move = sess.run(model.output, feed_dict={
                model.depths: old_depths,
                model.tile_ids: old_tile_ids
                })

            # get target column and rotation
            best_index = np.argmax(move, axis=-1)
            # explore instead
            if not inference_mode and np.random.uniform(0, 1) < random_move_probability:
                best_index = np.random.choice(4 * tetris.COLUMNS, BATCH_SIZE, True)

            col, rot = np.unravel_index(best_index, (tetris.COLUMNS, 4))

            reward, lost = game.drop_in(col, rot)
            cleared_lines = np.sum(reward)
            reward = 2 * reward.astype(np.float64)
            if give_bonus_points:
                reward += bonus_points()

            next_move = sess.run(model.output, feed_dict={
                model.depths: game.depths,
                model.tile_ids: game.tiles
                })

            lost_game_penalty = np.where(lost, PENALTY_PER_LOSS, 0)
            update = lost_game_penalty + reward + EMA_FACTOR * np.max(next_move, axis=-1)

            batch_indices = np.arange(BATCH_SIZE)
            move_copy = np.copy(move) # XXX
            move[batch_indices, best_index] = update

            # update model
            if not inference_mode:
                sess.run(model.optimizer, feed_dict={
                    model.feedback: move,
                    model.depths: old_depths,
                    model.tile_ids: old_tile_ids
                    })

            lost_games += np.sum(lost)

            ev = screen.get_key()
            if ev == ord('p'):
                inference_mode = not inference_mode
            elif ev == ord('+'):
                random_move_probability += 0.01
            elif ev == ord('-'):
                random_move_probability -= 0.01
            elif ev == ord('b'):
                give_bonus_points = not give_bonus_points
            elif ev == ord('l'):
                saver.restore(sess, SAVE_PATH)
                game_state = pickle.load(open(SAVE_PATH + "state", "rb"))
                lost_games = game_state["games"]
                random_move_probability = game_state["probability"]
            elif ev in (ord('q'), ord('s')):
                game_state = {"games": lost_games, "probability": random_move_probability}
                pickle.dump(game_state, open(SAVE_PATH + "state", "wb"))
                saver.save(sess, SAVE_PATH)
                if ev == ord('q'):
                    return

            render_boards(screen, game.unpadded_boards, labels=[
                game.score,
                np.round(update, 2),
                np.abs(next_move).max(axis=1).round(4),
            ], cutoff=CUTOFF)

            end_time = time.time()
            average_time = MEASUREMENT_EMA * average_time + (1 - MEASUREMENT_EMA) * (end_time - start_time)
            average_clears = MEASUREMENT_EMA * average_clears + (1 - MEASUREMENT_EMA) * cleared_lines
            average_losses = MEASUREMENT_EMA * average_losses + (1 - MEASUREMENT_EMA) * np.sum(lost)

            render_state(screen, [
                ('random move probability', np.round(random_move_probability, 4)),
                ('inference mode', inference_mode),
                ('games played', lost_games),
                ('bonus points', give_bonus_points),
                ('moves per second', int(BATCH_SIZE / average_time)),
                ('cleared lines per iteration', np.round(cleared_lines / iterations, 4)),
                ('games lost per iteration', np.round(lost_games / iterations, 4)),
                ('x', np.round(move[0], 4)),
            ], offset=4)
            # render_progress(screen, lost_games)
            screen.refresh()


def main():
    Screen.wrapper(train)


if __name__ == '__main__':
    main()
