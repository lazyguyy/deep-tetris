import tensorflow as tf
import numpy as np
import network
import tetris

from asciimatics.screen import Screen


NUM_EPISODES = 2**9
BATCH_SIZE = 2**8
LOSSES_PER_EPISODE = 2**4 * BATCH_SIZE
PENALTY_PER_LOSS = -100
EMA_FACTOR = 0.999
RANDOM_MOVE_BASE_PROBABILITY = 0.995
RANDOM_MOVE_PROBABILITY_DECAY = 0.9999

BOARD_SPACING = 1
LABEL_WIDTH = 20
CUTOFF = 8
GIVE_BONUS_POINTS = True

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
    for i in range(len(state)):
        label, value = state[i]
        y = tetris.ROWS + offset + i
        screen.print_at(label, x=0, y=y)
        screen.print_at(str(value), x=max_label_width + 2, y=y)

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

        random_move_probability = RANDOM_MOVE_BASE_PROBABILITY
        give_bonus_points = GIVE_BONUS_POINTS
        probability_override = False
        game = tetris.tetris_batch(BATCH_SIZE, rows=tetris.ROWS, cols=tetris.COLUMNS)

        lost_games = 0
        while ...:
            if not probability_override:
                random_move_probability = max(0.1, random_move_probability * RANDOM_MOVE_PROBABILITY_DECAY)

            old_depths = np.copy(game.depths)
            old_tile_ids = np.copy(game.tiles)
            move = sess.run(model.output, feed_dict={
                model.depths: old_depths,
                model.tile_ids: old_tile_ids
                })

            # get target column and rotation
            best_index = np.argmax(move, axis=-1)
            col, rot = np.unravel_index(best_index, (tetris.COLUMNS, 4))

            # explore instead
            if not probability_override and np.random.uniform(0, 1) < random_move_probability:
                rot = np.random.choice(4, BATCH_SIZE, True)
                col = np.random.choice(tetris.COLUMNS, BATCH_SIZE, True)

            reward, lost = game.drop_in(col, rot)
            reward = reward.astype(np.float64)
            if give_bonus_points:
                reward += game.bonus_points

            next_move = sess.run(model.output, feed_dict={
                model.depths: game.depths,
                model.tile_ids: game.tiles
                })

            lost_game_penalty = np.where(lost, PENALTY_PER_LOSS, 0)
            update = lost_game_penalty + reward + EMA_FACTOR * np.max(next_move, axis=-1)

            move[:, best_index] = update

            # update model
            sess.run(model.optimizer, feed_dict={
                model.feedback: move,
                model.depths: old_depths,
                model.tile_ids: old_tile_ids
                })

            lost_games += np.sum(lost)


            ev = screen.get_key()
            if ev in (ord('P'), ord('p')):
                probability_override = not probability_override
            elif ev == ord('+'):
                random_move_probability += 0.01
            elif ev == ord('-'):
                random_move_probability -= 0.01
            elif ev == ord('m'):
                game.GENERATE_UP_TO = min(game.GENERATE_UP_TO + 1, 7)
            elif ev == ord('n'):
                game.GENERATE_UP_TO = max(game.GENERATE_UP_TO - 1, 1)
            elif ev == ord('b'):
                give_bonus_points = not give_bonus_points
            elif ev in (ord('Q'), ord('q')):
                return

            state = [
                ('prob', np.round(random_move_probability, 4)),
                ('override', probability_override),
                ('games played', lost_games),
                ('tiles for choice', game.GENERATE_UP_TO),
                ('bonus points', give_bonus_points),
                ('max output', np.abs(next_move[:CUTOFF]).max(axis=1).round(2)),
                ('judgement', update[:CUTOFF].round(2)),
                # ('output', np.round(np.sum(move[0].reshape(tetris.COLUMNS, 4), axis=-1), 4)),
            ]
            render_boards(screen, game.get_boards(), labels=[game.score, np.round(game.bonus_points, 2)], cutoff=CUTOFF)
            render_state(screen, state, offset=2)
            # render_progress(screen, lost_games)
            screen.refresh()


def main():
    Screen.wrapper(train)


if __name__ == '__main__':
    main()
