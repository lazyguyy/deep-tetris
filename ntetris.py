import numpy as np

ROWS, COLUMNS = 20, 10

TILES = np.array([
    [[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],

    [[[0, 0, 0, 0], [7, 7, 7, 7], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 0, 7, 0], [0, 0, 7, 0], [0, 0, 7, 0], [0, 0, 7, 0]],
     [[0, 0, 0, 0], [7, 7, 7, 7], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 0, 7, 0], [0, 0, 7, 0], [0, 0, 7, 0], [0, 0, 7, 0]]],

    [[[0, 0, 0, 0], [2, 2, 2, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
     [[0, 2, 0, 0], [0, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0]],
     [[2, 0, 0, 0], [2, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 2, 2, 0], [0, 2, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]]],

    [[[0, 0, 0, 0], [3, 3, 3, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
     [[3, 3, 0, 0], [0, 3, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0]],
     [[0, 0, 3, 0], [3, 3, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 3, 0, 0], [0, 3, 0, 0], [0, 3, 3, 0], [0, 0, 0, 0]]],

    [[[4, 4, 0, 0], [0, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 4, 0, 0], [4, 4, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]],
     [[4, 4, 0, 0], [0, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 4, 0, 0], [4, 4, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]]],

    [[[0, 5, 5, 0], [5, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 0, 0, 0], [5, 5, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0]],
     [[0, 5, 5, 0], [5, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 0, 0, 0], [5, 5, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0]]],

    [[[0, 6, 0, 0], [6, 6, 6, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 6, 0, 0], [0, 6, 6, 0], [0, 6, 0, 0], [0, 0, 0, 0]],
     [[0, 0, 0, 0], [6, 6, 6, 0], [0, 6, 0, 0], [0, 0, 0, 0]],
     [[0, 6, 0, 0], [6, 6, 0, 0], [0, 6, 0, 0], [0, 0, 0, 0]]],
], dtype=np.int32)

NUM_TILES = 7# TILES.shape[0]
TILE_SIZE = TILES.shape[-1]

# get the tiles indexed by positions
def make_indices(positions):
    batch_size = positions.shape[0]
    # index upper left corner
    tile_indices = np.arange(TILE_SIZE, dtype=np.intp)
    # offset corner by tile position
    rows_offsets = positions[:, 0, np.newaxis] + tile_indices
    columns_offsets = positions[:, 1, np.newaxis] + tile_indices
    # generate indices, every '1' will be subject to broadcasting
    batch_index = np.arange(batch_size, dtype=np.intp).reshape((-1, 1, 1))
    rows_index = rows_offsets.reshape((-1, TILE_SIZE, 1))
    columns_index = columns_offsets.reshape((-1, 1, TILE_SIZE))
    # return ready-to-use tuple
    return batch_index, rows_index, columns_index


# test move for validity
def test_multiple_tiles(boards, tiles, positions):
    indices = make_indices(positions)
    # cut tile location from board
    cutouts = boards[indices]
    # check for overlap
    return np.any(np.logical_and(tiles, cutouts), axis=(1, 2))


# insert tiles into the board
def put_tiles_in_boards(boards, tiles, positions):
    indices = make_indices(positions)
    boards[indices] += tiles
    return boards


# get depth maps
def multiple_board_depths(boards):
    return (boards != 0).argmax(axis=1)


# return maximum drop depth for a given tile
def drop_depths(tiles, boards, positions):
    # get columns overlapping with tile
    tile_indices = np.arange(TILE_SIZE, dtype=np.intp)
    columns_offsets = positions[:, 1, np.newaxis] + tile_indices

    batch_index = np.arange(boards.shape[0], dtype=np.intp)[:, np.newaxis, np.newaxis]
    rows_index = np.arange(boards.shape[1], dtype=np.intp)[np.newaxis, :, np.newaxis]
    columns_index = columns_offsets[:, np.newaxis, :]
    # index board
    relevant_columns = boards[batch_index, rows_index, columns_index]
    # find downwards tile extent
    tile_extent = TILE_SIZE - np.argmax(tiles[:, ::-1, :], axis=1)
    # correct tile extent with the amount the tile has already dropped
    tile_extent += positions[:, 0, np.newaxis]
    # mask: true if field is below tile extent
    is_below_tile_extent = np.arange(relevant_columns.shape[1])[np.newaxis, :, np.newaxis] >= tile_extent[:, np.newaxis, :]
    # find first collision point below tile extent
    collisions = np.logical_and(relevant_columns != 0, is_below_tile_extent)
    collision_depths = np.argmax(collisions, axis=1)
    # find how much a tile can be dropped from its original position
    relative_collision_depths = collision_depths - tile_extent
    # filter out columns which the tile does not overlap by setting depth to max
    no_overlap = np.logical_not(np.any(tiles, axis=1))
    relevant_relative_collision_depths = np.where(no_overlap, ROWS, relative_collision_depths)
    # the drop depth is the minimum collision depth over the valid columns
    depths = np.min(relevant_relative_collision_depths, axis=1)
    return depths


# clear full rows
def clear_multiple_boards(boards):
    keep = np.logical_not(np.all(boards, axis=2))
    # count survivors
    prefix_sum = np.cumsum(keep, axis=1)
    # replace rows to remove with smaller value
    surviving_indices = np.where(keep, prefix_sum, 0)
    # sort the indices, rows to remove will be at the start
    sorted_indices = np.argsort(surviving_indices, axis=1)
    # replace rows to remove with zeros
    filled = np.where(keep[:, :, np.newaxis], boards, 0)

    # take boards in the same order as before
    boards_indices = np.arange(boards.shape[0], dtype=np.intp)[:, np.newaxis, np.newaxis]
    # select rows by new order
    rows_indices = sorted_indices[:, :, np.newaxis]
    # take columns in the same order as before
    columns_indices = np.arange(boards.shape[2], dtype=np.intp)[np.newaxis, np.newaxis, :]

    # update boards
    new_boards = filled[boards_indices, rows_indices, columns_indices]
    np.copyto(boards, new_boards)
    # return score
    return boards.shape[1] - np.sum(keep, axis=1)


MOVE_LEFT = 0
MOVE_RIGHT = 1
ROTATE = 2
DROP = 3
IDLE = 4

PADDING = TILE_SIZE - 1

class tetris_batch:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.boards = np.full(shape=(batch_size, ROWS + PADDING, COLUMNS + 2 * PADDING), fill_value=-1, dtype=np.int)
        self.boards[:, :-PADDING, PADDING:-PADDING] = 0
        self.tiles = np.empty(shape=batch_size, dtype=np.int)
        self.positions = np.zeros(shape=(batch_size, 2), dtype=np.intp)
        self.rotations = np.zeros(shape=batch_size, dtype=np.int)
        self.score = np.zeros(shape=batch_size, dtype=np.int)

        self.generate_new_tiles(np.full(batch_size, True))

    def make_moves(self, moves):
        new_positions = np.copy(self.positions)
        new_rotations = np.copy(self.rotations)

        # make moves
        new_positions[moves == MOVE_LEFT] += (0, -1)
        new_positions[moves == MOVE_RIGHT] += (0, 1)
        new_rotations[moves == ROTATE] += 1
        new_rotations[moves == ROTATE] %= 4

        # reset illegal moves
        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, new_rotations], new_positions)

        self.positions = np.where(is_not_okay[:, np.newaxis], self.positions, new_positions)
        self.rotations = np.where(is_not_okay, self.rotations, new_rotations)

        # perform drops
        drop_indices = moves == DROP
        depths = drop_depths(
            TILES[self.tiles[drop_indices], self.rotations[drop_indices]],
            self.boards[drop_indices],
            self.positions[drop_indices]
        )

        self.positions[drop_indices, 0] += depths

        # find points and test for lost boards
        points, lost = self.respawn_tiles(drop_indices)
        return points, lost

    def respawn_tiles(self, indices):
        # fix dropped tiles
        self.boards[indices] = put_tiles_in_boards(
            self.boards[indices],
            TILES[self.tiles[indices], self.rotations[indices]],
            self.positions[indices]
        )

        self.generate_new_tiles(indices)

        # clear lines, adjust score
        points = clear_multiple_boards(self.unpadded_boards)
        self.score += points

        # check whether players lost and restart the game if necessary
        lost = np.full(self.batch_size, False, dtype=np.bool)
        lost[indices] = test_multiple_tiles(
            self.boards[indices],
            TILES[self.tiles[indices], self.rotations[indices]],
            self.positions[indices]
        )
        self.reset_lost_boards(lost)
        return points, lost

    def reset_lost_boards(self, indices):
        self.boards[indices, :-PADDING, PADDING:-PADDING] = 0
        self.score[indices] = 0

    def generate_new_tiles(self, indices):
        new_tiles_count = np.sum(indices)

        self.tiles[indices] = np.random.choice(NUM_TILES, new_tiles_count, replace=True)
        self.positions[indices] = (0, PADDING)
        self.rotations[indices] = 0

    def advance(self):
        new_positions = self.positions + (1, 0)

        # test if drop is possible
        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, self.rotations % 4], new_positions)
        self.positions = np.where(is_not_okay, self.positions, new_positions)

        # spawn new tiles for each tile that dropped
        points, lost = self.respawn_tiles(is_not_okay)
        return points, lost

    def drop_in(self, col, rot):
        self.positions[:, 1] = col + PADDING
        self.rotations = rot

        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, self.rotations % 4], self.positions)
        okay = np.logical_not(is_not_okay)

        points = np.zeros(self.batch_size, dtype=np.int)
        lost = np.zeros(self.batch_size, dtype=np.bool)
        lost[is_not_okay] = True

        self.reset_lost_boards(is_not_okay)

        moves = np.full(self.batch_size, IDLE)
        moves[okay] = DROP

        valid_points, valid_lost = self.make_moves(moves)

        points[okay] = valid_points[okay]
        lost[okay] = valid_lost[okay]
        return points, lost

        # col = col + PADDING
        # max_moves = np.max(np.abs(col - self.positions[:, 1]))

        # for _ in range(max_moves):
        #     moves = np.full(self.batch_size, IDLE, dtype=np.int)
        #     moves[self.positions[:, 1] < col] = MOVE_RIGHT
        #     moves[self.positions[:, 1] > col] = MOVE_LEFT
        #     self.make_moves(moves)

        # for _ in range(3):
        #     moves = np.full(self.batch_size, IDLE, dtype=np.int)
        #     moves[self.rotations % 4 != rot % 4] = ROTATE
        #     self.make_moves(moves)

        # moves = np.full(self.batch_size, DROP, dtype=np.int)
        # points, lost = self.make_moves(moves)
        # return points, lost

    @property
    def unpadded_boards(self):
        return self.boards[:, :-PADDING, PADDING:-PADDING]

    @property
    def depths(self):
        return multiple_board_depths(self.boards[..., PADDING:-PADDING])
