import numpy as np

ROWS, COLUMNS = 20, 10

TILES = np.array([
    [[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],

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

    [[[0, 0, 0, 0], [7, 7, 7, 7], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 0, 7, 0], [0, 0, 7, 0], [0, 0, 7, 0], [0, 0, 7, 0]],
     [[0, 0, 0, 0], [7, 7, 7, 7], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 0, 7, 0], [0, 0, 7, 0], [0, 0, 7, 0], [0, 0, 7, 0]]]
], dtype=np.int32)

NUM_TILES = TILES.shape[0]
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
    # correct tile extent with the amount the dile has already dropped
    tile_extent += positions[:, 0, np.newaxis]
    # mask: true if field is below tile extent
    is_below_tile_extent = np.arange(relevant_columns.shape[1])[np.newaxis, :, np.newaxis] >= tile_extent
    # find first collision point below tile extent
    collisions = np.logical_and(relevant_columns != 0, is_below_tile_extent)
    collision_depths = np.argmax(collisions, axis=1)
    # find how much a tile can be dropped from its original position
    relative_collision_depth = collision_depths - tile_extent
    # filter out columns which the tile does not overlap by setting depth to max
    no_overlap = np.logical_not(np.any(tiles, axis=1))
    relevant_relative_collision_depth = np.where(no_overlap, ROWS, relative_collision_depth)
    # the drop depth is the minimum collision depth over the valid columns
    depths = np.min(relevant_relative_collision_depth, axis=1)
    return depths


# clear full rows
def clear_multiple_boards(boards):
    keep = np.logical_not(np.all(boards, axis=2))
    # count survivors
    prefix_sum = np.cumsum(keep, axis=1)
    # replace rows to remove with smaller value
    surviving_indices = np.where(keep, prefix_sum, -1)
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
    return 5 * (boards.shape[1] - np.sum(keep, axis=1)) ** 2


tile = TILES[4][0][np.newaxis, :, :]

ascii_board = [
    '#     ',
    '      ',
    '     #',
    '  ##  ',
    '   #  ',
]

unpadded_board = np.array([[1 if c != ' ' else 0 for c in line] for line in ascii_board])
board = np.ones((unpadded_board.shape[0] + 3, unpadded_board.shape[1] + 6), dtype=np.intp)
board[0:-3, 3:-3] = unpadded_board
board = board[np.newaxis, :, :]
print(board)
print(tile)
pos = np.array([(0, 3)])
print(drop_depths(tile, board, pos))
