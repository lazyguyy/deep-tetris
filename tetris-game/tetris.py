import numpy as np

BASIC_TILES = [
    # ##
    # ##
    [[1, 1],
     [1, 1]],
    #  # 
    # ###
    [[0, 1, 0],
     [1, 1, 1],
     [0, 0, 0]],
    #  ##
    # ##
    [[0, 1, 1],
     [1, 1, 0],
     [0, 0, 0]],
    # ##
    #  ##
    [[1, 1, 0],
     [0, 1, 1],
     [0, 0, 0]],
    # #
    # ###
    [[1, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],
    #   #
    # ###
    [[0, 0, 1],
     [1, 1, 1],
     [0, 0, 0]],
    # ####
    [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
    ]

def generate_all_rotations(tile):
    matrix = np.asmatrix(tile, dtype=np.int32)
    return matrix, np.rot90(matrix), np.rot90(matrix, 2), np.rot90(matrix, -1)

TILES = np.array([generate_all_rotations(tile) for tile in BASIC_TILES])

def test_single_tile(board, tile, position):
    x, y = position[0] + 1, position[1] + 1
    return np.any(np.logical_and(board[x:x + len(tile), y:y + len(tile)], tile))

test_multiple_tiles = np.vectorize(test_single_tile, signature="(m,n),(o,o),(2)->()")

def clear_single_board(board):
    delete = np.apply_along_axis(np.all, 1, board)
    lines = np.sum(delete)
    board[-lines:] = board[np.logical_not(delete)]
    board[:-lines] = np.zeros(board[:-lines].shape)

clear_multiple_boards = np.vectorize(clear_single_board, signature="(m,n)->()")

def put_tile_in_board(board, tile, position):
    x, y = position[0], position[1]
    board[x:x + len(tile), y:y + len(tile)] += tile

put_tiles_in_boards = np.vectorize(put_tile_in_board, signature="(m,n),(o,o),(2)->(m,n)")

class tetris_batch:


MOVE_LEFT = 0
MOVE_RIGHT = 1
ROTATE_CW = 2
DO_NOTHING = 3


class tetris_batch():

    def __init__(self, batch_size, rows=20, cols=10, drop_every=5):
        self.batch_size = batch_size
        # dimensions of the tetris grid
        self.rows = rows
        self.cols = cols
        # state of all boards
        self.boards = np.array([np.zeros((rows, cols + 2), dtype=np.int32) for _ in range(batch_size)])
        for board in self.boards:
            board[:, 0] = np.ones(rows)
            board[:, cols + 1] = np.ones(rows)
        # views so the walls are hidden
        self.views = np.array([board[:, 1:-1] for board in self.boards])
        # current tile for each board
        self.tiles = np.random.choice(len(TILES), batch_size, True)
        # current position of each tile for each board
        self.positions = np.zeros((batch_size, 2), dtype=np.int32)
        # current rotation of each tile for each board
        self.rotations = np.zeros(batch_size, dtype=np.int32)
        # after how many moves the tile drops
        self.drop_every = drop_every
        self.current_move = 0



    # 0 -> move left
    # 1 -> move right
    # 2 -> rotate
    # 4 -> do nothing
    def make_moves(self, moves):
        positions = np.copy(self.positions)
        rotations = np.copy(self.rotations)
        
        # make moves
        self.positions[moves == MOVE_LEFT] += [0, -1]
        self.positions[moves == MOVE_RIGHT] += [0, 1]
        self.rotations[moves == ROTATE_CW] += 1
        self.current_move += 1

        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, self.rotations % 4], self.positions)
        self.current_move += 1

        # test whether they are okay and reverse if necessary
        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, self.rotations % 4], self.positions)

        self.positions = np.where(is_not_okay, self.positions, positions)
        self.rotations = np.where(is_not_okay, self.rotations, rotations)

        # move all tiles down
        if current_move % drop_every == 0:
            positions = np.copy(self.positions) + [1, 0]

            is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, self.rotations % 4], self.positions)

            self.positions = np.where(is_not_okay, self.positions, positions)

            # spawn new tiles for all that have dropped
            new_tiles = sum(is_not_okay)
            self.tiles[is_not_okay] = np.random.choice(len(TILES), new_tiles, True)
            self.positions[is_not_okay] = np.zeros((new_tiles, 2), dtype=np.int32)
            self.rotations[is_not_okay] = np.zeros(new_tiles, dtype=np.int32)

        clear_multiple_boards(self.views)


    def get_boards(self):
        boards = np.copy(self.views)
        put_tiles_in_boards(boards, TILES[self.tiles, self.rotations % 4], self.positions)
        return boards

