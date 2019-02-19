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

vectorized_clear = np.vectorize(clear_single_board, signature="(m,n)->()")

def spawn_new_tile():
    return np.random.choice(TILES), [0, 0]


MOVE_LEFT = 0
MOVE_RIGHT = 1
ROTATE_CW = 2
ROTATE_CCW = 3
DO_NOTHING = 4


class tetris_batch():

    def __init__(self, batch_size, rows=20, cols=10, drop_every=5):
        self.batch_size = batch_size
        # dimensions of the tetris grid
        self.rows = rows
        self.cols = cols
        # state of all boards
        self.boards = np.array([np.zeros((rows, cols + 2), dtype=np.int32) for _ in range(batch_size)])
        for board in self.boards:
            board[:,0] = np.ones(rows)
            board[:,cols + 1] = np.ones(rows)
        # views so the walls are hidden
        self.views = np.array([board[:,1:-1] for board in self.boards])
        # current tile for each board
        self.tiles = np.random.choice(len(TILES), batch_size, True)
        # current position of each tile for each board
        self.positions = np.array([[0,0] for _ in range(batch_size)])
        # current rotation of each tile for each board
        self.rotations = np.zeros(batch_size, dtype=np.int32)
        # after how many moves the tile drops
        self.drop_every = drop_every
        self.current_move = 0



    # 0 -> move left
    # 1 -> move right
    # 2 -> rotate clockwise
    # 3 -> rotate counterclockwise
    # 4 -> do nothing
    def make_moves(self, moves):
        self.positions[moves == MOVE_LEFT] += [0, -1]
        self.positions[moves == MOVE_RIGHT] += [0, 1]
        self.rotations[moves == ROTATE_CW] += -1
        self.rotations[moves == ROTATE_CCW] += 1
        self.current_move += 1

        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, self.rotations % 4], self.positions)

