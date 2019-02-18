import numpy as np

tiles = [
    # ##
    # ##
    np.asmatrix([[1, 1],
                 [1, 1]],
                dtype=np.int32),
    #  # 
    # ###
    np.asmatrix([[0, 1, 0],
                 [1, 1, 1],
                 [0, 0, 0]],
                dtype=np.int32),
    #  ##
    # ##
    np.asmatrix([[0, 1, 1],
                 [1, 1, 0],
                 [0, 0, 0]],
                dtype=np.int32),
    # ##
    #  ##
    np.asmatrix([[1, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]],
                dtype=np.int32),
    # #
    # ###
    np.asmatrix([[1, 0, 0],
                 [1, 1, 1],
                 [0, 0, 0]],
                dtype=np.int32),
    #   #
    # ###
    np.asmatrix([[0, 0, 1],
                 [1, 1, 1],
                 [0, 0, 0]],
                dtype=np.int32),
    # ####
    np.asmatrix([[1, 1, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                dtype=np.int32)
    ]

def test_single_tile(board, tile, position):
    print(board)
    print(tile)
    print(position)
    return np.any(np.logical_and(board[position[0]:position[0] + len(tile[0]), position[1]:position[1] + len(tile)], tile))

test_multiple_tiles = np.vectorize(test_single_tile, signature="(m,n),(o,o),(2)->()")

class tetris_batch():

    vectorized_rotate = np.vectorize(np.rot90, signature="(m,m),k->(m,m)")

    def __init__(self, batch_size, rows=20, cols=10, drop_every=5):
        self.batch_size = batch_size
        self.rows = rows
        self.cols = cols
        self.boards = np.array([np.zeros((rows, cols), dtype=np.int32) for _ in range(batch_size)])
        self.tiles = np.random.choice(tiles, batch_size, True)
        self.positions = np.array([[0,0] for _ in range(batch_size)])
        self.drop_every = drop_every
        self.current_move = 0

    # 0 -> move left
    # 1 -> move right
    # 2 -> rotate clockwise
    # 3 -> rotate counterclockwise
    # 4 -> do nothing
    def make_moves(self, moves):
        self.positions[moves == 0] += [0,-1]
        self.positions[moves == 1] += [0, 1]
        vectorized_rotate(self.tiles[moves == 2], 1)
        vectorized_rotate(self.tiles[moves == 3], -1)
        self.current_move += 1

        is_not_okay = test_multiple_tiles(self.boards, self.tiles, self.positions)

    def get_boards()

board = np.zeros((4,4), dtype=np.int32)
board[0,2] = 1
boards = [board, board]

ts = [tiles[1], tiles[2]]

ps = [[0, 0], [0, 0]]