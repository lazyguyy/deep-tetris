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

    def __init__(self, batch_size, rows=20, cols=10, drop_every=5):
        self.batch_size = batch_size
        self.rows = rows
        self.cols = cols
        self.boards = np.array([np.zeros((rows, cols), dtype=np.int32) for _ in range(batch_size)])
        self.tiles = np.random.choice(tiles, batch_size, True)
        self.positions = np.array([[0,0] for _ in range(batch_size)])
        self.drop_every = drop_every

    # 0 -> move left
    # 1 -> move right
    # 2 -> rotate clockwise
    # 3 -> rotate counterclockwise
    # 4 -> do nothing
    def make_moves(moves):
        self.positions[move == 0] += [0,-1]
        self.positions[move == 1] += [0, 1]
        


board = np.zeros((4,4), dtype=np.int32)
board[0,2] = 1
boards = [board, board]

ts = [tiles[1], tiles[2]]

ps = [[0, 0], [0, 0]]