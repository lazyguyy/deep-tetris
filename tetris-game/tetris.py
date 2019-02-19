import numpy as np

BASIC_TILES = [
    # ##
    # ##
    [[1, 1],
     [1, 1]],
    #  # 
    # ###
    [[0, 2, 0],
     [2, 2, 2],
     [0, 0, 0]],
    #  ##
    # ##
    [[0, 3, 3],
     [3, 3, 0],
     [0, 0, 0]],
    # ##
    #  ##
    [[4, 4, 0],
     [0, 4, 4],
     [0, 0, 0]],
    # #
    # ###
    [[5, 0, 0],
     [5, 5, 5],
     [0, 0, 0]],
    #   #
    # ###
    [[0, 0, 6],
     [6, 6, 6],
     [0, 0, 0]],
    # ####
    [[0, 0, 0, 0],
     [7, 7, 7, 7],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
    ]

def generate_all_rotations(tile):
    matrix = np.array(tile)
    return matrix, np.rot90(matrix, -1), np.rot90(matrix, 2), np.rot90(matrix, 1)

TILES = np.array([generate_all_rotations(tile) for tile in BASIC_TILES])

def test_single_tile(board, tile, position):
    x, y = position[0], position[1]
    return np.any(np.logical_and(board[x:x + len(tile), y:y + len(tile)], tile))

test_multiple_tiles = np.vectorize(test_single_tile, signature="(m,n),(),(2)->()")

def clear_single_board(board):
    delete = np.apply_along_axis(np.all, 1, board)
    lines = np.sum(delete)
    if lines == 0:
        return
    board[-lines:] = board[np.logical_not(delete)]
    board[:-lines] = np.zeros(board[:-lines].shape)

clear_multiple_boards = np.vectorize(clear_single_board, signature="(m,n)->()")

def put_tile_in_board(board, tile, position):
    board.flags.writeable = True
    x, y = position[0], position[1]
    copy_into = board[x:x + len(tile), y:y + len(tile)]
    copy_into += tile[:copy_into.shape[0], :copy_into.shape[1]]
    return board

put_tiles_in_boards = np.vectorize(put_tile_in_board, signature="(m,n),(),(2)->(m,n)")

class tetris_batch:


    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    ROTATE_CW = 2
    DO_NOTHING = 3


    def __init__(self, batch_size, rows=20, cols=10, drop_every=5):
        self.batch_size = batch_size
        # dimensions of the tetris grid
        self.rows = rows
        self.cols = cols
        # offsets
        self.row_offset = 2
        self.col_offset = 2

        # state of all boards
        self.boards = np.zeros((batch_size, rows + self.row_offset, cols + 2 * self.col_offset), dtype=np.int32)
        for board in self.boards:
            board[:, :self.col_offset]        = np.ones((rows + self.row_offset, self.col_offset), dtype=np.int32)
            board[:, cols + self.col_offset:] = np.ones((rows + self.row_offset, self.col_offset), dtype=np.int32)
            board[rows:] = np.ones((self.row_offset, cols + 2 * self.col_offset), dtype=np.int32)
        # views so the walls are hidden
        self.views = np.array([board[:-self.row_offset, self.col_offset:-self.col_offset] for board in self.boards], dtype=np.int32)
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
        positions[moves == tetris_batch.MOVE_LEFT] += [0, -1]
        positions[moves == tetris_batch.MOVE_RIGHT] += [0, 1]
        rotations[moves == tetris_batch.ROTATE_CW] += 1

        self.current_move += 1

        # test whether they are okay and reverse if necessary
        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, rotations % 4], positions + [0, self.col_offset])

        self.positions = np.where(is_not_okay, self.positions, positions)
        self.rotations = np.where(is_not_okay, self.rotations, rotations)

        # move all tiles down
        if self.current_move % self.drop_every == 0:
            positions = np.copy(self.positions) + [1, 0]

            is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, self.rotations % 4], positions + [0, self.col_offset])

            self.positions = np.where(is_not_okay, self.positions, positions)

            if not any(is_not_okay):
                return
                
            # fix all tiles that have dropped
            self.views[is_not_okay] = put_tiles_in_boards(self.views[is_not_okay],
                                                          TILES[self.tiles[is_not_okay], self.rotations[is_not_okay] % 4],
                                                          self.positions[is_not_okay])

            # spawn new tiles for all that have dropped
            new_tiles = sum(is_not_okay)
            self.tiles[is_not_okay] = np.random.choice(len(TILES), new_tiles, True)
            self.positions[is_not_okay] = np.zeros((new_tiles, 2), dtype=np.int32)
            self.rotations[is_not_okay] = np.zeros(new_tiles, dtype=np.int32)

            clear_multiple_boards(self.views)


    def get_boards(self):
        boards = np.copy(self.views)
        boards = put_tiles_in_boards(boards, TILES[self.tiles, self.rotations % 4], self.positions)
        return boards


board = tetris_batch(1, 5, 5)
print(board.get_boards())

def wrap(move):
    board.make_moves(np.array([move]))
    print(board.get_boards())