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

def in_bounds(x, y, board):
    return 0 <= x < len(board) and 0 <= y < len(board[0])

def test_single_tile(board, tile, position):
    x, y, l = position[0], position[1], len(tile)
    return np.any(np.logical_and(tile, board[x:x + l, y:y + l]))

test_multiple_tiles = np.vectorize(test_single_tile, signature="(m,n),(),(2)->()")

def clear_single_board(board):
    board.flags.writeable = True
    delete = np.logical_not(np.apply_along_axis(np.all, 1, board))
    lines = np.sum(delete)
    if lines == 0:
        return 0
    board[-lines:] = board[delete]
    board[:-lines] = np.zeros(board[:-lines].shape)
    return (len(board) - lines)**2

clear_multiple_boards = np.vectorize(clear_single_board, signature="(m,n)->()")

def put_tile_in_board(board, tile, position):
    board.flags.writeable = True
    x, y, l = position[0], position[1], len(tile)
    board[x:x + l, y:y + l] += tile
    return board

put_tiles_in_boards = np.vectorize(put_tile_in_board, signature="(m,n),(),(2)->(m,n)")

class tetris_batch:


    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    ROTATE = 2
    DO_NOTHING = 3


    def __init__(self, batch_size, rows=20, cols=10):
        self.batch_size = batch_size
        # dimensions of the tetris grid
        self.rows = rows
        self.cols = cols
        self.offset = 3
        # state of all boards
        self.boards = np.ones((batch_size, rows + self.offset, cols + 2 * self.offset), dtype=np.int32)
        for board in self.boards:
            board[:-self.offset, self.offset:-self.offset] = np.zeros((rows, cols))
        # current tile for each board
        self.tiles = np.random.choice(len(TILES), batch_size, True)
        # current position of each eachs board tile
        self.positions = np.zeros((batch_size, 2), dtype=np.int32) + [0, self.offset]
        # current rotation of eachs board tile
        self.rotations = np.zeros(batch_size, dtype=np.int32)
        # game state
        self.lost = np.zeros(batch_size, dtype=np.bool)



    # 0 -> move left
    # 1 -> move right
    # 2 -> rotate
    # 4 -> do nothing
    def make_moves(self, moves):
        positions = np.copy(self.positions)
        rotations = np.copy(self.rotations)

        # make moves
        positions[moves == tetris_batch.MOVE_LEFT]  += [0, -1]
        positions[moves == tetris_batch.MOVE_RIGHT] += [0, 1]
        rotations[moves == tetris_batch.ROTATE]     += 1

        # test whether they are okay and reverse if necessary
        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, rotations % 4], positions)

        # update position if possible
        self.positions = np.where(np.dstack([is_not_okay, is_not_okay]), self.positions, positions).squeeze(axis=0)
        self.rotations = np.where(is_not_okay, self.rotations, rotations)



    def advance(self):
        positions = np.copy(self.positions) + [1, 0]

        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, self.rotations % 4], positions)

        self.positions = np.where(np.dstack([is_not_okay, is_not_okay]), self.positions, positions).squeeze(axis=0)

        lost = np.zeros(self.batch_size, dtype=np.bool)

        if any(is_not_okay):

            # fix all tiles that dropped
            self.boards[is_not_okay] = put_tiles_in_boards(self.boards[is_not_okay],
                            TILES[self.tiles[is_not_okay], self.rotations[is_not_okay] % 4],
                            self.positions[is_not_okay])

            # spawn new tiles for each tile that dropped
            new_tiles = sum(is_not_okay)
            self.tiles[is_not_okay] = np.random.choice(len(TILES), new_tiles, True)
            self.positions[is_not_okay] = np.zeros((new_tiles, 2), dtype=np.int32) + [0, self.offset]
            self.rotations[is_not_okay] = np.zeros(new_tiles, dtype=np.int32)

            lost[is_not_okay] = test_multiple_tiles(self.boards[is_not_okay],
                                    TILES[self.tiles[is_not_okay], self.rotations[is_not_okay] % 4],
                                    self.positions[is_not_okay])
            prefix_sum = np.cumsum(lost)
            for i, board in enumerate(self.boards):
                if lost[i]:
                    board[:-self.offset, self.offset:-self.offset] = np.zeros((self.rows, self.cols))

        return clear_multiple_boards(self.boards[:,:-self.offset, self.offset:-self.offset]), lost

    def get_boards(self):
        boards = np.copy(self.boards)
        put_tiles_in_boards(boards, TILES[self.tiles, self.rotations % 4], self.positions)
        return boards[:,:-self.offset, self.offset:-self.offset]
