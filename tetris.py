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

def single_board_heights(board):
    return (board != 0).argmax(axis=0)

multiple_board_heights = lambda boards: np.apply_along_axis(single_board_heights, 1, boards)

def drop_single_tile(tile, heights, position):
    # print(position)
    x, y, l = position[0], position[1], len(tile)
    # print(heights[y:y + l])
    # print(single_board_heights(tile[::-1]))
    return min(heights[y:y + l] - (x + (l - single_board_heights(tile[::-1]))))

drop_multiple_tiles = np.vectorize(drop_single_tile, signature="(),(m),(2)->()")

class tetris_batch:


    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    ROTATE = 2
    DROP = 3
    IDLE = 4

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



    # 0 -> move left
    # 1 -> move right
    # 2 -> rotate
    # 3 -> drop
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

        # update position where possible
        self.positions = np.where(np.dstack([is_not_okay, is_not_okay]), self.positions, positions).squeeze(axis=0)
        self.rotations = np.where(is_not_okay, self.rotations, rotations)

        if np.any(moves == tetris_batch.DROP):
            heights = multiple_board_heights(self.boards[moves == tetris_batch.DROP])
            drop_heights = drop_multiple_tiles(
                            TILES[self.tiles[moves == tetris_batch.DROP], rotations[moves == tetris_batch.DROP] % 4],
                            heights,
                            positions[moves == tetris_batch.DROP])
            self.positions[moves == tetris_batch.DROP] += np.dstack([drop_heights, np.zeros(drop_heights.shape, dtype=np.int32)]).squeeze(axis=0)
            points, lost = self.respawn_tiles(moves == tetris_batch.DROP)


        lost = np.zeros(self.batch_size, dtype=np.bool)
        points = np.zeros(self.batch_size, dtype=np.int32)


        return points, lost

    # respawns new tiles and resets lost boards
    def respawn_tiles(self, players):
        lost = np.zeros(self.batch_size, dtype=np.bool)
        points = np.zeros(self.batch_size, dtype=np.int32)

        # fix all tiles that dropped
        self.boards[players] = put_tiles_in_boards(self.boards[players],
                                TILES[self.tiles[players], self.rotations[players] % 4],
                                self.positions[players])

        new_tiles = sum(players)

        self.tiles[players] = np.random.choice(len(TILES), new_tiles, True)
        self.positions[players] = np.zeros((new_tiles, 2), dtype=np.int32) + [0, self.offset]
        self.rotations[players] = np.zeros(new_tiles, dtype=np.int32)

        points[players] = clear_multiple_boards(self.boards[players,:-self.offset, self.offset:-self.offset])

        print(self.boards[players])

        lost[players] = test_multiple_tiles(self.boards[players],
                            TILES[self.tiles[players], self.rotations[players] % 4],
                            self.positions[players])
        prefix_sum = np.cumsum(lost)
        for i, board in enumerate(self.boards):
            if lost[i]:
                print(f"player {i} lost the game")
                board[:-self.offset, self.offset:-self.offset] = np.zeros((self.rows, self.cols))
        return points, lost

    # # this will never be used by a human and is just here for computer training
    # def drop_in(self, tiles, rot, col):


    def advance(self):
        positions = np.copy(self.positions) + [1, 0]

        is_not_okay = test_multiple_tiles(self.boards, TILES[self.tiles, self.rotations % 4], positions)

        self.positions = np.where(np.dstack([is_not_okay, is_not_okay]), self.positions, positions).squeeze(axis=0)

        lost = np.zeros(self.batch_size, dtype=np.bool)
        points = np.zeros(self.batch_size, dtype=np.int32)

        if np.any(is_not_okay):
            # spawn new tiles for each tile that dropped
            points, lost = self.respawn_tiles(is_not_okay)

        return points, lost

    def get_boards(self):
        boards = np.copy(self.boards)
        put_tiles_in_boards(boards, TILES[self.tiles, self.rotations % 4], self.positions)
        return boards#[:,:-self.offset, self.offset:-self.offset]

    def get_heights(self):
        return multiple_board_heights(self.boards[...,self.offset:-self.offset])
