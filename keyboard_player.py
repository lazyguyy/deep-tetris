import asyncio
import tetris
import numpy as np


def character(num):
    return ' ' if not num else str(num)


def render_board(board):
    for l in range(len(board)):
        line = ''.join(character(e) for e in board[l, :])
        print(f"|{line}|")
    print("-" * (len(board[0]) + 2))


def get_move():
    while True:
        move = input("ENTER MOVE W/A/S/D:").lower()
        if move in 'wasd' or not move:
            return move
        parts = move.split(" ")
        return parts


def main():

    batch_size = 1
    board = tetris.tetris_batch(batch_size, rows=20)

    while True:

        points, lost = np.zeros(batch_size, dtype=np.int32), [False]
        for i in range(10):
            render_board(board.get_boards()[0])
            move = get_move()
            if move == 'd':
                move = 1 * np.ones(batch_size, dtype=np.int32)
            elif move == 'a':
                move = 0 * np.ones(batch_size, dtype=np.int32)
            elif move == 'w':
                move = 2 * np.ones(batch_size, dtype=np.int32)
            elif move == 's':
                move = 3 * np.ones(batch_size, dtype=np.int32)
            elif move == '':
                move = 4 * np.ones(batch_size, dtype=np.int32)
            else:
                board.drop_in(int(move[0]) * np.ones(batch_size, dtype=np.int32), int(move[1]) * np.ones(batch_size, dtype=np.int32))
            board.make_moves(move)
            print(points[0], ["", "GAME OVER"][lost[0]])

        updated_points, lost = board.advance()
        points += updated_points

if __name__ == '__main__':
    main()
