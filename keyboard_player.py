import asyncio
import tetris
import numpy as np


def character(num):
    return ' ' if not num else str(num)


def render_board(board):
    for l in range(len(board)):
        line = ''.join(character(e) for e in board[l, :])
        print(f"|{line}|")


def get_move():
    while True:
        move = input("ENTER MOVE W/A/S/D:").lower()
        if move in 'wasd' or not move:
            return move
        print("Illegal move!")


def main():

    batch_size = 512
    board = tetris.tetris_batch(batch_size)

    while True:

        points, lost = np.zeros(batch_size, dtype=np.int32), [False]
        for i in range(2):
            render_board(board.get_boards()[0])
            move = get_move()
            if move == 'd':
                move = 1 * np.ones(batch_size)
            elif move == 'a':
                move = 0 * np.ones(batch_size)
            elif move == 'w':
                move = 2 * np.ones(batch_size)
            else:
                move = 3 * np.ones(batch_size)
            board.make_moves(move)
            print(points[0], ["", "GAME OVER"][lost[0]])

        updated_points, lost = board.advance()
        points += updated_points

if __name__ == '__main__':
    main()
