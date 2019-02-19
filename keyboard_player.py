import logic.tetris as tetris
import numpy as np


def character(num):
    return ' ' if not num else '#'


def render_board(board):
    for l in range(len(board)):
        line = ''.join(character(e) for e in board[l, :])
        print(f"|{line}|")


def get_move():
    while True:
        move = input("ENTER MOVE (W/A/S/D)").lower()
        if move in 'wasd' or not move:
            return move
        print("Illegal move!")


def main():

    board = tetris.tetris_batch(1, drop_every=2)

    while True:
        render_board(board.get_boards()[0])
        move = get_move()
        if move == 'd':
            move = np.array([1])
        elif move == 'a':
            move = np.array([0])
        elif move == 'w':
            move = np.array([2])
        else:
            move = np.array([3])

        points, lost = board.make_moves(move)
        print(points[0], ["", "GAME OVER"][lost[0]])

if __name__ == '__main__':
    main()
