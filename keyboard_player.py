
def character(num):
    return ' ' if not num else '#'


def render_board(board):
    for l in range(len(board)):
        line = ''.join(character(e) for e in board[l, :])
        print(line)


def get_move():
    while True:
        move = input("ENTER MOVE (R/L/U/N)").lower()
        if move in 'rlnu':
            return move
        print("Illegal move!")


def main():

    board = ...

    move = get_move()
    if move == 'r':
        ...
    elif move == 'l':
        ...
    elif move == 'u':
        ...
    elif move == 'n':
        ...

    pass


if __name__ == '__main__':
    main()
