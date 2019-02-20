import os
# os.environ["KIVY_NO_CONSOLELOG"] = "1"

import kivy
kivy.require('1.10.1')

from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import *
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.widget import Widget

import numpy as np

COLOR_MAP = {
    1: (0.15, 0.27, 0.58),
    2: (0.36, 0.67, 0.17),
    3: (0.93, 0.49, 0.09),
    4: (0.17, 0.69, 0.6),
    5: (0.97, 0.83, 0.09),
    6: (0.87, 0.09, 0.51),
    7: (0.89, 0.15, 0.1),
}

class TetrisGame(Widget):
    def __init__(self):
        super().__init__()

        self.board = B
        self.tile_margin = 1
        self.size_hint = None, None
        self.pos_hint = None, None

    def update_size(self):
        r, c = self.board.shape
        ratio = c / r

        w, h = self.parent.size
        if h > w * ratio:
            w = h / ratio
        else:
            h = w * ratio

        self.size = w, h

    def draw(self, *args):
        self.canvas.clear()
        self.update_size()

        tile_size = self.width / self.board.shape[0]

        with self.canvas:
            for (row, column), color_id in np.ndenumerate(self.board):
                if color_id == 0:
                    continue
                Color(rgb=COLOR_MAP[color_id])
                Rectangle(
                    pos=(tile_size * column + self.tile_margin, tile_size * row + self.tile_margin),
                    size=(tile_size - 2 * self.tile_margin, tile_size - 2 * self.tile_margin)
                )


B = np.zeros((20, 10))
B[1:5, 0:2] = 1
B[6:8, 1:7] = 2

class TetrisApp(App):
    def __init__(self):
        super().__init__()
        self.title = "Tetris"
        self.game = TetrisGame()

    def build(self):
        layout = AnchorLayout(anchor_x='center', anchor_y='center')
        layout.add_widget(self.game)
        self.game.draw()
        return layout


if __name__ == '__main__':
    # Window.fullscreen = 'auto'
    TetrisApp().run()
