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
    1: (1, 0, 0),
    2: (1, 0, 0),
    3: (1, 0, 0),
    4: (1, 0, 0),
    5: (1, 0, 0),
    6: (1, 0, 0),
    7: (1, 0, 0),
}

class TetrisGame(Widget):
    def __init__(self):
        super().__init__()

        self.board = B
        self.tile_margin = 1
        self.size_hint = None, None

        with self.canvas:
            Callback(self.update)

    def update_size(self):
        r, c = self.board.shape
        w, h = self.parent.size
        if h / r < w / c:
            w = h * c / r
        else:
            h = w * r / c
        self.size = w, h

    def update(self, instr):
        self.canvas.clear()
        self.update_size()

        tile_size = self.width / self.board.shape[0]

        with self.canvas:
            Callback(self.update)
            Color(rgb=(0, 0, 1))
            Rectangle(pos=self.pos, size=self.size)
            for (row, column), color_id in np.ndenumerate(self.board):
                if color_id == 0:
                    continue
                Color(rgb=COLOR_MAP[color_id])
                Rectangle(
                    pos=(self.x + tile_size * column + self.tile_margin, self.y + tile_size * row + self.tile_margin),
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
        return layout


if __name__ == '__main__':
    # Window.fullscreen = 'auto'
    TetrisApp().run()
