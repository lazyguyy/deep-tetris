import kivy
import os
import network
import ntetris as tetris
import numpy as np
import tensorflow as tf

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import *
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget

kivy.require('1.10.1')

LOAD_PATH = "./model/model"

COLOR_MAP = {
    1: (0.15, 0.27, 0.58),
    2: (0.36, 0.67, 0.17),
    3: (0.93, 0.49, 0.09),
    4: (0.17, 0.69, 0.6),
    5: (0.97, 0.83, 0.09),
    6: (0.87, 0.09, 0.51),
    7: (0.89, 0.15, 0.1),
}

class TetrisBoardWidget(Widget):
    def __init__(self):
        super().__init__()

        self.board = np.zeros((1, 1))
        self.tile_margin = 1
        self.size_hint = None, None

        with self.canvas:
            self.callback = Callback(self.update)

    def update_size_and_position(self):
        r, c = self.board.shape
        w, h = self.parent.size
        x, y = 0, 0
        if h / r < w / c:
            new_w = h * c / r
            x = (w - new_w) / 2
            w = new_w
        else:
            new_h = w * r / c
            y = (h - new_h) / 2
            h = new_h
        self.size = w, h
        self.pos = x, y

    def show_board(self, board):
        self.board = board
        self.update()

    def update(self, *_):
        self.canvas.clear()
        self.update_size_and_position()

        tile_size = self.height / self.board.shape[0]

        with self.canvas:
            self.callback = Callback(self.update)
            for (rev_row, column), color_id in np.ndenumerate(self.board):
                if color_id == 0:
                    continue
                row = self.board.shape[0] - 1 - rev_row
                Color(rgb=COLOR_MAP[color_id])
                x = self.x + tile_size * column + self.tile_margin
                y = self.y + tile_size * row + self.tile_margin
                size = tile_size - 2 * self.tile_margin
                Rectangle(pos=(x, y), size=(size, size))


class TetrisApp(App):
    def __init__(self):
        super().__init__()
        self.title = "Tetris"
        self.renderer = TetrisBoardWidget()

        # restore model
        self.sess = tf.Session()
        self.model = network.depths_network()
        self.sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(self.sess, LOAD_PATH)

        self.game = tetris.tetris_batch(1)

        Clock.schedule_interval(self.step, 0.1)

    def step(self, *_):
        move = self.sess.run(self.model.output, feed_dict={
            self.model.depths: self.game.depths,
            self.model.tile_ids: self.game.tiles,
            })

        best_index = np.argmax(move, axis=1)
        col, rot = np.unravel_index(best_index, (network.DROPPABLE_COLUMNS, 4))
        col -= network.COLUMN_OFFSET
        self.game.drop_in(col, rot)

        self.renderer.show_board(self.game.unpadded_boards[0])

    def on_stop(self):
        self.sess.close()

    def build(self):
        layout = BoxLayout()
        layout.add_widget(self.renderer)
        return layout


if __name__ == '__main__':
    if os.environ.get("FULLSCREEN", 1) != 0:
        Window.fullscreen = 'auto'
    TetrisApp().run()
