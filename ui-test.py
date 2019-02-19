import kivy
kivy.require('1.10.1')

from kivy.app import App
from kivy.uix.anchorlayout import AnchorLayout
from kivy.graphics import *
from kivy.uix.widget import Widget

class FixedRatioLayout(AnchorLayout):
    def __init__(self, ratio=2.0):
        super().__init__(anchor_x='center', anchor_y='center')
        self.ratio = ratio

    def do_layout(self, *largs):
        w, h = self.size
        if h > w * self.ratio:
            w = h / self.ratio
        else:
            h = w * self.ratio

        for child in self.children:
            child.size_hint = None, None
            child.size = w, h
        super().do_layout(*largs)


class TetrisGame(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Rectangle(size=(50, 50))


class MyApp(App):
    def build(self):
        layout = FixedRatioLayout()
        layout.add_widget(TetrisGame())
        return layout


if __name__ == '__main__':
    MyApp().run()
