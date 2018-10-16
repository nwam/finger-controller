"""
GameInput provides static methods to send inputs to a game.
GameInput is game and system dependant.

NOTE: The gesture to input mapping is game independant,
so this class may need to be restructured.

Usage:
game_input = GameInput()
game_input.enabled = True
game_input.jump()
game_input.stop_jump()
game_input.walk(direction)
game_input.stop_move()
game_input.kick()
"""
import pynput
from pynput.keyboard import Key as PKey
import enum
import time
import threading

class Direction(enum.Enum):
    LEFT = 0
    RIGHT = 1

class Key(enum.Enum):
    A = 0
    B = 1
    LEFT = 10
    RIGHT = 11
    UP = 12
    DOWN = 13

class GameInput:

    def __init__(self, tap_time=1/30, key_a='f', key_b='d', key_l=PKey.left,
            key_r=PKey.right, key_u=PKey.up, key_d=PKey.down, enabled=False):
        self.keys = {}
        self.keys[Key.A] = key_a
        self.keys[Key.B] = key_b
        self.keys[Key.LEFT] = key_l
        self.keys[Key.RIGHT] = key_r
        self.keys[Key.UP] = key_u
        self.keys[Key.DOWN] = key_d
        self.pressed = dict([(key, False) for key in self.keys.keys()])
        self.tap_time = tap_time # seconds
        self.enabled = enabled
        self.keyboard = pynput.keyboard.Controller()

    def keydown(self, key):
        if not self.pressed[key]:
            # Linux only
            self.keyboard.press(self.keys[key])
        self.pressed[key] = True

    def keyup(self, key):
        if self.pressed[key]:
            # Linux only
            self.keyboard.release(self.keys[key])
        self.pressed[key] = False

    def walk(self, direction=None):
        if direction is Direction.LEFT:
            self.keydown(Key.LEFT)
            self.keyup(Key.RIGHT)
        else:
            self.keydown(Key.RIGHT)
            self.keyup(Key.LEFT)

    def run(self, direction=None):
        self.keydown(Key.B)
        if direction is Direction.LEFT:
            self.keydown(Key.LEFT)
        else:
            self.keydown(Key.RIGHT)

    def stop_move(self):
        self.keyup(Key.LEFT)
        self.keyup(Key.RIGHT)

    def stop_run(self):
        self.keyup(Key.B)

    def jump(self):
        self.keydown(Key.A)

    def stop_jump(self):
        self.keyup(Key.A)

    def kick(self):
        self.keydown(Key.B)
        time.sleep(self.tap_time)
        self.keyup(Key.B)

    def perform(self, action):
        if not self.enabled or action is None:
            pass
        elif action == 'stand':
            self.stop_move()
        elif action == 'run' or action == 'walk':
            self.walk()
        elif action == 'jump':
            self.jump()
        elif action == 'down_jump':
            pass
        elif action == 'kick':
            self.kick()

    def do(self, action):
        threading.Thread(target=self.perform, args=(action,)).start()
