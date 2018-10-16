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
import subprocess
import enum
import time
import threading

class Direction(enum.Enum):
    LEFT = 0
    RIGHT = 1

class Keys(enum.Enum):
    A = 0
    B = 1
    LEFT = 10
    RIGHT = 11
    UP = 12
    DOWN = 13

class GameInput:

    def __init__(self, tap_time=1/30, key_a='f', key_b='d', key_l='Left',
            key_r='Right', key_u='Up', key_d='Down', enabled=False):
        self.keys = {}
        self.keys[Keys.A] = key_a
        self.keys[Keys.B] = key_b
        self.keys[Keys.LEFT] = key_l
        self.keys[Keys.RIGHT] = key_r
        self.keys[Keys.UP] = key_u
        self.keys[Keys.DOWN] = key_d
        self.states = dict([(key, False) for key in self.keys.keys()])
        self.tap_time = tap_time # seconds
        self.enabled = enabled

    def keydown(self, key):
        if not self.states[key]:
            # Linux only
            subprocess.run(['xte', 'keydown {}'.format(self.keys[key])])
        self.states[key] = True

    def keyup(self, key):
        if self.states[key]:
            # Linux only
            subprocess.run(['xte', 'keyup {}'.format(self.keys[key])])
        self.states[key] = False

    def walk(self, direction=None):
        if direction is Direction.LEFT:
            self.keydown(Keys.LEFT)
            self.keyup(Keys.RIGHT)
        else:
            self.keydown(Keys.RIGHT)
            self.keyup(Keys.LEFT)

    def run(self, direction=None):
        self.keydown(Keys.B)
        if direction is Direction.LEFT:
            self.keydown(Keys.LEFT)
        else:
            self.keydown(Keys.RIGHT)

    def stop_move(self):
        self.keyup(Keys.LEFT)
        self.keyup(Keys.RIGHT)

    def stop_run(self):
        self.keyup(Keys.B)

    def jump(self):
        self.keydown(Keys.A)

    def stop_jump(self):
        self.keyup(Keys.A)

    def kick(self):
        self.keydown(Keys.B)
        time.sleep(self.tap_time)
        self.keyup(Keys.B)

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
