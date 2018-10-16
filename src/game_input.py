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

class Direction(enum.Enum):
    LEFT = 0
    RIGHT = 1

class GameInput:

    def __init__(self, tap_time=1/30, key_a='f', key_b='d', key_l='Left',
            key_r='Right', key_u='Up', key_d='Down'):
        self.key_a = key_a
        self.key_b = key_b
        self.key_l = key_l
        self.key_r = key_r
        self.key_u = key_u
        self.key_d = key_d

        self.tap_time = tap_time # seconds

        self.enabled = False

    @staticmethod
    def keydown(key):
        subprocess.run(['xte', 'keydown {}'.format(key)]) # Linux only

    @staticmethod
    def keyup(key):
        subprocess.run(['xte', 'keyup {}'.format(key)]) # Linux only

    def walk(self, direction=None):
        if direction is Direction.LEFT:
            GameInput.keydown(self.key_l)
            GameInput.keyup(self.key_r)
        else:
            GameInput.keydown(self.key_r)
            GameInput.keyup(self.key_l)

    def run(self, direction=None):
        GameInput.keydown(self.key_b)
        if direction is Direction.LEFT:
            GameInput.keydown(self.key_l)
        else:
            GameInput.keydown(self.key_r)

    def stop_move(self):
        GameInput.keyup(self.key_l)
        GameInput.keyup(self.key_r)

    def stop_run(self):
        GameInput.keyup(self.key_b)

    def jump(self):
        GameInput.keydown(self.key_a)

    def stop_jump(self):
        GameInput.keyup(self.key_a)

    def kick(self):
        GameInput.keydown(self.key_b)
        time.sleep(self.tap_time)
        GameInput.keyup(self.key_b)

    def do(self, action):
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
