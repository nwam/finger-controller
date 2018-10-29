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
            key_r=PKey.right, key_u=PKey.up, key_d=PKey.down, enabled=False,
            jump_cooldown=0.8):
        self.keys = {}
        self.keys[Key.A] = key_a
        self.keys[Key.B] = key_b
        self.keys[Key.LEFT] = key_l
        self.keys[Key.RIGHT] = key_r
        self.keys[Key.UP] = key_u
        self.keys[Key.DOWN] = key_d
        self.pressed = dict([(key, False) for key in self.keys.keys()])

        self.direction = Direction.RIGHT
        self.enabled = enabled
        self.tap_time = tap_time # seconds
        self.jump_cooldown = jump_cooldown
        self.jump_id = 0

        self.keyboard = pynput.keyboard.Controller()
        self.threads = []

    def keydown(self, key):
        pressed = False
        if not self.pressed[key]:
            self.keyboard.press(self.keys[key])
            pressed = True
        self.pressed[key] = True
        return pressed

    def keyup(self, key):
        released = False
        if self.pressed[key]:
            self.keyboard.release(self.keys[key])
            released = True
        self.pressed[key] = False
        return released

    def walk(self, direction=Direction.RIGHT):
        self.stop_run()
        if direction is Direction.LEFT:
            self.keyup(Key.RIGHT)
            self.keydown(Key.LEFT)
        else:
            self.keyup(Key.LEFT)
            self.keydown(Key.RIGHT)

    def run(self, direction=Direction.RIGHT):
        self.keydown(Key.B)
        if direction is Direction.LEFT:
            self.keyup(Key.RIGHT)
            self.keydown(Key.LEFT)
        else:
            self.keyup(Key.LEFT)
            self.keydown(Key.RIGHT)

    def stop_move(self):
        self.keyup(Key.LEFT)
        self.keyup(Key.RIGHT)

    def stop_run(self):
        self.keyup(Key.B)

    def jump(self):
        pressed = self.keydown(Key.A)

        if pressed:
            self.jump_id = threading.get_ident()
            time.sleep(self.jump_cooldown)
            if self.jump_id == threading.get_ident():
                self.stop_jump()

    def stop_jump(self):
        self.keyup(Key.A)
        self.jump_id = threading.get_ident()

    def kick(self):
        self.keydown(Key.B)
        time.sleep(self.tap_time)
        self.keyup(Key.B)

    def duck(self):
        self.keydown(Key.DOWN)

    def perform(self, action):
        if not self.enabled or action is None:
            pass
        elif action == 'stand':
            self.stop_move()
        elif action == 'walk':
            self.walk(self.direction)
        elif action == 'run':
            self.run(self.direction)
        elif action == 'jump':
            self.jump()
        elif action == 'jumpd':
            self.stop_jump()
        elif action == 'kick':
            self.kick()
        elif action == 'duck':
            self.duck()
        elif action == 'movef':
            self.direction = Direction.RIGHT
        elif action == 'moveb':
            self.direction = Direction.LEFT

    def do(self, action):
        t = threading.Thread(target=self.perform, args=(action,))
        self.threads.append(t)
        t.start()
