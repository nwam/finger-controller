"""
GameInput provides static methods to send inputs to a game.
GameInput is game and system dependant.

NOTE: The gesture to input mapping is game independant,
so this class may need to be restructured.

Usage:
GameInput.jump()
GameInput.stop_jump()
GameInput.walk(direction)
GameInput.stop_move()
GameInput.kick()
"""
import subprocess

class GameInput:
    KEY_A = 'f'
    KEY_B = 'd'

    KEY_L = 'Left'
    KEY_R = 'Right'
    KEY_U = 'Up'
    KEY_D = 'Down'

    TAP_TIME = 1/30 # seconds

    @staticmethod
    def keydown(key):
        subprocess.run(['xte', 'keydown {}'.format(key)]) # Linux only

    @staticmethod
    def keyup(key):
        subprocess.run(['xte', 'keyup {}'.format(key)]) # Linux only


    @classmethod
    def walk(cls, direction=None):
        if direction is Direction.LEFT:
            cls.keydown(cls.KEY_L)
            cls.keyup(cls.KEY_R)
        else:
            cls.keydown(cls.KEY_R)
            cls.keyup(cls.KEY_L)

    @classmethod
    def run(cls, direction=None):
        cls.keydown(cls.KEY_B)
        if direction is Direction.LEFT:
            cls.keydown(cls.KEY_L)
        else:
            cls.keydown(cls.KEY_R)

    @classmethod
    def stop_move(cls):
        cls.keyup(cls.KEY_L)
        cls.keyup(cls.KEY_R)

    @classmethod
    def stop_run(cls):
        cls.keyup(cls.KEY_B)


    @classmethod
    def jump(cls):
        cls.keydown(cls.KEY_A)

    @classmethod
    def stop_jump(cls):
        cls.keyup(cls.KEY_A)


    @classmethod
    def kick(cls):
        cls.keydown(cls.KEY_B)
        time.sleep(TAP_TIME)
        cls.keyup(cls.KEY_B)
