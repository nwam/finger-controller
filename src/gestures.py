"""
This module translates information about the hand into gestures.

Usage:
    hand = Hand()

    while True:
        # ...
        # get info about hand
        # ...

        hand.position = position
        hand.leg_speed = leg_speed
        hand.check_cooldowns()

        if Gesture.RUN in hand.gestures:
            # Press run button
        # ...
"""
from enum import Enum
import time
import numpy as np

class Gesture(Enum):
    IDLE = 0
    RUN  = 1
    JUMP = 2
    KICK = 3

class Direction(Enum):
    RIGHT = 'R'
    LEFT = 'L'

class Hand:
    def __init__(self):
        self.run = False
        self.jump = False
        self.kick = False

        self._leg_speed = 0.0
        self.body_facing = Direction.LEFT

        self.r_direction = Direction.RIGHT
        self.screen_width = 176 # unrelated to hand, but whatever
        self.r_screen_ratio = 7/9 # how much of the screen width is forward?
        #self.r_direction_thresh = 5.0 # px/frame
        #self.r_direction_ratio = 1.5 # x must be ratio more than y
        self.r_thresh = 3.5 # avg opflow
        self.w_thresh = 1.0 # walk thresh

        self.j_thresh = 7.5 # px/frame
        self.j_ratio  = 1.5 # y must be ratio more than x
        self.j_reset_thresh = 4.0 #px/frame
        self.j_start = 0
        self.j_cooldown = 0.8 # seconds
        self._position = np.zeros(2)
        self._velocity = np.zeros(2)
        self.velocity_alpha = 0.5

        self.k_start = 0

    @property
    def leg_speed(self):
        return self._leg_speed

    @property
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity


    @leg_speed.setter
    def leg_speed(self, speed):
        self._leg_speed = speed

        if self.leg_speed > self.w_thresh:
            self.run = True
        else:
            self.run = False

    @position.setter
    def position(self, pos):
        prev_pos = self.position

        # Set velocity
        if None not in pos and None not in prev_pos:
            self.velocity = self.velocity_alpha * self.velocity + (1-self.velocity_alpha) * (prev_pos - pos)
        else:
            self.velocity = np.zeros(2)

        # Update run direction
        if None not in pos:
            h_pos = pos[0] if self.body_facing is Direction.LEFT else self.screen_width - pos[0]
            if h_pos > self.screen_width * self.r_screen_ratio:
                self.r_direction = Direction.LEFT
            else:
                self.r_direction = Direction.RIGHT

            self._position = np.array(pos)

    @velocity.setter
    def velocity(self, v):
        if v[0] is not None and v[1] is not None:
            self._velocity = np.array(v)

        # Update jump
        if abs(self.velocity[1]) > abs(self.velocity[0])*self.j_ratio:
            if self.velocity[1] > self.j_thresh:
                self.jump = True
                self.j_start = time.time()

            elif self.jump == True and self.velocity[1] < -1*self.j_reset_thresh:
                self.jump = False

#         # Update run direction
#         if abs(self.velocity[0]) > abs(self.velocity[1])*self.r_direction_ratio:
#             h_velocity = self.velocity[0]
#             if self.body_facing is Direction.RIGHT:
#                 h_velocity *= -1

#             if self.r_direction is Direction.LEFT and self.velocity[0] > self.r_direction_thresh:
#                 self.r_direction = Direction.RIGHT
#             elif self.r_direction is Direction.RIGHT and self.velocity[0] < -1*self.r_direction_thresh:
#                 self.r_direction = Direction.LEFT

    def check_cooldowns(self):
        if time.time() - self.j_start > self.j_cooldown:
            self.jump = False



    @property
    def gestures(self):
        g = []
        if self.run:
            g.append(Gesture.RUN)
        if self.jump:
            g.append(Gesture.JUMP)
        if self.kick:
            g.append(Gesture.KICK)
        return g

    def gestures_pretty(self):
        g = ''
        g += 'R' if self.run  else '0'
        g += 'J' if self.jump else '0'
        g += 'K' if self.kick else '0'
        return g

if __name__ == '__main__':
    import pdb
    hand = Hand()
    hand.j_cooldown = 5.0
    hand.position = (0, 4)
    print(hand.velocity)
    print(hand.gestures_pretty())
    print(hand.r_direction)
    hand.position = (-19, 3)
    print(hand.velocity)
    print(hand.gestures_pretty())
    print(hand.r_direction)

    print(hand.gestures_pretty())
    hand.velocity = (4, -16)
    print(hand.gestures_pretty())
