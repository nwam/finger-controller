"""
This module contains Recording and functions to create balanced sequence of
Recording objects.
"""
import random
import enum

gestures = ['stand', 'walk', 'run', 'jump', 'jumpd', 'kick', 'duck', 'movef',
        'moveb']

class CamSide(enum.Enum):
    LEFT = 0
    RIGHT = 1

class CamProps:
    def __init__(self, side):
        self.side = side

class RecMode(enum.Enum):
    BEFORE = 0
    MIDDLE = 1
    AFTER = 2

class Recording:
    """
    This class describes a snippet of a recorded video which we want to
        analyze. When recording data, instead of having the data-gatherer
        click start and stop, only one frame needs to be "recorded", and
        RecMode will manage which other frames should be kept.

    Params
        label: the frames' label for the CNN
        info: what is displayed to the user
        rec_mode: which frames we want to keep relative to frame
        n_frames: how many frames we want to keep
        frame: the "recorded" frame
    """
    def __init__(self, label, info, rec_mode=RecMode.AFTER, n_frames=7):
        self.label = label
        self.info = info
        self.rec_mode = rec_mode
        self.n_frames = n_frames
        self.frame = None

    def get_frames(self):
        if self.rec_mode == RecMode.AFTER:
            return list(range(self.frame, self.frame+self.n_frames))
        elif self.rec_mode == RecMode.MIDDLE:
            half = self.n_frames // 2
            return list(range(self.frame-half, self.frame+half+1))
        elif self.rec_mode == RecMode.BEFORE:
            return list(range(self.frame-self.n_frames, self.frame))

    def __str__(self):
        return '{}'.format(self.info)

    def __repr__(self):
        return '{}'.format(self.info)

def generate_sequence(positions=['front', 'center', 'back']):
    n = len(positions)
    seq = []

    transitions = {}
    remain_transitions = 0
    for pos in positions:
        transitions[pos] = [p for p in positions if p != pos]
        remain_transitions += len(transitions[pos])

    gesture_countdown = []
    for _ in range(n):
        gesture_countdown.append(random.randrange(n-1))

    pos = random.choice(positions)
    print('Starting position is {}'.format(pos))

    while len(transitions[pos]) > 0:
        if gesture_countdown[positions.index(pos)] == 0:
            for gesture in random_gestures():
                seq.append(gesture)
        gesture_countdown[positions.index(pos)] -= 1

        randi = random.randrange(len(transitions[pos]))
        if remain_transitions > 1:
            while len(transitions[transitions[pos][randi]]) == 0:
                randi = random.randrange(len(transitions[pos]))
        new_pos = transitions[pos].pop(randi)
        remain_transitions -= 1

        if positions.index(pos) > positions.index(new_pos):
            direction = 'f'
        else:
            direction = 'b'
        pos = new_pos

        seq.append(Recording('move'+direction, 'move ->'+pos))

    if gesture_countdown[positions.index(pos)] == 0:
        for gesture in random_gestures():
            seq.append(gesture)

    return seq

def random_gestures(n=1):
    static_gestures = []
    for i in range(n):
        static_gestures.append(Recording('stand', 'stand'))
        static_gestures.append(Recording('walk', 'walk'))
        static_gestures.append(Recording('run', 'run'))
        static_gestures.append(Recording('jump', 'jump'))
        static_gestures.append(Recording('jumpd', 'down jump'))
        static_gestures.append(Recording('kick', 'kick', RecMode.MIDDLE))
        static_gestures.append(Recording('duck', 'duck'))
    random.shuffle(static_gestures)
    return static_gestures

def random_mock_gestures(second=30):
    def randt(n=1):
        return int(second/2 + random.randint(int(n*second*0.5), int(n*second*1.5)))

    gestures = [
            Recording('stand', 'stand', n_frames=randt()),
            Recording('stand', 'stand', n_frames=randt(2)),
            Recording('stand', 'stand', n_frames=randt(2)),
            Recording('stand', 'stand', n_frames=randt()),
            Recording('stand', 'stand', n_frames=randt()),
            Recording('stand', 'stand', n_frames=randt()),
            Recording('stand', 'stand', n_frames=randt()),

            Recording('walkf', 'walk', n_frames=randt()),
            Recording('walkf', 'walk', n_frames=randt(2)),
            Recording('walkf', 'walk', n_frames=randt(1.5)),
            Recording('walkf', 'walk', n_frames=randt(3)),
            Recording('walkf', 'walk', n_frames=randt(2)),
            Recording('walkf', 'walk', n_frames=randt()),
            Recording('walkf', 'walk', n_frames=randt(1.5)),
            Recording('walkf', 'walk', n_frames=randt(1.5)),
            Recording('walkf', 'walk', n_frames=randt(1.5)),
            Recording('runf', 'run', n_frames=randt()),
            Recording('runf', 'run', n_frames=randt()),
            Recording('runf', 'run', n_frames=randt(3)),
            Recording('runf', 'run', n_frames=randt(2)),
            Recording('walkb', 'walk back', n_frames=randt(1.5)),
            Recording('walkb', 'walk back', n_frames=randt()),
            Recording('walkb', 'walk back', n_frames=randt()),
            Recording('walkb', 'walk back', n_frames=randt(2)),
            Recording('walkb', 'walk back', n_frames=randt(2)),
            Recording('runb', 'run back', n_frames=randt(2)),
            Recording('runb', 'run back', n_frames=randt()),

            Recording('jumpb', 'jump', n_frames=randt()),
            Recording('jumpb', 'jump', n_frames=randt()),
            Recording('jumpb', 'jump', n_frames=randt()),
            Recording('jumpb', 'jump', n_frames=randt()),
            Recording('jumpb', 'jump', n_frames=randt()),
            Recording('jumpb', 'jump', n_frames=randt()),
            Recording('jumpb', 'jump', n_frames=randt()),
            Recording('jumpb', 'jump', n_frames=randt()),
            Recording('jumps', 'hop', n_frames=randt()),
            Recording('jumps', 'hop', n_frames=randt()),
            Recording('jumps', 'hop', n_frames=randt()),
            Recording('jumps', 'hop', n_frames=randt()),
            Recording('jumps', 'hop', n_frames=randt()),
            Recording('kick', 'kick', n_frames=randt()),
            Recording('kick', 'kick', n_frames=randt()),
            Recording('kick', 'kick', n_frames=randt()),
            Recording('kick', 'kick', n_frames=randt()),
            Recording('kick', 'kick', n_frames=randt()),
            Recording('kick', 'kick', n_frames=randt()),
            Recording('duck', 'duck', n_frames=randt()),
            Recording('duck', 'duck', n_frames=randt()),
            Recording('duck', 'duck', n_frames=randt()),
            Recording('duck', 'duck', n_frames=randt())
    ]

    random.shuffle(gestures)
    return gestures


if __name__ == '__main__':
    print(generate_sequence())
