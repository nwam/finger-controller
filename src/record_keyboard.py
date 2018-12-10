"""
This module is used to have a tester play a mock game with a keyboard in
order to get accuracy data and reaction time data

Usage:
    Run
        $ python record_keyboard.py
    to initiate the mock game. Data will be saved in ../mock_data.
    All data will be saved in a timestamped .pickle file.

    The .pickle file will contain a dictionary {'recordings':r, 'presses':p} where
    r is a list of Recording objects and p is a list of KeyPress objects.
"""

import time
import os
import recording
import pynput
import cv2
import numpy as np
import pickle

presses = []

class KeyPress:
    def __init__(self, key, press):
        self.key = key
        self.press = press
        self.release = None

    def __str__(self):
        return 'KeyPress({}, {}, {})'.format(self.key, self.press, self.release)

    def __repr__(self):
        return self.__str__()

def on_press(key):
    if key is None:
        return

    for press in reversed(presses):
        if press.key == key:
            if press.release is not None:
                presses.append(KeyPress(key, time.time()))
            return

    presses.append(KeyPress(key, time.time()))

def on_release(key):
    if key is None:
        return

    for press in reversed(presses):
        if press.key == key:
            press.release = time.time()
        return

def record(recordings, out_fname, second):
    h = 300
    w = 700

    listener = pynput.keyboard.Listener(on_press, on_release)
    listener.start()

    for recording in recordings:
        recording.frame = time.time()

        for frame_num in range(recording.n_frames):

            action_display = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(action_display, recording.info, (4,h-16),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
            cv2.imshow('frame', action_display)

            key = cv2.waitKey(int(1000/second)) & 0xFF
            if key == ord('q'):
                exit()

    listener.stop()
    pickle.dump(
            {'recordings':recordings, 'presses':presses},
            open(out_fname, 'wb'))


if __name__ == '__main__':
    output_dir = '../mock_data/kb/'
    out_fname = os.path.join(output_dir, '{}.pickle'.format(str(time.time())))
    second = 30
    recordings = recording.random_mock_gestures(second=second)

    record(recordings, out_fname, second)
