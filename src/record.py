"""
This module is used to simultaneously record and label data.

Usage:
    Run

        $ python record.py 124

    to record from IP Webcam at http://192.168.0.124:8080/video. Actions will
    be displayed on-screen along side the video source. Press f to record the
    current Recording.
"""
import argparse
import time
import pickle
import enum
import os
import cv2
import numpy as np
from capture import Capture, CapType
from cnn_input import CnnInput

class RecMode(enum.Enum):
    BEFORE = 0
    MIDDLE = 1
    AFTER = 2

class Recording:
    def __init__(self, label, text, rec_mode, n_frames):
        self.label = label
        self.text = text
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


def record(cap_source, cap_type, recordings, output_dir='../data/'):
    cap = Capture(cap_source, cap_type)
    output_prefix = os.path.join(output_dir,
            str(time.time()))
    output_path = output_prefix + '.avi'
    output_pickle = output_prefix + '.pickle'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (160,120))

    frame_i = 0
    rec_i = 0
    record_n = 0
    first_frame = cap.read()[1]
    cnn_input = CnnInput(first_frame)
    h = first_frame.shape[0]

    while cap.is_opened() and (rec_i < len(recordings) or record_n <= 0):
        ret, frame = cap.read()
        if ret == False:
            break
        frame_i += 1

        out.write(frame)

        if record_n > 0:
            cv2.circle(frame, (6,6), (5), (0,0,255), cv2.FILLED)
            record_n -= 1

        cnn_input.update(frame)
        cnn_input_show = cv2.resize(cnn_input.frame, (h,h))
        if rec_i < len(recordings):
            cv2.putText(frame, recordings[rec_i].text, (2, h-3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        cv2.imshow('frame', np.hstack((frame, cnn_input_show)))

        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):
            break
        if key == ord('f') and record_n <= 0:
            recordings[rec_i].frame = frame_i
            if recordings[rec_i].rec_mode == RecMode.AFTER:
                record_n = recordings[rec_i].n_frames
            elif recordings[rec_i].rec_mode == RecMode.MIDDLE:
                record_n = recordings[rec_i].n_frames // 2
            else:
                record_n = 1
            rec_i += 1


    pickle.dump(recordings, open(output_pickle, 'wb'))
    cap.kill()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cap_source', type=str,
            help='The source of the Capture object.')
    parser.add_argument('cap_type', nargs='*', default=['video'],
            help='The type of the Capture device used as video input.')
    args = parser.parse_args()

    cap_source = args.cap_source
    cap_source_template = 'http://192.168.0.{}:8080/video'
    if args.cap_source.isdigit():
        cap_source = cap_source_template.format(str(args.cap_source))

    cap_type = args.cap_type[0].lower()
    if cap_type == 'video':
        cap_type = CapType.VIDEO
    elif cap_type == 'camera':
        cap_type = CapType.CAMERA

    recordings = []
    recordings.append(Recording('run', 'run front close', RecMode.AFTER, 9))
    recordings.append(Recording('run', 'run ->mid', RecMode.AFTER, 9))
    recordings.append(Recording('run', 'run mid', RecMode.AFTER, 9))
    recordings.append(Recording('kick', 'kick', RecMode.MIDDLE, 9))
    recordings.append(Recording('run', 'run ->back', RecMode.AFTER, 9))
    recordings.append(Recording('run', 'run back', RecMode.AFTER, 9))

    record(cap_source, cap_type, recordings)
