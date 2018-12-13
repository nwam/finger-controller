"""
This module is used to simultaneously record and label data.

Usage:
    Run

        $ python record.py 124

    to record from IP Webcam at http://192.168.0.124:8080/video. Actions will
    be displayed on-screen along side the video source. Press f to record the
    current Recording.

    The video will be stored at a timestamped file, and Recording data and
    camera data will be stored at a file with the same name and a .pickle
    extension.
"""
import argparse
import time
import pickle
import os
import cv2
import numpy as np
from capture import Capture, CapType
from cnn_input import CnnInput
import recording
from recording import CamSide, CamProps
from finger_controller import h_pos_ratio

def record(cap_source, cap_type, recordings, cam_props, mock, output_dir='../data/'):
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

    while cap.is_opened() and (rec_i < len(recordings) or record_n+1 > 0):
        ret, frame = cap.read()
        if ret == False:
            break
        frame_i += 1

        out.write(frame)

        if record_n+1 > 0 or mock:
            cv2.circle(frame, (6,6), (5), (0,0,255), cv2.FILLED)
            record_n -= 1

        h_line = int(h_pos_ratio * frame.shape[1])
        cv2.line(frame, (h_line, 0), (h_line, h), (0,0,255))
        cnn_input.update(frame)
        cnn_input_show = cv2.resize(cnn_input.frame, (h,h))
        action_display = np.zeros((h, frame.shape[1]+h, 3), dtype=np.uint8)
        if record_n > 0:
            complete = (recordings[rec_i-1].n_frames - record_n) /\
                    recordings[rec_i-1].n_frames
            cv2.rectangle(action_display, (0,0),
                    (int(complete*(frame.shape[1]+h)-1), 16),
                    (0, 255-int(2*complete*255), 0), cv2.FILLED)
        if mock:
            cv2.putText(action_display, recordings[rec_i-1].info, (4, h-16),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
        elif rec_i < len(recordings):
            cv2.putText(action_display, recordings[rec_i].info, (4, h-16),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
        cv2.imshow('frame', np.vstack((np.hstack(
            (frame, cnn_input_show)), action_display)))

        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):
            os.remove(output_path)
            exit()
        if (key == ord('f') or mock) and record_n <= 0:
            try:
                recordings[rec_i].frame = frame_i
            except:
                break
            if recordings[rec_i].rec_mode == recording.RecMode.AFTER:
                record_n = recordings[rec_i].n_frames
            elif recordings[rec_i].rec_mode == recording.RecMode.MIDDLE:
                record_n = recordings[rec_i].n_frames // 2
            else:
                record_n = 1
            rec_i += 1

    pickle.dump((cam_props, recordings), open(output_pickle, 'wb'))
    cap.kill()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('camera_side', type=str,
            help='The side of the user where the camera is placed')
    parser.add_argument('cap_source', type=str,
            help='The source of the Capture object.')
    parser.add_argument('cap_type', nargs='*', default=['video'],
            help='The type of the Capture device used as video input.')
    parser.add_argument('-m', '--mock', action='store_true',
            help='Flag to record a mock game instead of data.')
    args = parser.parse_args()

    mock = args.mock

    output_dir = '../data/'
    if mock:
        output_dir = '../mock_data/fp/'

    cam_side = args.camera_side.lower()
    if cam_side == 'left' or cam_side == 'l':
        cam_side = CamSide.LEFT
    elif cam_side == 'right' or cam_side == 'r':
        cam_side = CamSide.RIGHT
    else:
        print('Invalid camera side. Please use l or r.')
        exit()

    cap_source = args.cap_source
    cap_source_template = 'http://192.168.0.{}:8080/video'
    if args.cap_source.isdigit():
        cap_source = cap_source_template.format(str(args.cap_source))

    cap_type = args.cap_type[0].lower()
    if cap_type == 'video':
        cap_type = CapType.VIDEO
    elif cap_type == 'camera':
        cap_type = CapType.CAMERA

    if not mock:
        recordings = recording.generate_sequence()
    else:
        recordings = recording.random_mock_gestures()

    cam_props = CamProps(cam_side)

    record(cap_source, cap_type, recordings, cam_props, mock, output_dir)
