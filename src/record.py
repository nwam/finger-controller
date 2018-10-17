import argparse
import time
import pickle
import os
import cv2
import numpy as np
from capture import Capture, CapType
from cnn_input import CnnInput

def record(cap_source, cap_type, max_frames=300, output_dir='../data/'):
    gesture_name = 'kick'

    cap = Capture(cap_source, cap_type)
    recorded_frames = set()
    output_prefix = os.path.join(output_dir,
            '{}-{}'.format(gesture_name, str(time.time())))
    output_path = output_prefix + '.avi'
    output_pickle = output_prefix + '.pickle'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (160,120))

    n_frames = 0
    record = False
    first_frame = cap.read()[1]
    cnn_input = CnnInput(first_frame)
    h = first_frame.shape[0]

    while cap.is_opened() and len(recorded_frames) < max_frames:
        ret, frame = cap.read()
        if ret == False:
            break

        out.write(frame)

        if record:
            recorded_frames.add(n_frames)
            cv2.circle(frame, (6,6), (5), (0,0,255), cv2.FILLED)
        n_frames += 1

        cnn_input.update(frame)
        cnn_input_show = cv2.resize(cnn_input.frame, (h,h))
        cv2.imshow('frame', np.hstack((frame, cnn_input_show)))

        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            record = not record
        if key == ord('f') and not record:
            recorded_frames.add(n_frames)
            cv2.circle(frame, (6,6), (5), (45,210,45), cv2.FILLED)


    pickle.dump(recorded_frames, open(output_pickle, 'wb'))
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


    record(cap_source, cap_type)
