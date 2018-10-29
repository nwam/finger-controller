"""
This is the main loop of the controller.

It reads video, passes it through the CNN, and performs an input action to the
game based on the CNN's output.

Usage:
    Run

        python finger_people.py models/best_model.hdf5 124

    to record from IP Webcam at http://192.168.0.124:8080/video then
    start a game and press g to enable keyboard inputs from finger_people.
"""

import cv2
import keras
import argparse
import numpy as np

from cnn_input import CnnInput
from vision import MHB
import dataset
from capture import Capture, CapType
from game_input import GameInput
from recording import CamSide, CamProps

sticky_size = 2

def finger_people(model_path, cap_source, cap_type, cam_props, record=None):
    model = keras.models.load_model(model_path)
    cap = Capture(cap_source, cap_type)
    game_input = GameInput()

    ret, first_frame = cap.read()
    cnn_input = CnnInput(first_frame)

    mhb = MHB(cnn_input.flow, np.ones((2,2)))
    h_speed_alpha = 0.2
    h_speed_thresh = 5.0
    h_speed = h_speed_thresh
    h_classes = ['run', 'walk']

    action = None
    prev_label = None
    sticky = 0

    h = first_frame.shape[0]


    while cap.is_opened():
        ret, frame = cap.read()
        if not ret:
            break
        if cam_props.side == CamSide.LEFT:
            frame = cv2.flip(frame, 1)

        ''' CNN '''
        cnn_input.update(frame)
        cnn_input_4d = np.expand_dims(cnn_input.frame, 0)
        prediction = model.predict(cnn_input_4d)

        class_id = np.argmax(prediction)
        class_label = dataset.id_to_gesture[class_id]

        ''' WALK vs RUN '''
        if class_label not in h_classes:
            h_speed = h_speed_thresh
        else:
            h_speed = h_speed_alpha*mhb.compute() + (1-h_speed_alpha)*h_speed
            if h_speed > h_speed_thresh:
                class_label = 'run'
            else:
                class_label = 'walk'

        ''' STICKY OUTPUT '''
        if class_label != prev_label:
            sticky_i = 0
        if sticky_i < sticky_size:
            sticky_i += 1
        else:
            action = class_label
        prev_label = class_label

        ''' GAME INPUT '''
        game_input.do(action)

        ''' OUTPUT / DEBUG '''
        cnn_input_show = cv2.resize(cnn_input.frame, (h,h))
        cv2.putText(frame, action, (2, h-3), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        cv2.putText(frame, str(int(h_speed)), (2, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        hmag_vis = cv2.cvtColor(cv2.resize((mhb.hmag*255).astype(np.uint8), (h,h)), cv2.COLOR_GRAY2RGB)
        debug_frame = np.hstack((frame, cnn_input_show, hmag_vis))
        cv2.imshow('frame', debug_frame)

        if record:
            record.write(frame)

        ''' KEYBOARD INPUT '''
        key = cv2.waitKey(2) & 0xFF
        if key == ord('q'):
            break
        if key == ord('g'):
            game_input.enabled = not game_input.enabled

    cap.kill()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
            help='Path to gesture recognition .hdf5 keras model')
    parser.add_argument('camera_side', type=str,
            help='The side of the user where the camera is placed')
    parser.add_argument('cap_source', type=str,
            help='The source of the Capture object.')
    parser.add_argument('cap_type', nargs='*', default=['video'],
            help='The type of the Capture device used as video input.')
    parser.add_argument('-r', '--record', action='store_true',
            help='Flag to record the debug frames to rec.avi.')
    args = parser.parse_args()

    model_path = args.model

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

    record = None
    if args.record:
        record_path = 'rec.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        record = cv2.VideoWriter(record_path, fourcc, 30.0, (160,120))

    cam_props = CamProps(cam_side)

    finger_people(model_path, cap_source, cap_type, cam_props, record)
