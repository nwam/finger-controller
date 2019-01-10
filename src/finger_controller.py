"""
This is the main loop of the controller.

It reads video, passes it through the CNN, and performs an input action to the
game based on the CNN's output.

Usage:
    Run

        python finger_controller.py models/best_model.hdf5 124

    to record from IP Webcam at http://192.168.0.124:8080/video then
    start a game and press g to enable keyboard inputs from finger_controller.
"""

import cv2
import keras
import argparse
import numpy as np

from cnn_input import CnnInput
from vision import MHB
import dataset
from dataset import gesture_ids
from capture import Capture, CapType
from game_input import GameInput
from recording import CamSide, CamProps
import debug as debugutils
from post_process import RunProcessor, StickyTolerance

h_pos_ratio = 0.425
flap_thresh = 0.2

def finger_controller(model_path, cap_source, cap_type, cam_props, record=None, debug=False):
    model = keras.models.load_model(model_path)
    cap = Capture(cap_source, cap_type)
    game_input = GameInput()
    ret, first_frame = cap.read()
    cnn_input = CnnInput(first_frame)
    run_processor = RunProcessor(cnn_input)
    sticky_tolerance = StickyTolerance()

    action = None
    hpos_color = None
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
        prediction = model.predict(cnn_input_4d)[0]

        class_id = np.argmax(prediction)
        flap_pred = prediction[dataset.gesture_ids['flap']]
        dflap_pred = prediction[dataset.gesture_ids['dflap']]
        if flap_pred > flap_thresh or dflap_pred > flap_thresh:
            if flap_pred > dflap_pred:
                class_id = dataset.gesture_ids['flap']
            else:
                class_id = dataset.gesture_ids['dflap']
        class_label = dataset.id_to_gesture[class_id]

        ''' STICKY OUTPUT and TOLERANCE'''
        action = sticky_tolerance.process(class_label, prediction[class_id], action)

        ''' GAME INPUT '''
        game_input.do(action)

        ''' OUTPUT / DEBUG / FEEDBACK '''
        cv2.putText(frame, action, (2, h-3), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        cv2.putText(frame, str(int(run_processor.h_speed)),
                (2, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        h_line = int(h_pos_ratio * frame.shape[1])
        cv2.line(frame, (h_line, 0), (h_line, h), (0,0,255))

        if debug:
            cnn_input_debug = cv2.resize(cnn_input.frame, (h,h))
            mhb_debug = debugutils.mhb_frame(run_processor.mhb, h, h)
            debug_frame = np.hstack((frame, cnn_input_debug, mhb_debug))
            prediction_debug = debugutils.prediction_frame(
                    prediction, 300, debug_frame.shape[1])
            debug_frame = np.vstack((debug_frame, prediction_debug))
        else:
            debug_frame = frame

        if record:
            record.write(frame)

        cv2.imshow('frame', debug_frame)

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
            help='Flag to record the frames to rec.avi.')
    parser.add_argument('-d', '--debug', action='store_true',
            help='Display debug information along with frames.')
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

    cap_type = args.cap_type[0].lower()
    if cap_type == 'video':
        cap_type = CapType.VIDEO
    elif cap_type == 'camera':
        cap_type = CapType.CAMERA

    cap_source = args.cap_source
    cap_source_template = 'http://192.168.0.{}:8080/video'
    if args.cap_source.isdigit() and cap_type == CapType.VIDEO:
        cap_source = cap_source_template.format(str(args.cap_source))

    record = None
    if args.record:
        record_path = 'rec.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        record = cv2.VideoWriter(record_path, fourcc, 30.0, (160,120))

    cam_props = CamProps(cam_side)

    finger_controller(model_path, cap_source, cap_type, cam_props, record,
            debug=args.debug)
