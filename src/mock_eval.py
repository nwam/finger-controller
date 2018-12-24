import os
import pickle
from capture import Capture, CapType
from recording import CamSide
from cnn_input import CnnInput
from record_keyboard import KeyPress
from post_process import RunProcessor, StickyTolerance
import keras
import argparse
import dataset
import numpy as np
from pynput.keyboard import Key

data_dir = '../mock_data/'
kb_dir = os.path.join(data_dir, 'kb')
fp_dir = os.path.join(data_dir, 'fp')

def eval_fc(fname_pickle, model_path):
    cam_props, recordings = pickle.load(open(fname_pickle, 'rb'))
    speeds = np.zeros(len(recordings))
    hits = np.zeros(len(recordings))
    extras = np.zeros(len(recordings))

    fname_avi = os.path.splitext(fname_pickle)[0] + '.avi'
    model = keras.models.load_model(model_path)
    cap = Capture(fname_avi, CapType.VIDEO)
    ret, first_frame = cap.read()
    cnn_input = CnnInput(first_frame)
    run_processor = RunProcessor(cnn_input)
    sticky_tolerance = StickyTolerance()
    action = None

    frame_num = 0
    rec_i = 1

    while cap.is_opened():

        ret, frame = cap.read()
        if not ret:
            break
        if cam_props.side == CamSide.LEFT:
            frame = cv2.flip(frame, 1)

        cnn_input.update(frame)
        cnn_input_4d = np.expand_dims(cnn_input.frame, 0)
        prediction = model.predict(cnn_input_4d)[0]
        class_id = np.argmax(prediction)
        class_label = dataset.id_to_gesture[class_id]

        class_label, direction = run_processor.process(class_label)
        action = sticky_tolerance.process(class_label, prediction[class_id], action)
        if direction is not None and action in ['walk', 'run']:
            action = action + direction

        target = recordings[rec_i].label
        if target in ['jumpb', 'jumps']:
            target = 'jump'
        if action == target and hits[rec_i] == 0:
            hits[rec_i] = 1
            speeds[rec_i] = frame_num - recordings[rec_i].frame

        frame_num += 1
        if rec_i + 1 < len(recordings) and frame_num >= recordings[rec_i+1].frame:
            if hits[rec_i] == 0:
                speeds[rec_i] = frame_num - recordings[rec_i].frame
            rec_i += 1

    return hits, speed, extras

def eval_kb(fname_pickle):
    data = pickle.load(open(fname_pickle, 'rb'))
    recordings = data['recordings']
    presses = data['presses']
    pressed = []
    speeds = np.zeros(len(recordings))
    hits = np.zeros(len(recordings))
    extras = np.zeros(len(recordings))

    p = 0 # press index

    for press in presses:
        if press.release is None:
            press.release = press.press + 0.1

    for r, recording in enumerate(recordings):
        while r+1 >= len(recordings) or \
                (len(pressed) > 0 and pressed[-1].release < recordings[r+1].frame) or \
                (p < len(presses) and presses[p].press < recordings[r+1].frame):

            if len(pressed) == 0 and p >= len(presses):
                break
            elif len(pressed) > 0 and p >= len(presses):
                t = pressed.pop().release
            elif len(pressed) == 0 and p < len(presses):
                pressed.append(presses[p])
                t = presses[p].press
                p += 1
            else:
                if pressed[-1].release < presses[p].press:
                    t = pressed.pop().release
                else:
                    pressed.append(presses[p])
                    t = presses[p].press
                    p += 1

            pressed.sort(key=lambda p: p.release, reverse=True)
            action = get_kb_action(pressed)

            target = recording.label
            if target in ['jumpb', 'jumps']:
                target = 'jump'
            if action == target and hits[r] == 0:
                hits[r] = 1
                speeds[r] = t - recording.frame

        if hits[r] == 0:
            if r+1 < len(recordings):
                speeds[r] = recordings[r+1].frame - recordings[r].frame
            else:
                speeds[r] = 1.5

    import pdb
    pdb.set_trace()
    return hits, speeds, extras

def get_kb_action(pressed):
    pressed_set = set([p.key for p in pressed])

    if pressed_set == set():
        return 'stand'
    elif pressed_set == set([Key.right]):
        return 'walkf'
    elif pressed_set == set([Key.right, 'd']):
        return 'runf'
    elif pressed_set == set([Key.left]):
        return 'walkb'
    elif pressed_set == set([Key.left, 'd']):
        return 'runb'
    elif pressed_set == set(['f']):
        return 'jump'
    elif pressed_set == set(['d']):
        return 'kick'
    elif pressed_set == set([Key.down]):
        return 'duck'
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
            help='Path to gesture recognition .hdf5 keras model')
    args = parser.parse_args()

    eval_kb(os.path.join(kb_dir, '1544500883.9772544.pickle'))
    #eval_fc(os.path.join(fp_dir, '1544492021.8871984.pickle'), args.model)
