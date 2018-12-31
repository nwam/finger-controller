"""
This modules reads recorded videos, computes the CnnInput of each frame and
writes only labeled frames' CnnInput to their appropriate directories.
It also saves a pickle which bundles each individual Recording.

Usage:
        $ python preprocess.py

    The preprocessed data will be saved in ../data/preprocessed/
    The bundle data will be saved at ../data/preprocessed/bundles.pickle and
    will contain a structure of the format

        {'kick': [['kick/1-1-1.png', 'kick/1-1-2.png', ..], [..], ..],
                'jump': .., ..}

    which is a dictionary where each key is a gesture and each value is a list
    of bundles of paths to files of the same Recording. For example,
    '1-1-1.png' and '1-1-2.png' are both from the same Recording of a 'kick'.
"""

import os
import pickle
import cv2
from capture import Capture, CapType
from cnn_input import CnnInput
from recording import Recording, RecMode, gestures, CamSide, CamProps

data_dir = '../data/'


def preprocess(data_dir=data_dir):
    ftype = '.png'
    output_dir_prefix = os.path.join(data_dir, 'preprocessed')
    output_pickle = os.path.join(output_dir_prefix, 'bundles.pickle')
    data = {}

    for gesture in gestures:
        data[gesture] = [[]]
        output_dir = os.path.join(output_dir_prefix, gesture)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    file_id = 0
    for filename in os.listdir(data_dir):
        if os.path.splitext(filename)[1] != '.avi':
            continue

        input_file = os.path.join(data_dir, filename)
        print('Processing {} ...'.format(input_file))
        pickle_file = '{}.pickle'.format(os.path.splitext(input_file)[0])
        cap = Capture(input_file, CapType.VIDEO)
        ret, first_frame = cap.read()
        cnn_input = CnnInput(first_frame, debug=True)
        if os.path.exists(pickle_file):
            cam_props, recordings = pickle.load(open(pickle_file, 'rb'))

        frames_i = 0
        rec_i = 0
        recf_i = 0
        rec_frames = recordings[rec_i].get_frames()
        output_dir = os.path.join(output_dir_prefix, recordings[rec_i].label)
        label = None

        while cap.is_opened() and rec_i < len(recordings):
            if len(recordings[rec_i].get_frames()) == 0:
                rec_i += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break
            if cam_props.side == CamSide.LEFT:
                frame = cv2.flip(frame, 1)

            cnn_input.update(frame)

            # Move to the next Recording object
            if recf_i >= len(rec_frames):
                recf_i = 0
                while rec_frames[recf_i] < frames_i and rec_i < len(recordings) - 1:
                    rec_i += 1
                    rec_frames = recordings[rec_i].get_frames()
                    output_dir = os.path.join(
                            output_dir_prefix, recordings[rec_i].label)
                data[recordings[rec_i].label].append([])

            # Save a frame
            if frames_i == rec_frames[recf_i]:
                output_name = '{}-{}-{}{}'.format(
                        str(file_id), str(rec_i), str(recf_i), ftype)
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, cnn_input.frame)
                data[recordings[rec_i].label][-1].append(
                        os.path.join(recordings[rec_i].label, output_name))
                recf_i += 1
            else:
                output_name = '{}{}'.format(len(data['nothing']), ftype)
                output_path = os.path.join(
                        output_dir_prefix, 'nothing', output_name)
                cv2.imwrite(output_path, cnn_input.frame)
                data['nothing'].append(
                        [os.path.join('nothing', output_name)])


            frames_i += 1

        file_id += 1
        cap.kill()

    pickle.dump(data, open(output_pickle, 'wb'))

if __name__ == '__main__':
    preprocess()
