import os
import pickle
import cv2
from capture import Capture, CapType
from cnn_input import CnnInput
import dataset

data_dir = '../data/'
gestures = dataset.class_ids.keys()

def preprocess(data_dir=data_dir):
    ftype = '.png'
    for gesture in gestures:
        output_dir = os.path.join(data_dir, 'preprocessed', gesture)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        name_id = 0
        for filename in os.listdir(data_dir):
            if filename[:len(gesture)] != gesture or \
                    os.path.splitext(filename)[1] != '.avi':
                continue

            input_file = os.path.join(data_dir, filename)
            recorded_frames = None
            pickle_file = '{}.pickle'.format(os.path.splitext(input_file)[0])
            if os.path.exists(pickle_file):
                recorded_frames = pickle.load(open(pickle_file, 'rb'))

            cap = Capture(input_file, CapType.VIDEO)
            ret, first_frame = cap.read()
            cnn_input = CnnInput(first_frame, debug=True)

            n_frames = 0
            while cap.is_opened():
                ret, frame = cap.read()
                if not ret:
                    break

                cnn_input.update(frame)

                output_name = os.path.join(output_dir, str(name_id) + ftype)
                if recorded_frames is None or n_frames in recorded_frames:
                    cv2.imwrite(output_name, cnn_input.frame)
                    name_id += 1
                n_frames += 1

                key = cv2.waitKey(2) & 0xFF
                if key == ord('q'):
                    break

            cap.kill()

if __name__ == '__main__':
    preprocess()
