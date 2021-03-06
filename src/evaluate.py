"""
This module contains functions for evaluating trained models.

Run

    python evaluate.py model/amazing_model.hdf5

to evaluate amazing_model.hdf5.
"""
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import os
import argparse
import random
import dataset

data_dir = dataset.data_dir
gestures = dataset.gestures
gesture_ids = dataset.gesture_ids
id_to_gesture = dataset.id_to_gesture
paths = dataset.flatten_bundles(dataset.get_bundles())

def confusion_matrix(model, data_dir=data_dir, n=1197):
    confusion = np.zeros((dataset.n_classes, dataset.n_classes))

    for gesture in gesture_ids:
        actual_id = gesture_ids[gesture]
        random.shuffle(paths[gesture])
        for path in paths[gesture][:n]:
            frame = cv2.imread(os.path.join(data_dir, path))
            frame = np.expand_dims(frame, 0)
            prediction = model.predict(frame)
            prediction_id = np.argmax(prediction)
            confusion[actual_id][prediction_id] += 1

    confusion = confusion / n * 100
    return np.round(confusion).astype(int)

def plot_confusion_matrix(confusion):
    plt.imshow(confusion, cmap='Greens')
    plt.title('Gesture CNN Confusion Matrix', fontsize=16)
    tick_marks = np.arange(len(gestures))
    plt.xticks(tick_marks, gestures, rotation=45, fontsize=12)
    plt.yticks(tick_marks, gestures, fontsize=12)

    thresh = confusion.max() / 2
    for i, j in itertools.product(
            range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, '{}%'.format(confusion[i, j]),
                horizontalalignment='center',
                color='white' if confusion[i, j] > thresh else 'black')

    plt.xlabel('Predicted gesture', fontsize=14)
    plt.ylabel('Actual gesture', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
            help='Path to the model to evaluate')
    args = parser.parse_args()

    model = keras.models.load_model(args.path)
    confusion = confusion_matrix(model)
    plot_confusion_matrix(confusion)
