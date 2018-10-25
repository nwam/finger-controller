"""
This module holds various constants, functions, and a class for handling the
dataset.

Usage:
    train_gen, validation_gen = get_generators()

Important constants:
    input_shape: A tuple holding the shape of the input layer to the CNN.
    gesture_ids: Unique IDs for each gesture.
    is_to_class: Inverse of gesture_ids.
    n_classes: The number of unique gestures.
"""
import keras.utils
from keras.preprocessing.image import ImageDataGenerator
import os
import pickle
import random
import copy
import numpy as np
import cv2
from collections import defaultdict

input_shape = (28, 28, 3)
gestures = ['stand', 'walk', 'run', 'jump', 'jumpd', 'kick', 'duck', 'movef',
        'moveb']
gesture_ids = dict([(gesture, i) for i, gesture in enumerate(gestures)])
id_to_gesture = dict([(v,k) for k,v in gesture_ids.items()])
n_classes = len(gesture_ids)
data_dir = '../data/preprocessed/'
bundles_path = os.path.join(data_dir, 'bundles.pickle')

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, data_dir=data_dir, batch_size=32,
                 dim=input_shape, n_classes=n_classes, shuffle=True,
                 datagen=None):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.on_epoch_end()
        self.datagen = datagen

    def __data_generation(self, list_IDs_temp):
        """ Generates data containing batch_size samples """
        # X : (n_samples, *dim)
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            path = os.path.join(self.data_dir, ID)
            X[i,] = cv2.imread(path)
            if self.datagen:
                X[i,] = self.datagen.random_transform(X[i,])
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

def get_bundles(path=bundles_path):
    return pickle.load(open(bundles_path, 'rb'))

def flatten_bundles(bundles):
    flat_bundles = {}
    for gesture in bundles:
        flat_bundles[gesture] = \
                [path for bundle in bundles[gesture] for path in bundle]
    return flat_bundles

def get_labels(bundles, gesture_ids=gesture_ids):
    """
    Flattens and reverses bundles dict.

    Args:
        bundles: a dict labels to list of bundles of frames.
        gesture_ids: a dict of labels to their ids

    Returns:
        labels: a dict of paths to their labels.
    """
    labels = {}
    for gesture in bundles:
        for bundle in bundles[gesture]:
            for path in bundle:
                labels[path] = gesture_ids[gesture]
    return labels

def partition_labels(bundles, train=0.8):
    """ Returns a random train/validation partition of labels' keys.

    Args:
        bundles: a dict labels to list of bundles of frames.
        train: a number [0,1] specifying the percent of partition for train.

    Returns:
        partition: dictionary with 'train' and 'validation' keys, each
            containing a list of keys from `bundles`.
    """
    partition = {}
    partition['train'] = []
    partition['validation'] = []

    for gesture in bundles:
        random.shuffle(bundles[gesture])
        split = int(train*len(bundles[gesture]))

        for i, bundle in enumerate(bundles[gesture]):
            for path in bundle:
                if i < split:
                    partition['train'].append(path)
                else:
                    partition['validation'].append(path)

    return partition

def get_generators():
    """ Creates and returns a train and test generators. """
    bundles = get_bundles()
    labels = get_labels(bundles)
    partition = partition_labels(bundles)

    datagen = ImageDataGenerator(
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    rescale=1/255,
                    shear_range=0.25,
                    zoom_range=0.15,
                    fill_mode='nearest')

    training_generator = DataGenerator(partition['train'], labels,
            datagen=datagen)
    validation_generator = DataGenerator(partition['validation'], labels)

    return training_generator, validation_generator
