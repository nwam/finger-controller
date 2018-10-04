import os
import random
import copy
import keras.utils
import numpy as np
import cv2

data_dir = '../data/preprocessed/'
input_shape = (28, 28, 3)
class_ids = {'stand': 0, 'walk': 2, 'run': 3, 'jumps': 1}
ids_classes = dict([(v,k) for k,v in class_ids.items()])
n_classes = len(class_ids)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, data_dir=data_dir, batch_size=32,
                 dim=input_shape, n_classes=n_classes, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.on_epoch_end()

    def __data_generation(self, list_IDs_temp):
        """ Generates data containing batch_size samples """
        # X : (n_samples, *dim)

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = cv2.imread(data_dir + ID)
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

def get_labels(path=data_dir, class_ids=class_ids):
    """
    Returns a dict of _keys_ to _labels_ where the _labels_ are the top-level
    dir names in `path` and the _keys_ are the relative filenames from `path`.
    """
    labels = {}
    for root, dirs, files in os.walk(path):
        for f in files:
            rel_root = os.path.relpath(root, path)
            rel_path = os.path.join(rel_root, f)
            label = os.path.normpath(rel_root).split(os.sep)[0]
            labels[rel_path] = class_ids[label]
    return labels

def partition_labels(labels, train=0.8):
    """ Returns a random train/validation partition of labels' keys.

    Args:
        labels: a dict of ids to labels.
        train: a number [0,1] specifying the percent of partition for train.

    Returns:
        partition: dictionary with 'train' and 'validation' keys, each
            containing a list of keys from `labels`.
    """
    partition = {}

    keys = list(labels.keys())
    random.shuffle(keys)
    split = int(train*len(keys))

    partition['train'] = keys[:split]
    partition['validation'] = keys[split:]

    return partition

def get_generators():
    labels = get_labels()
    partition = partition_labels(labels)

    training_generator = DataGenerator(partition['train'], labels)
    validation_generator = DataGenerator(partition['validation'], labels)

    return training_generator, validation_generator
