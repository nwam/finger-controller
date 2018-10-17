"""
This module contains functions for building and training the CNN.

Usage:
        $ python learn.py

    The model will be saved to 'checkpoints/'
"""
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import os
import dataset
import argparse

n_classes = dataset.n_classes
checkpoints_dir = './checkpoints'

def build_model(input_shape=dataset.input_shape, n_classes=n_classes):
    model = Sequential()
    model.add(Conv2D(40, kernel_size=5, padding='same',
            input_shape=input_shape, activation='relu'))
    model.add(Conv2D(50, kernel_size=5, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.33))

    model.add(Conv2D(70, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(100, kernel_size=3, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.33))

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model

def train(saved_model=None):
    #K.set_image_data_format('channels_last')
    #numpy.random.seed(0)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    training_generator, validation_generator = dataset.get_generators()

    if saved_model == None:
        model = build_model()
        model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    else:
        model = keras.model.load_model(saved_model)

    checkpoint_callback = ModelCheckpoint(os.path.join(checkpoints_dir,
            'checkpoint-{epoch:02d}-{acc:.3f}.hdf5'))
    tensrboard_callback = TensorBoard(log_dir='./tensorboard_logs')

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=150,
                        use_multiprocessing=True,
                        workers=6,
                        callbacks=[checkpoint_callback, tensrboard_callback])

    model.save(os.path.join(checkpoints_dir, 'final_model.hdf5'))

    #scores = model.evaluate(x_test, y_test, verbose = 10 )
    #print(scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
            help='A path to the model to resume training.')
    args = parser.parse_args()
    model = args.model

    train()
