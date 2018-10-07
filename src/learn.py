import numpy as np
import pandas as pd
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import dataset

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

def train():
    #K.set_image_data_format('channels_last')
    #numpy.random.seed(0)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    training_generator, validation_generator = dataset.get_generators()

    model = build_model()

    model.compile(loss='categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(os.path.join(checkpoints_dir,
            'checkpoint-{epoch:02d}-{acc:.3f}.hdf5'))

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=20,
                        use_multiprocessing=True,
                        workers=6,
                        callbacks=[checkpoint_callback])

    model.save(os.path.join(checkpoints_dir, 'final_model.hdf5'))

    #scores = model.evaluate(x_test, y_test, verbose = 10 )
    #print(scores)

if __name__ == '__main__':
    train()
