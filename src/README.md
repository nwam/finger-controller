# Finger People source
There are many parts to finger people. We will describe how these parts interact and how to use these modules together.

### Running Finger People
To run Finger People with an existing model, streaming video from 192.168.0.124:8080/video from a camera on the right, run

    python finger_people.py models/amazing_model.hdf5 r 124

### Doing everything from scratch
In order to train a model, we must gather a decent amount of data with `record.py`, then preprocess the data for the CNN with `preprocess.py`, and build and train a CNN with `learn.py`.

    python record.py r 124
    python record.py l 124
    python record.py r 124
    ...
    python preprocess.py
    python learn.py

To evaluate the model (which may help with debugging), we use `evaluate.py` on the trained model. Once the model is trained, we can [run finger people on the trained model](#running-finger-people).

See the file headers for more information.
