# Finger People source
There are many parts to finger people. We will describe how these parts interact and how to use these modules together.

### Running Finger People
To run Finger People with an existing model, run

    python finger_people.py models/my_model 124

### Doing everything from scratch
In order to train a model, we must gather data with `record.py`, then preprocess the data for the CNN with`preprocess.py`, and build and train a CNN with `learn.py`.

    python record.py 124
    python preprocess.py
    python learn.py

See the file headers for more information.
