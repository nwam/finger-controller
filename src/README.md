# Finger Controller source
There are many parts to finger controller. We will describe how these parts interact and how to use these modules together.

### Running Finger Controller
To run Finger Controller with an existing model, streaming video from `192.168.0.124:8080/video` from a camera on the right, run

    python finger_controller.py models/amazing_model.hdf5 r 124

The preferred capture method for finger controller is to use a mobile device's camera and stream the video to the host device. This is preferable because mobile devices are ubiquitous and easy to place for Finger Controller's use. Currently, we use [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en), and Android app that lets us stream video from our Android device to our host device through local WiFi.

### Doing everything from scratch
In order to train a model, we must gather a decent amount of data with `record.py`, then preprocess the data for the CNN with `preprocess.py`, and build and train a CNN with `learn.py`.

    python record.py r 124
    python record.py l 124
    python record.py r 124
    ...
    python preprocess.py
    python learn.py

To evaluate the model (which may help with debugging), we use `evaluate.py` on the trained model. Once the model is trained, we can [run finger controller on the trained model](#running-finger-controller).

See the file headers for more information.
