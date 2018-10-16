"""
CnnInput is a class that, given video frames, creates input frames for the CNN.

CnnInput uses computer vision (optical flow and mhi) to create a frame that
provides the neural network sufficient information for understanding the
gesture's movement over the past short period of time.

Usage:
    cnn_input = CnnInput(capture.read())
    while True:
        cnn_input.update(capture.read())
        print(cnn_input.frame) # holds the updated input for the CNN
"""
import cv2
import numpy as np
import dataset
import vision

class CnnInput:
    def __init__(self, first_frame, edge_clip=3, mhi_alpha=0.25, debug=False):
        self.edge_clip = edge_clip
        self.original_shape = tuple([n + 2*self.edge_clip
                for n in reversed(dataset.input_shape[:2])])
        self.debug = debug

        first_frame = self._prepare_frame(first_frame)
        self.flow = vision.OpticalFlow(first_frame)
        self.mhi = vision.MHI(self.flow.flow_vis.shape, np.uint8, mhi_alpha)

    def update(self, frame):
        resized_frame = self._resize_frame(frame)
        prepared_frame = self._gray_frame(resized_frame)

        self.flow.update(prepared_frame)
        self.mhi.update(self.flow.vis)
        clipped_mhi = self.mhi.mhi[self.edge_clip:-self.edge_clip,
                self.edge_clip:-self.edge_clip]

        if self.debug:
            cv2.imshow('cnn_input', np.hstack((resized_frame, self.flow.vis, self.mhi.mhi)))

        self.frame = clipped_mhi

    def _resize_frame(self, frame):
        if frame.shape[:2] != self.original_shape:
            frame = cv2.resize(frame, self.original_shape)
        return frame

    def _gray_frame(self, frame):
        if frame.ndim > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _prepare_frame(self, frame):
        frame = self._resize_frame(frame)
        frame = self._gray_frame(frame)
        return frame

