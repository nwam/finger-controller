"""
This module contains classes that perform computer vision alogirhtms.
"""
import cv2
import numpy as np

class OpticalFlow:
    """
    Optical Flow computes a visual representation of optical flow on each
    call to update().

    Usage:
        frame = get_first_frame()
        flow = OpticalFlow(frame)
        while True:
            frame = get_frame()
            flow.update(frame)
            flow.flow_vis # returns a visualization of optical flow
    """
    # Parameters for farneback optical flow
    fb_params = dict(
        pyr_scale = 0.5,
        levels = 3,
        winsize = 5,
        iterations = 3,
        poly_n = 5,
        poly_sigma = 1.2,
        flags = 0)

    def __init__(self, frame):
        """ Initializes optical flow using the first frame of the sequence. """
        shape = frame.shape
        if frame.ndim == 2:
            shape = shape + (3,)
        self.flow_vis = np.ones(shape, dtype=frame.dtype) * 255
        self.prev = self._prepare_frame(frame)

    def update(self, frame):
        """ Computes optical flow using a greyscale frame. """
        frame = self._prepare_frame(frame)
        self.flow = cv2.calcOpticalFlowFarneback(
                self.prev, frame, None, **self.fb_params)
        self.prev = frame
        self._compute_vis()

    def _compute_vis(self):
        """
        Computes an HSV representation of the optical flow and returns
        a BGR version of this.
        """
        mag, ang = cv2.cartToPolar(self.flow[...,0], self.flow[...,1])
        self.flow_vis[...,0] = ang*180/np.pi/2
        self.flow_vis[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        self.vis = cv2.cvtColor(self.flow_vis, cv2.COLOR_HSV2BGR)

    def _prepare_frame(self, frame):
        if frame.ndim > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

class MHI:
    """
    Motion History Image applies exponential smoothing to images on each call
    to update().

    Usage:
        mhi = MHI(frame.shape, np.uint8, 0.25)
        while True:
            motion = get_motion(get_frame())
            mhi.update(motion)
            mhi.mhi # returns expoential smoothed motion
    """

    def __init__(self, shape, dtype, alpha):
        """
        Args:
            shape: The shape of the MHI.
            dtype: The numpy type of the MHI.
            alpha: The percent [0,1] of the update frame to use in exponential
                smoothing.
        """
        self.mhi = np.zeros(shape, dtype=dtype)
        self.alpha = alpha
        self.dtype = dtype

    def update(self, frame):
        """ Apply exponential smoothing to the current MHI and frame. """
        self.mhi = (self.alpha*frame +
                (1-self.alpha)*self.mhi).astype(self.dtype)

