"""
This module contains classes that perform computer vision alogirhtms.
"""
import cv2
import numpy as np
import scipy.signal

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
            flow.vis # returns a visualization of optical flow
    """
    # Parameters for farneback optical flow
    fb_params = dict(
        pyr_scale = 0.5,
        levels = 5,
        winsize = 7,
        iterations = 5,
        poly_n = 5,
        poly_sigma = 1.2,
        flags = 0)

    def __init__(self, frame):
        """ Initializes optical flow using the first frame of the sequence. """
        shape = frame.shape
        if frame.ndim == 2:
            shape = shape + (3,)
        self.hsv = np.ones(shape, dtype=frame.dtype) * 255
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
        self.mag, self.ang = cv2.cartToPolar(self.flow[...,0], self.flow[...,1])
        self.hsv[...,0] = self.ang*180/np.pi/2
        self.hsv[...,2] = cv2.normalize(self.mag, None, 0, 255, cv2.NORM_MINMAX)
        self.vis = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

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

class MHB:
    """
    Max Horizontal Blob finds the max horizontal blob of a kernel in an
    optical flow calculation.
    """
    def __init__(self, cnn_input, kernel):
        """
        Args:
            cnn_input: CnnInput object
            kernel: numpy array
        """
        self.clip = cnn_input.edge_clip
        self.flow = cnn_input.flow
        self.kernel = kernel
        self.hmag = np.ones([v-2*self.clip for v in self.flow.hsv.shape[:2]])
        self.mhi = MHI(self.hmag.shape, np.float, cnn_input.mhis[0].alpha)

    def compute(self):
        self.hmag = self.flow.mag*np.cos(self.flow.ang)**2
        self.hmag = self.hmag[self.clip:-self.clip, self.clip:-self.clip]
        convd = scipy.signal.convolve2d(self.hmag, self.kernel, mode='same')
        self.mhi.update(convd)
        best_location = np.unravel_index(
                np.argmax(self.mhi.mhi), self.mhi.mhi.shape)
        return self.mhi.mhi[best_location], best_location
