import cv2
import numpy as np

class OpticalFlow:
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
        self.flow_vis = np.ones_like(frame) * 255
        self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def update(self, grey_frame):
        """ Computes optical flow using a greyscale frame """
        self.flow = cv2.calcOpticalFlowFarneback(
                self.prev, grey_frame, None, **self.fb_params)
        self.prev = grey_frame
        self._compute_vis()

    def _compute_vis(self):
        """
        Compute an HSV representation of the optical flow and return
        a BGR version of this.
        """
        mag, ang = cv2.cartToPolar(self.flow[...,0], self.flow[...,1])
        self.flow_vis[...,0] = ang*180/np.pi/2
        self.flow_vis[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        self.vis = cv2.cvtColor(self.flow_vis, cv2.COLOR_HSV2BGR)

class MHI:
    def __init__(self, shape, dtype, alpha=0.5):
        self.mhi = np.zeros(shape, dtype=dtype)
        self.alpha = alpha
        self.dtype = dtype

    def update(self, frame):
        self.mhi = (self.alpha*frame +
                (1-self.alpha)*self.mhi).astype(self.dtype)
