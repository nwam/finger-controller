"""
Input capture methods for finger-people.

This module allows us to select various methods of input for finger-people.
The different types are camera, video, and web. Camera and web (usually a
video stream from a device on the same LAN) are used for actual finger-people
use while video is useful for tests since you can retest with the same video.

Usage:
# First choose a capture source and type
# Camera
cap source = 0 # or another openCV camera ID
cap_type = CapType.CAMERA

# Video: Local File
cap_source = 'path/to/video/file'
cap_type = CapType.VIDEO

# Video: Droidcam
cap_source = 'http://user:machine@192.168.0.20:4747/mjpegfeed?320x240'
cap_type = CapType.VIDEO

# Video: IpWebcam
cap_source = 'http://192.168.0.34:8080/video'
cap_type = CapType.VIDEO

# Web (Depreciated)
cap_source = 'http://192.168.0.12:8080/shot.jpg'
cap_type = CapType.WEB


# Then create and use the Capture object
cap = Capture(cap_source, cap_type)

while cap.is_opened():
    frame = cap.read()
    # Do something with frame

cap.kill()
"""

import cv2
import enum
import urllib

class CapType(enum.Enum):
    CAMERA = 0
    VIDEO = 1
    WEB = 2

class Capture:
    def __init__(self, source, source_type):
        if type(source) == int and source_type == CapType.CAMERA:
            cap = cv2.VideoCapture(source)
            w = 180
            h = w * 3/4
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            #cap.set(cv2.CAP_PROP_FPS, 60)

        elif source_type == CapType.VIDEO:
            cap = cv2.VideoCapture(source)
            #cap.set(cv2.CAP_PROP_FPS, 30)

        elif source_type == CapType.WEB:
            cap = WebCapture(source)

        else:
            raise ValueError('No valid source and source type were provided')

        self.cap = cap

    def read(self):
        return self.cap.read()

    def is_opened(self):
        return self.cap.isOpened()

    def kill(self):
        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


class WebCapture:
    def __init__(self, url):
        self.url = url

    def read(self):
        try:
            img_resp = urllib.request.urlopen(self.url)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(img_np, -1)
            return True, img
        except:
            return False, None

    def isOpened(self):
        return True

    def release(self):
        pass
