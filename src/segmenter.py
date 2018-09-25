"""
Methods for identifying and masking skin/hand from scene.
"""

import cv2
import numpy as np
import math

def hist_mask(img, hist, thresh=1, channels=[0,1], ranges=[0,180,0,256]):
    """
    Mask img according to a histogram using back projection.
    """
    backProj = cv2.calcBackProject([img], channels, hist, ranges, 1)
    if thresh is not None:
        ret, mask = cv2.threshold(backProj, thresh, 255, cv2.THRESH_BINARY)
    else:
        mask = backProj
    return mask

def get_roi_sample(cap, size=1/8):
    '''
    Draws a rectangle on the screen
    Returns the pixels in the region upon pressing space

    size is a fraction of the screen height
    '''
    _ret, frame = cap.read()
    h = frame.shape[0]
    w = frame.shape[1]

    size /= 2
    p0 = (int(w/2-h*size), int(h/2-h*size))
    p1 = (int(w/2+h*size), int(h/2+h*size))

    while cap.is_opened():
        _ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        #frame = cv2.GaussianBlur(frame, (13,13), 0)

        if not _ret:
            break

        frame_display = frame.copy()
        cv2.rectangle(frame_display, p0, p1, (0,255,0), 2)
        cv2.imshow('frame', frame_display)

        key = cv2.waitKey(5)
        if key == ord(' '):
            break

    box_sample = frame[p0[1]:p1[1], p0[0]:p1[0]]
    return box_sample

def mean_hist(samples, channels=[0,1], ranges=[0,180,0,256], bins=[32,32]):
    '''
    Gets a mean histogram of all the sample images.

    Inputs:
        samples is a list of sample images
        channels defines the channels to use
        ranges specifies the range of each channel
        num_bins is the number of bins for each channel

    Outputs:
        hist is the mean histogram
    '''
    hists = np.array([cv2.calcHist([sample], channels, None, bins, ranges) for sample in samples])
    hist = np.mean(hists, axis=0)
    return hist

def lowest_large_blob(mask, blob_thresh):
    ''' Return a mask of the lowest blob larger than blob_thresh '''
    # Find the lowest contour above the min size -- this should be a hand
    im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lowest_contour = None
    lowest_contour_height = math.inf
    lowest_contour_area = 0

    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if contour_area <= blob_thresh:
            continue

        m = cv2.moments(contour)
        contour_height = m['m10']/m['m00']

        if contour_height < lowest_contour_height:
            lowest_contour = i
            lowest_contour_height = contour_height
            lowest_contour_area = contour_area

    # Draw in hand blob
    mask = np.zeros_like(skin_mask)
    if lowest_contour:
        cv2.drawContours(mask, contours, lowest_contour, 255, -1)

    return mask

def largest_blob(mask, thresh=750):
    ''' Return a mask of the lowest blob larger than blob_thresh '''
    im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > thresh:
            return largest_contour

    return None

def contour_pos(contour):
    m = cv2.moments(contour)
    if m['m00'] == 0:
        cx = None
        cy = None
    else:
        cx = m['m10']/m['m00']
        cy = m['m01']/m['m00']

    return np.array((cx,cy))

def contour2mask(contour, shape):
    # Draw in hand blob
    mask = np.zeros_like(shape)
    if contour is not None:
        cv2.drawContours(mask, [contour], 0, 255, -1)

    return mask


#### DEPRECIATED ####

def hsv_mask(img, lower, upper):
    '''
    Given a BGR image, returns a mask of pixels within lower and upper HSV space

    mask = hsv_mask(img, lower, upper)

    input
        img: A BGR image
        lower: 3-tuple with hue, saturation, and value of lower cutoff of mask
        upper: 3-tuple with hue, saturation, and value of upper cutoff of mask

    output
        mask: binary mask of pixels with HSV values between lower and upper
    '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((img.shape[:2]), dtype=np.uint8)

    if lower[0] > upper[0]:
        # To account for hue wrapping around 180 to 0
        lower_middle = (180, upper[1], upper[2])
        upper_middle = (0,   lower[1], lower[2])

        mask_lower = cv2.inRange(hsv, lower, lower_middle)
        mask_upper = cv2.inRange(hsv, upper_middle, upper)

        mask = mask_lower + mask_upper

    else:
        mask = cv2.inRange(hsv, lower, upper)

    return mask

def get_hsv_range(imgs):
    '''
    min_hsv, max_hsv = get_hsv_range(imgs)

    Finds and returns the bounds of a set of hsv images.
    Shifts hue such that blue is around the 180/0 border
    so that skin range can be easily evaluated

    Input
        imgs: bgr images

    Output
        min_hsv: 3-tuple containing the mininum hue, saturation, and value found in img
        max_hsv: 3-tuple containing the maximum hue, saturation, and value found in img
    '''
    min_hsvs = []
    max_hsvs = []

    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = img[:,:,0]
        sat = img[:,:,1]
        val = img[:,:,2]

        hue = (hue + 90)%180 # shift skin hues away from the 180/0 border
        min_hue, max_hue, _, _ = cv2.minMaxLoc(hue)
        min_sat, max_sat, _, _ = cv2.minMaxLoc(sat)
        min_val, max_val, _, _ = cv2.minMaxLoc(val)

        min_hsvs.append((min_hue, min_sat, min_val))
        max_hsvs.append((max_hue, max_sat, max_val))

    # NOTE: these functions could be changed
    #... using minimax for now
    min_hsv = np.amin(np.array(min_hsvs), axis=0)
    max_hsv = np.amax(np.array(max_hsvs), axis=0)

    min_hsv[0] = (min_hsv[0] + 90)%180
    max_hsv[0] = (max_hsv[0] + 90)%180

    return min_hsv, max_hsv
