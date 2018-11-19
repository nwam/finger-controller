import numpy as np
import cv2
import dataset

def prediction_frame(prediction, h, w):
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    for i, probability in enumerate(prediction):
        gesture = dataset.id_to_gesture[i]
        height = int(h*i/len(prediction)+20)
        cv2.putText(frame, gesture, (2, height),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (200,120,0))
        cv2.rectangle(frame,
                (60, height-10), (60+int(probability*(w-60)), height),
                (0,int(255*probability), int(255*(1-probability))), cv2.FILLED)

    return frame

def mhb_frame(mhb, h, w):
    mhb_frame = cv2.cvtColor((mhb.mhi.mhi*25).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    mhb_frame = cv2.resize(mhb_frame, (w,h))
    return mhb_frame

def put_hpos_text(frame, h_pos, h_pos_thresh, pos=(2,25)):
    h_pos_color = (120,120,120)
    if h_pos > h_pos_thresh:
        h_pos_color = (120,120,0)
    elif h_pos < h_pos_thresh:
        h_pos_color = (0,75,200)

    cv2.putText(frame, str(int(h_pos)),
            pos, cv2.FONT_HERSHEY_DUPLEX, 0.5, h_pos_color)

def hpos_color(hpos, max_hpos, hpos_thresh, shape, mid_margin=3,
        lower_bound=2, upper_margin=5):
    hue_best = 60
    hue_worst = 120
    upper_thresh = hpos_thresh + mid_margin
    lower_thresh = hpos_thresh - mid_margin
    upper_bound = max_hpos - upper_margin

    percent_to_edge = 0
    if hpos > upper_bound or hpos < lower_bound:
        percent_to_edge = 1
    elif hpos > upper_thresh:
        percent_to_edge = (hpos - upper_thresh) / (upper_bound - upper_thresh)
    elif hpos < lower_thresh:
        percent_to_edge = (lower_thresh - hpos) / (lower_thresh - lower_bound)
    hue = hue_worst + (hue_best - hue_worst)*(1-percent_to_edge)**1

    frame = np.ones(shape + (3,), dtype=np.uint8)*255
    frame[:,:,0] = hue
    return cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
