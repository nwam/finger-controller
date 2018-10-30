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
