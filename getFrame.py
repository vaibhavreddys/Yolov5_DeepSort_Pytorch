import numpy as np
import cv2
import copy
import os

def n_th_frame(source, n=1):
    cap = cv2.VideoCapture(source)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    success, frame = cap.read()
    cap.release()
    print ("Total frames: {}".format(total_frames))
    if success:
        return frame
    return None