import cv2
from skvideo import io
import numpy as np
import cv2
import skvideo

video = skvideo.io.vreader("/home/paul/test.mp4")

for frame in video:
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lab[:,:,1] = 128
    lab[:,:,2] = 128
    lab2 = cv2.normalize(lab, None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    lab2 *= 255
    lab3 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    cv2.imshow('frame',lab3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break