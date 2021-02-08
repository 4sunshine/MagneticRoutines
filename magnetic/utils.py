import cv2
import numpy as np


def put_text(image, text, color=(255, 255, 255), origin=(0, 1)):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    h, w = np.shape(image)[:2]
    org = (int(origin[0] * w), int(origin[1] * h))
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    return cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)