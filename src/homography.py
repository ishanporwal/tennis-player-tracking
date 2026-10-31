import json
from pathlib import Path

import cv2
import numpy as np

def load_homography(calibration_path):
    # load a previously calculated homography matrix
    with open(calibration_path, "r") as file:
        calibration = json.load(file)
    return np.array(calibration["homography"], dtype=np.float64)

def pixel_to_court(pixel_x, pixel_y, homography):
    # convert image pixel coordinates to court coordinates in meters
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography)
    court_x, court_y = transformed[0][0]
    return float(court_x), float(court_y)