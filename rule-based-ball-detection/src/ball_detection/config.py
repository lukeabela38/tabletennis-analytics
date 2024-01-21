import cv2
from enum import Enum

class BallDetectionConfig(Enum):
    DP = 1.5 # 1.5
    MINIMUM_DISTANCE = 20 # 20
    PARAM1 = 300 # 300
    PARAM2 = 0.90 # 0.90
    MINIMUM_RADIUS = 0 # 0
    MAXIMUM_RADIUS = 100 # 100
    METHOD = cv2.HOUGH_GRADIENT_ALT
    ALPHA = 1.5 # weighting for colour score
    BETA = 1 # weighting for distance score