import cv2, math
import numpy as np
from ball_detection.config import BallDetectionConfig

WHITE: int = 255
BLACK: int = 0

def colour_to_gray(img: np.array) -> np.array:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

def hough_circle_transform(img: np.array, singular_output: bool = True) -> np.array:

    ## https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    img = colour_to_gray(img)

    width, height = img.shape
    centre_coordinates = [width//2, height//2]

    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, BallDetectionConfig.METHOD.value,
                               BallDetectionConfig.DP.value,
                               BallDetectionConfig.MINIMUM_DISTANCE.value,
                               param1=BallDetectionConfig.PARAM1.value,
                               param2=BallDetectionConfig.PARAM2.value,
                               minRadius=BallDetectionConfig.MINIMUM_RADIUS.value,
                               maxRadius=BallDetectionConfig.MAXIMUM_RADIUS.value
                               )
    
    circles = np.uint16(np.around(circles))
    
    if singular_output:

        contenders = circles[0,:]
        contender: int = 0

        max_distance: float = math.dist(centre_coordinates, [width, height])

        alpha: float = BallDetectionConfig.ALPHA.value
        beta: float = BallDetectionConfig.BETA.value

        scores: list = []

        for i in range(len(contenders)):

            ## colour metric
            h, w, r = contenders[i][0], contenders[i][1], contenders[i][2]
            colour_score = img[w][h]/WHITE

            ## distance metric
            contenders_coordinates = [w, h]
            distance_score = 1 - math.dist(contenders_coordinates, centre_coordinates)/max_distance

            distance_colour_product_contender = (alpha * colour_score + beta * distance_score)/(alpha + beta)
            print(f"Scores: {distance_score}, {colour_score}, {distance_colour_product_contender}")

            scores.append(distance_colour_product_contender)

        contender = np.argmax(np.array(scores))
        cv2.circle(cimg,(contenders[contender][0],contenders[contender][1]),contenders[contender][2],(0,255,0),2)
    
    else:

        for circle in circles[0,:]:
            cv2.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),2)


    return cimg