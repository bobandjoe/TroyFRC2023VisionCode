'''
LimeLight pipline for cube.
'''

import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

def runPipeline(image, llrobot):
    #constants for code
    lower_threshold = np.array([122, 92, 55])   # determined experimentally
    upper_threshold = np.array([131, 255, 255])   # determined experimentally

    #initialize variables in case they return nothing
    max_area_contour = np.array([[]])
    llpython = [0,0]

    # convert image to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    noise_reduction = cv.blur(thresh,(15,15))
    noise_reduction = cv.inRange(noise_reduction, 169, 255)
    
    # Find all the contours in the thresholded image with all contour-members of the "family" within the image and no approximation
    contours, _ = cv.findContours(noise_reduction, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Calculate max area by a contour, assuming the largest area --> closest object
    if not len(contours) == 0:
        max_area_contour = contours[0]
        for c in contours:
            area = cv.contourArea(c)
            # Ignore contours that are too small or too large
            if area < 1000 or 300000 < area:
                continue

            if area > cv.contourArea(max_area_contour):
                max_area_contour = c
        '''
        Wiki image moments: https://en.wikipedia.org/wiki/Image_moment
        
        '''
        M = cv.moments(max_area_contour)
        if not M['m00'] == 0 and len(max_area_contour) > 5:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            llpython[0] = cX
            llpython[1] = cY
            
            # draw the contour and center of the shape on the image
            cv.circle(image, (cX, cY), 7, (0, 0, 0), -1)

    return max_area_contour,image,llpython
  
