'''
LLPython = LimeLight Python array used to relay information to the robot (through limelight) with Network Tables.
LimeLight docs: https://docs.limelightvision.io/en/latest/
'''

import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

invert_angle = False
previous_angle = None

def runPipeline(image, llrobot):
    global invert_angle
    global previous_angle
    #constants for code
    lower_threshold = np.array([15, 0, 120])   # determined experimentally
    upper_threshold = np.array([31, 255, 255])   # determined experimentally

    #initialize variables in case they return nothing
    max_area_contour = np.array([[]])
    llpython = [0,0,0,0]

    # convert image to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # convert the hsv image to binary image + noise reduction
    thresh = cv.inRange(hsv, lower_threshold, upper_threshold)
    noise_reduction = cv.erode(thresh, np.ones((10, 10), np.uint8), iterations = 1)
    noise_reduction = cv.blur(thresh,(15,15))
    noise_reduction = cv.inRange(noise_reduction, 169, 255)
    
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(noise_reduction, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    if not len(contours) == 0:
        max_area_contour = contours[0]
        for c in contours:
            area = cv.contourArea(c)
            # Ignore contours that are too small or too large
            if area < 1000 or 300000 < area:
                continue

            if area > cv.contourArea(max_area_contour):
                max_area_contour = c

        print("max contour area", cv.contourArea(max_area_contour))
        M = cv.moments(max_area_contour)
        '''
        OpenCV moments/contours docs: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        Wikipedia Image Moment info page: https://en.wikipedia.org/wiki/Image_moment
        '''
        # find the centroid of the image with calculations of image moments
        if not M['m00'] == 0 and len(max_area_contour) > 5: #First validate the moment
            cX = int(M["m10"] / M["m00"]) #Then calculate X of centroid
            cY = int(M["m01"] / M["m00"]) #Y of centroid
            llpython[1] = cX
            llpython[2] = cY
            llpython[3] = M['m00'] # area
            
            # draw the contour and center of the shape on the image
            cv.circle(image, (cX, cY), 7, (0, 0, 0), -1)

            leftmost = tuple(max_area_contour[max_area_contour[:,:,0].argmin()][0])
            rightmost = tuple(max_area_contour[max_area_contour[:,:,0].argmax()][0])
            topmost = tuple(max_area_contour[max_area_contour[:,:,1].argmin()][0])
            bottommost = tuple(max_area_contour[max_area_contour[:,:,1].argmax()][0])

            cv.circle(image, leftmost, 5, (0, 0, 0), 2)
            cv.circle(image, rightmost, 5, (0, 0, 0), 2)
            cv.circle(image, topmost, 5, (0, 0, 0), 2)
            cv.circle(image, bottommost, 5, (0, 0, 0), 2)

            (x,y),(MA,ma),angle = cv.fitEllipse(max_area_contour)

            count = 0
            if leftmost[1] > cY:
                count += 1
            if rightmost[1] > cY:
                count += 1
            if topmost[1] > cY:
                count += 1
            if bottommost[1] > cY:
                count += 1
                
            if previous_angle is not None:
                if angle - previous_angle > 90: # jumping from 0 to 180 degrees
                    if count > 2: # cone tip pointing down
                        invert_angle = True
                    else: # cone tip pointing ups
                        invert_angle = False
                elif angle - previous_angle < -90: # jumping from 180 to 0 degrees
                    if count > 2: # cone tip pointing down
                        invert_angle = False
                    else: # cone tip pointing up
                        invert_angle = True

            previous_angle = angle

            if invert_angle:
                angle += 180

            image = cv.putText(image, str(round(angle)), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)    
            llpython[0] = angle

    return max_area_contour,image,llpython

cap = cv.VideoCapture(cv.CAP_DSHOW)
while True:
    _, frame = cap.read()
    if frame is None:
        print("Error reading frame")
        exit(1)
    
    mac, img, _ = runPipeline(frame, None)

    cv.imshow("i", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

