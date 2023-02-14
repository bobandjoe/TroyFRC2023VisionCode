""" 
Detects objects in the camera stream. 
The main loop could work better, but I don't want to work on finding another threshold.

BTW: I made a docstring for all of the functions, so you can hover over them to see what they do.

For reference:

HSV:           https://www.geeksforgeeks.org/color-spaces-in-opencv-python/
Thresholding:  https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
Contours:      https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
"""

import argparse

from icecream import ic
import numpy as np
import cv2 as cv

from fps_counter import FPSCounter
from threshold_lib import load_threshold, Threshold

from contour_lib import (
    get_maximum_contour,
    get_contour_center,
    get_farthest_point,
    get_angle,
    get_sides,
    draw_contour_points,
)


def infinite_frame_stream():
    """
    Returns a generator that yields frames from the webcam.
    Exits the program if the user presses 'q' and waits between frames.
    Saves the frame if the user presses 's' and the save argument is True.
    """
    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        if frame is None:
            print("Error reading frame")
            exit(1)
        # Exit if the user presses 'q'
        key = cv.waitKey(1)
        if key & 0xFF == ord("q"):
            exit(0)
        yield frame


def get_object(image, threshold):
    """Finds the largest object in the image that is within the threshold."""
    # Blurs the image to reduce noise
    processed = cv.medianBlur(image, 5)

    # Converts the image to the Hue, Saturation, Value color space, which makes it easier to detect objects of a certain color
    processed = cv.cvtColor(processed, cv.COLOR_BGR2HSV)

    # Converts the image to a binary image, where the object of interest (between the lower and upper thresholds) is white and the rest is black
    processed = cv.inRange(processed, threshold.lower, threshold.upper)

    # Finds objects in the new binary image, which should be easy because the image is only black and white.
    contours, _ = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Gets the largest contour in the image, hopefully the correct object if the threshold is set well.
    largest_object = get_maximum_contour(contours)
    return largest_object


def process_object(threshold):
    """
    Given a threshold, this function will process the camera stream
    and detect the object of interest.

    Args:
        threshold (Threshold): An object with `lower` and `upper` attributes, which are HSV values.
    """
    for frame in infinite_frame_stream():
        largest_object = get_object(frame, threshold)

        if largest_object is None:
            FPSCounter().count()
            continue

        # Gets the center of the largest contour
        object_center = get_contour_center(largest_object)
        farthest_point = get_farthest_point(largest_object, object_center)

        object_angle = get_angle(object_center, farthest_point)

        sides = get_sides(largest_object)
        draw_contour_points(frame, largest_object)
        draw_contour_points(frame, sides, (0, 0, 255))

        # Draws stuff on the frame
        cv.circle(frame, object_center, 5, (0, 255, 0), -1)
        cv.circle(frame, farthest_point, 5, (0, 0, 255), -1)
        cv.line(frame, object_center, farthest_point, (255, 0, 0), 2)
        show_text(frame, f"Angle: {object_angle:.1f}")

        
        cv.imshow("frame", frame)
        FPSCounter().count()


def show_text(frame, text):
    """Shows text on the top-left of the frame. Made another function because this looks bad in the main loop."""
    cv.putText(
        frame,
        text,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )


cube_threshold = Threshold(
    lower=np.array([118, 87, 86]),
    upper=np.array([133, 255, 255]),
)

cone_threshold = Threshold(
    lower=np.array([18, 44, 101]),
    upper=np.array([31, 232, 255]),
)

# The threshold for the printed picture of the cone I was using for testing
printed_cone_threshold = Threshold(
    lower=np.array([ 10,  72, 136]), 
    upper=np.array([ 63, 176, 204]),
)


def main():
    """The main function. Detects the cone in the camera stream."""
    
    # gets the file path from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    args = parser.parse_args()
    
    if args.path is None:
        raise ValueError("Please provide a path to load the threshold from.")
    
    threshold = load_threshold(args.path)
    
    process_object(threshold=threshold)


if __name__ == "__main__":
    main()
