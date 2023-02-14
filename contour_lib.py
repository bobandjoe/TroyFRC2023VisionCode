""" Useful functions for working with points and contours. """

import numpy as np
import cv2 as cv


def filter_contours_by_sides(contours, min_sides, max_sides):
    return filter(
        lambda contour: min_sides <= get_num_sides(contour) <= max_sides,
        contours
    )


def filter_contours(contours):
    """ Returns a list of contours that are within the size range. """
    return [
        contour for contour in contours \
        if 1_000 < cv.contourArea(contour) < 300_000
    ]


def get_maximum_contour(contours):
    """ Returns the contour with the largest area in the list of contours or None if there are no contours. """
    contours = filter_contours(contours)
    if len(contours) == 0:
        return None
    return max(contours, key=cv.contourArea)


def get_contour_center(contour):
    """ Returns the center of the contour. """
    moments = cv.moments(contour)
    try:
        # No one knows what this does, but it works.
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    except ZeroDivisionError:
        center_x = 0
        center_y = 0
    return np.array([center_x, center_y])


def distance_squared(point1, point2):
    """ Returns the distance between two points, but squared for performance because we only need the distance for comparison. """
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def get_angle(point1, point2):
    """ Returns the angle from the first point to the second point in degrees. """
    degrees = np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
    while degrees < 0:
        degrees += 360
    # Make the degrees negative to make the angle counterclockwise
    # Resulting in the following:
    #        90
    #    180    0
    #       270
    return -degrees + 360


def get_farthest_point(contour, center=None):
    """ Returns the farthest point from the center of the contour. """
    if center is None:
        center = get_contour_center(contour)
    return max(
        (point[0] for point in contour),
        key=lambda point: distance_squared(center, point),
    )


def get_orientation(contour, center=None):
    """ Returns the angle of the farthest point from the center of the contour. """
    farthest_point = get_farthest_point(contour, center)
    return get_angle(center, farthest_point)


def draw_contour_points(frame, contour, color=(255, 255, 255), radius=3):
    """ Draws the points of the contour on the frame. """
    for point in contour:
        cv.circle(frame, tuple(point[0]), radius, color, -1)


def get_sides(contour, accuracy=0.02):
    """ Returns the approximate number of sides of the contour. """
    return cv.approxPolyDP(contour, accuracy * cv.arcLength(contour, True), True)

