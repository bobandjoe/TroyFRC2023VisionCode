""" Saves and loads the threshold to a file."""

import json
import os.path
import numpy as np
from dataclasses import dataclass

import cv2 as cv

from icecream import ic

HUE_RANGE = 10

@dataclass
class Threshold:
    """
    Holds the lower and upper threshold color values for the objects.
    I like it this way but we could replace it with just passing static values to `process_object`.
    """

    lower: np.ndarray
    upper: np.ndarray

    def to_json(self):
        """Converts the threshold to a JSON serializable object."""
        return {"lower": self.lower.tolist(), "upper": self.upper.tolist()}

    @classmethod
    def from_json(cls, data):
        """Converts a JSON object to a threshold."""
        return cls(lower=np.array(data["lower"]), upper=np.array(data["upper"]))


def generate_starting_threshold(metadatum, path):
    """Looks at the image's center and generates a threshold around it"""
    image = cv.imread(f"test_images/{path}/{metadatum['image_name']}")
    color = image[metadatum["center_y"], metadatum["center_x"]]
    return Threshold(
        lower=np.array([color[0] - HUE_RANGE, 0, 0]),
        upper=np.array([color[0] + HUE_RANGE, 255, 255]),
    )


def load_threshold(path, image_metadata=None):
    """Returns the initial threshold to test."""
    save_path = f"test_data/{path}/threshold.json"
    if os.path.isfile(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            threshold = Threshold.from_json(json.load(f))
    else:
        if image_metadata is None:
            raise ValueError("Threshold file does not exist.")
        else:
            threshold = generate_starting_threshold(image_metadata[0], path)
    ic(threshold)
    return threshold


def save_threshold(threshold, path):
    """Saves the threshold to a file."""
    save_path = f"test_data/{path}/threshold.json"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(threshold.to_json(), f)  
    except PermissionError as error:
        print(error)
        save_threshold(threshold, path)
