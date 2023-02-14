""" Tests a threshold against the images in test_images. """

import os.path
import json
import argparse

import numpy as np
from icecream import ic
import cv2 as cv

from objectdetection import get_object, Threshold
from contour_lib import get_contour_center

# The threshold to test
from objectdetection import printed_cone_threshold as test_threshold


# an argument parser with a --path argument
parser = argparse.ArgumentParser("Tests a threshold against the images in test_images.")
parser.add_argument("--path", type=str, help="The path to the test data file.")
args = parser.parse_args()
save_path = f"test_data/{args.path}/threshold.json"

DID_NOT_FIND_IMAGE_PENALTY = 1000

def make_random_change(threshold):
    """Makes a random change to the threshold for testing."""
    return Threshold(
        lower=threshold.lower + np.random.randint(-10, 10, 3),
        upper=threshold.upper + np.random.randint(-10, 10, 3),
    )


def generate_test_thresholds(base_threshold):
    """Generates a bunch of random thresholds to test."""
    for _ in range(5):
        yield make_random_change(base_threshold)


# Gets the names of all of the data files in order of their number
data_file_names = sorted(
    os.listdir("test_data"), 
    key=lambda x: int("".join(c for c in x if c.isdigit()))
)

image_data = []
images = []

for file_name in data_file_names:
    with open(f"test_data/{file_name}", "r", encoding="utf-8") as f:
        data = json.load(f)
        image_data.append(data)
        images.append(cv.imread(f"test_images/{data['image_name']}"))


def rate_threshold(threshold):
    """Returns how many pixels off the threshold gets on average."""
    amounts_off = []
    not_found = 0

    for datum, image in zip(image_data, images): 
        largest_object = get_object(image, threshold)
        if largest_object is None:
            not_found += 1
            continue
        center = get_contour_center(largest_object)
        amounts_off.append(abs(center[0] - datum["center_x"]) + abs(center[1] - datum["center_y"]))
    
    # return the median of the amounts off
    return np.median(amounts_off) + (DID_NOT_FIND_IMAGE_PENALTY * not_found) / len(images)



def main():
    """Tests the threshold against the test data."""
    best = test_threshold
    best_score = rate_threshold(best)
    ic(best_score)
    while True:
        for threshold in generate_test_thresholds(best):
            score = rate_threshold(threshold)
            if score < best_score:
                best = threshold
                best_score = score
                print(f"New best: {best_score}")
                ic(threshold)
                with open("best_threshold.json", "a", encoding="utf-8") as f:
                    json.dump("\n" + ic.format(threshold) + f" # {best_score}", f)
            ic(score)


if __name__ == "__main__":
    main()
