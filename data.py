""" Loads data from test_images and gets a person to label it. """

import os
import json
import argparse

import cv2 as cv

from icecream import ic

def get_number_from_string(string: str) -> int:
    """ Returns the number in a string. e.g. "test_123.jpg" -> 123 """
    return int("".join(c for c in string if c.isdigit()))

# Argument parser that takes a folder name
parser = argparse.ArgumentParser("Labels the data in the folder inside test_images given by --path")
parser.add_argument("--path", type=str, help="The folder inside test_images to label")
args = parser.parse_args()

images_dir_path = "test_images/" + args.path
data_path   = "test_data/" + args.path + "/data.json"

if not os.path.exists("test_data/" + args.path):
    os.mkdir("test_data/" + args.path)

# Get the images and data, and sort them by the number in the filename
images        = sorted(os.listdir(images_dir_path), key=get_number_from_string)

# get only the numbers from previous_data

try:
    with open(data_path, "r", encoding="utf-8") as f:
        user_center_data = json.load(f)
except FileNotFoundError:
    user_center_data = []

previous_data = [get_number_from_string(x["image_name"]) for x in user_center_data]

def save_click(event, x, y, flags, param):
    """ Saves the data when the user clicks. """
    if event == cv.EVENT_LBUTTONDBLCLK:
        ic(x, y, image)
        user_center_data.append({"center_x": x, "center_y": y, "image_name":image})
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(user_center_data, indent=4))


for image in images:
    if get_number_from_string(image) in previous_data:
        continue
    ic(image)
    img = cv.imread(f"{images_dir_path}/{image}")
    cv.imshow("image", img)
    cv.setMouseCallback("image", save_click)
    cv.waitKey(0)
    cv.destroyAllWindows()