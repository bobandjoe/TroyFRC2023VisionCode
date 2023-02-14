""" Saves images to a folder. """

import os.path

import cv2 as cv

from icecream import ic

class ImageSaver:
    """Handles logic for saving files to the test_images folder."""

    PATH_FORMAT_STRING = "test_images/{}/test_{}.jpg"

    # Singleton
    def __new__(cls, *_args, **_kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(ImageSaver, cls).__new__(cls)
        return cls.instance

    def __init__(self, folder):
        ic(folder)
        self.folder = folder
        if not os.path.exists(f"test_images/{self.folder}"):
            os.makedirs(f"test_images/{self.folder}")
        self.file_count = 0
        while os.path.exists(self.path):
            self.file_count += 1

    def save(self, image):
        """Saves an image to the test_images folder."""
        cv.imwrite(self.path, image)
        self.file_count += 1

    @property
    def path(self):
        """Returns the path of the next file to be saved."""
        return self.PATH_FORMAT_STRING.format(self.folder, self.file_count)
