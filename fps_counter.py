""" Contains the FPSCounter class. """

from time import perf_counter

class FPSCounter:
    """Counts the number of frames per second. Generated entirely by Copilot."""

    # Singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(FPSCounter, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.start = perf_counter()
        self.frames = 0

    def count(self):
        """Called every frame. Prints the FPS if a second has passed."""
        self.frames += 1
        if perf_counter() - self.start > 1:
            print(f"FPS: {self.frames}")
            self.start = perf_counter()
            self.frames = 0