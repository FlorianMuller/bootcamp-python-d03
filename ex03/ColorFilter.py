# I M P O R T A N T: Execute from root with `python -m ex03.ColorFilter`
import numpy as np
from ex01.ImageProcessor import ImageProcessor


class ColorFilter:
    @staticmethod
    def invert(array):
        return 1 - array

    @staticmethod
    def to_blue(array):
        blue_array = np.zeros(array.shape, dtype=array.dtype)

        for line, blue_line in zip(array, blue_array):
            for pixel, blue_pixel in zip(line, blue_line):
                blue_pixel[2] = pixel[2]

        return blue_array

    @staticmethod
    def to_green(array):
        return array


if __name__ == "__main__":
    ip = ImageProcessor
    cf = ColorFilter

    img = ip.load("./ex03/Igor.jpg")
    # img = ip.load("./ex03/elon.jpg")
    ip.display(img)

    # Invert
    ip.display(cf.invert(img))

    # To blue
    ip.display(cf.to_blue(img))
