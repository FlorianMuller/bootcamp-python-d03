# I M P O R T A N T: Execute from sub directory with `python -m ex03.ColorFilter`
import numpy as np
from ex01.ImageProcessor import ImageProcessor


class ColorFilter:
    @staticmethod
    def invert(array):
        return 1 - array

    @staticmethod
    def to_blue(array):
        blue_array = np.zeros(array.shape, dtype=array.dtype)
        blue_array[:, :, 2] = array[:, :, 2]
        return blue_array

    @staticmethod
    def to_green(array):
        return array * [0, 1, 0]

    @staticmethod
    def to_red(array):
        return array - ColorFilter.to_blue(array) - ColorFilter.to_green(array)

    @staticmethod
    def celluloid(array, treshold=4):
        treshold = treshold if treshold > 0 else 1
        values = np.linspace(0, 255, num=treshold, dtype=int)
        values_limit = []
        for i, val in enumerate(values):
            left_dist = (val - values[i - 1]) / 2 if i > 0 else 0
            right_dist = (values[i + 1] - val) / 2 if i < len(values) - 1 else 1
            values_limit.append((val, (val - left_dist, val + right_dist)))

        def cel_shading(pix):
            for val, limit in values_limit:
                if limit[0] <= pix < limit[1]:
                    return val
        cel_shading = np.vectorize(cel_shading)
        return cel_shading(array)

    @staticmethod
    def to_grayscale(array, filter="w"):
        if filter in ["w", "‘weighted’"]:
            new_array = (array * [0.299, 0.587, 0.114]).sum(axis=2, keepdims=True).astype(int)
            return np.tile(new_array, (1, 1, 3))
        elif filter in ["m", "mean"]:
            new_array = (array.sum(axis=2, keepdims=True) / 3).astype(int)
            return np.broadcast_to(new_array, (*new_array.shape[:2], 3))


if __name__ == "__main__":
    ip = ImageProcessor
    cf = ColorFilter

    img = ip.load("./ex03/Igor.jpg")
    # img = ip.load("./ex03/elon.jpg")

    # Invert
    ip.display(cf.invert(img))

    # To blue
    ip.display(cf.to_blue(img))

    # To green
    ip.display(cf.to_green(img))

    # To red
    ip.display(cf.to_red(img))

    # Celluloid
    for i in range(2, 7):
        print(f"Treshold {i}")
        ip.display(cf.celluloid(img, treshold=i))

    # Greyscale
    print("w")
    ip.display(cf.to_grayscale(img))
    print("m")
    ip.display(cf.to_grayscale(img, filter="m"))
