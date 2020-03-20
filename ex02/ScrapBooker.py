import numpy as np
from ex01.ImageProcessor import ImageProcessor


class ScrapBooker:
    @staticmethod
    def crop(array, dimensions, position=(0, 0)):
        return np.copy(array[position[0]:position[0] + dimensions[0], position[1]:position[1] + dimensions[1]])

    @staticmethod
    def thin(array, n, axis=0):
        if axis == 0:
            return np.copy(array[:, :array.shape[1] - n])
        else:
            return np.copy(array[:array.shape[1] - n])

    @staticmethod
    def juxtapose(array, n, axis=0):
        if axis == 0:
            return np.tile(array, (n, 1, 1) if array.ndim == 3 else n)

        else:
            return np.tile(array, (n, 1))

    @staticmethod
    def mosaic(array, dimensions):
        return ScrapBooker.juxtapose(ScrapBooker.juxtapose(array, dimensions[0]), dimensions[1], axis=1)


if __name__ == "__main__":
    sb = ScrapBooker

    arr = np.eye(5)
    # [1. 0. 0. 0. 0.]
    # [0. 1. 0. 0. 0.]
    # [0. 0. 1. 0. 0.]
    # [0. 0. 0. 1. 0.]
    # [0. 0. 0. 0. 1.]

    # Crop
    print(sb.crop(arr, (4, 2), (1, 3)), end="\n\n")

    # Thin
    print(sb.thin(arr, 2), end="\n\n")
    print(sb.thin(arr, 4, axis=1), end="\n\n")

    # Juxtapose
    print(sb.juxtapose(arr, 3), end="\n\n")
    print(sb.juxtapose(arr, 2, axis=1), end="\n\n")

    # Mozaic
    print(sb.mosaic(arr, (3, 5)))


# ~~~~ Test with image ~~~~
# (Execute with `d03: python -m ex02.ScrapBooker`)
def test_with_img():
    ip = ImageProcessor
    sb = ScrapBooker

    img = ip.load("./ex01/daruma.jpg")
    ip.display(img)

    # Crop
    ip.display(sb.crop(img, (90, 90), (70, 70)))

    # Thin
    ip.display(sb.thin(img, 50))
    ip.display(sb.thin(img, 50, axis=1))

    # Juxtapose
    ip.display(sb.juxtapose(img, 3))
    ip.display(sb.juxtapose(img, 10, axis=1))

    # Mozaic
    ip.display(sb.mosaic(img, (3, 5)))


if __name__ == "__main__":
    test_with_img()
    pass
