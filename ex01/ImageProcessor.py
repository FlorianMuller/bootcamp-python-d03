import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImageProcessor:
    @staticmethod
    def load(path):
        img = mpimg.imread(path)
        print(f"Loading image of dimensions {img.shape[0]} x {img.shape[1]}")
        return img

    @staticmethod
    def display(array):
        plt.rcParams["toolbar"] = 'None'
        plt.axis('off')

        plt.imshow(array)
        plt.show()


if __name__ == "__main__":
    imp = ImageProcessor()

    arr = imp.load("./42AI.png")
    # arr = imp.load("./daruma.jpg")
    print(repr(arr))

    imp.display(arr)
