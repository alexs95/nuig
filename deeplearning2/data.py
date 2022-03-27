# Download dataset from here: https://www.kaggle.com/greg115/abstract-art
# Unzip into abstract_art_512 directory
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib.pyplot import figure


class AbstractArtDataset:
    # Source: https://www.kaggle.com/greg115/abstract-art
    def __init__(self, path, size=28, greyscale=False):
        self.df = None
        self._greyscale = greyscale
        self._size = size
        self._path = path
        self._images = []
        self._load()

    def visualise(self, inx=0, grid_size=4):
        grid = np.array(self._images[inx:inx + (grid_size * grid_size)])
        index, height, width, channels = grid.shape
        grid = (
            grid.reshape(grid_size, grid_size, self._size, self._size, channels)
                .swapaxes(1, 2)
                .reshape(self._size * grid_size, self._size * grid_size, channels)
        )
        figure(figsize=(4, 4))
        plt.imshow(grid, cmap='gray')
        plt.show()

    def _load(self):
        for path in list(glob.glob(self._path + "/*")):
            _, artist, _ = os.path.basename(path).split("_")
            img = cv2.imread(path, 1)
            img = cv2.resize(img, (self._size, self._size), interpolation=cv2.INTER_LINEAR)
            if self._greyscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.reshape((self._size, self._size, 1))
            self._images.append(img)
            self.df = np.array(self._images)


art = AbstractArtDataset("./abstract_art_512", 28, True)
art.visualise(20)
art.df.shape