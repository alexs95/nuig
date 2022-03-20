import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib.pyplot import figure


class AbstractArtDataset:
    # Source: https://www.kaggle.com/greg115/abstract-art
    def __init__(self, path, size=128):
        self.size = size
        self.path = path
        self.images = []
        self.artists = []

    def visualise(self, inx=0, size=4):
        grid = np.array(self.images[inx:inx + (size * size)])
        index, height, width, intensity = grid.shape
        grid = (
            grid.reshape(size, size, self.size, self.size, intensity)
                .swapaxes(1, 2)
                .reshape(self.size * size, self.size * size, intensity)
        )
        figure(figsize=(12, 12))
        plt.imshow(grid)
        plt.show()

    def load(self):
        for path in list(glob.glob(self.path + "/*"))[:20]:
            _, artist, _ = os.path.basename(path).split("_")
            img = cv2.imread(path, 1)
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            self.images.append(img)
            self.artists.append(artist)
