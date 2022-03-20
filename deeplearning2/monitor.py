import numpy as np
from keras.api import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import os


class ModelMonitor(keras.callbacks.Callback):
    def __init__(self, generator, img_shape, latent_size, path, size):
        super().__init__()
        self.size = size
        self.generator = generator
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.path = path
        os.makedirs(path, exist_ok=True)
        os.makedirs(path + '/epochs', exist_ok=True)
        os.makedirs(path + '/checkpoints', exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.size * self.size, self.latent_size))
        generated_images = self.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5
        grid = np.array(generated_images)
        grid = (
            grid.reshape(self.size, self.size, self.img_shape[0], self.img_shape[0])
                .swapaxes(1, 2)
                .reshape(self.img_shape[0] * self.size, self.img_shape[0] * self.size)
        )
        plt.figure(figsize=(8, 8))
        plt.imsave('{}/epochs/{}.png'.format(self.path, epoch), grid, cmap='gray', vmin=0, vmax=255)
        plt.imshow(grid, cmap='gray', vmin=0, vmax=255)
        plt.show()