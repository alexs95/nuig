from generator import generative_model
from monitor import ModelMonitor
from keras.api import keras
from vae import get_vae
from wgan import WGAN
import tensorflow as tf


IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 2048
LATENT_SIZE = 128
EPOCHS = 300
VAE_PATH = ''
WGAN_PATH = ''


fashion_mnist = keras.datasets.fashion_mnist
(train_images, _), (test_images, _) = fashion_mnist.load_data()
print(f"Number of examples: {len(train_images)}")
print(f"Shape of the images in the dataset: {train_images.shape[1:]}")

# Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
train_images = train_images.reshape(train_images.shape[0], *IMG_SHAPE).astype("float32")
dataset = tf.data.Dataset.from_tensor_slices((train_images - 127.5) / 127.5)
dataset = dataset.shuffle(buffer_size=2048).batch(BATCH_SIZE)

generator = generative_model(IMG_SHAPE, LATENT_SIZE)

# VAE Method =================================================================
vae_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae, encoder = get_vae(IMG_SHAPE, LATENT_SIZE, generator)
vae.compile(vae_optimizer)
callback = ModelMonitor(VAE_PATH, generator, latent_size=LATENT_SIZE)
vae.fit(dataset.map(lambda x: (x, x)), callbacks=[callback], epochs=EPOCHS)


# WGAN Method =================================================================
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
wgan = WGAN(
    img_shape=IMG_SHAPE,
    latent_size=LATENT_SIZE,
    generator=generator,
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=generator_optimizer,
)
wgan.compile()
callback = ModelMonitor(VAE_PATH, generator, latent_size=LATENT_SIZE)
wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[callback])


# TODO
# Make model saving and loadable more easy -> load_epoch()
# Images to tensorflow datasets
# Implement model monitor
# Implement loader functions functions
# Update loss function in VAE to support larger sizes
# Update generative model to be able to handle different size models
# Update encoder to be able to handle different size models
