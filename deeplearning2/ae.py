import keras
from keras import backend as K, layers
from keras.losses import mse
import tensorflow as tf
from encoder import encoder_block


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        # override the inherited .call(self, inputs) method
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))  # N(0, 1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  # N(mu, sigma)


def get_autoencoder(img_shape, latent_size, generator, variable=False):
    img_input = layers.Input(shape=img_shape)
    x = encoder_block(img_input)
    if variable:
        z_mean = layers.Dense(latent_size, name="z_mean")(x)
        z_log_var = layers.Dense(latent_size, name="z_log_var")(x)
        z = Sampling()((z_mean, z_log_var))
    else:
        z = layers.Dense(latent_size, name="z")(x)
    encoder = keras.models.Model(img_input, z, name="encoder")
    outputs = generator(z)
    ae = keras.models.Model(img_input, outputs, name="autoencoder")

    # Loss approach from
    # https://gist.github.com/tik0/6aa42cabb9cf9e21567c3deb309107b7
    reconstruction_loss = mse(img_input, outputs)  # xhat should match x
    reconstruction_loss = 784 * K.mean(reconstruction_loss)
    if variable:
        variable = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        ae.add_loss(variable)
        ae.add_metric(variable, name='kl_loss', aggregation='mean')

    # Set up our losses on the model, and create them as metrics too.
    # The Model's loss is the sum of the two losses.
    ae.add_loss(reconstruction_loss)
    ae.add_metric(reconstruction_loss, name='mse_loss', aggregation='mean')

    return ae, encoder


# Update model monitor
# Implement AE
# Check VAE code to see if there is anything wrong with using the encoder code.
# Use model monitor to display grid of images

