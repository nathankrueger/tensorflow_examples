import tensorflow as tf
import numpy as np
import keras
import os
from pathlib import Path

base_dir = Path(os.path.dirname(__file__))

class Sampler(keras.layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal.shape(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) + epsilon

def get_encoder(
    input_img_shape,
    latent_dim: int,
    conv_layer_sizes: list[int] = [32, 64, 128],
) -> keras.Model:
    x = encoder_inputs = keras.Input(shape=input_img_shape)
    for idx, layer_size in enumerate(conv_layer_sizes):
        if idx == len(conv_layer_sizes) - 1:
            name = 'encoder_last_conv_layer'
        else:
            name = None
        x = keras.layers.Conv2D(layer_size, 3, activation='relu', strides=2, padding='same', name=name)(x)
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)
    encoder_model = keras.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')

    return encoder_model

def get_decoder(
    encoder_final_conv_shape,
    latent_dim: int,
    conv_layer_sizes: list[int] = [128, 64, 32],
) -> keras.Model:
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = keras.layers.Dense(
        encoder_final_conv_shape[0] * encoder_final_conv_shape[1] * encoder_final_conv_shape[2],
        activation='relu'
    )(latent_inputs)
    x = keras.layers.Reshape(encoder_final_conv_shape)(x)
    for layer_size in conv_layer_sizes:
        x = keras.layers.Conv2DTranspose(layer_size, 3, activation='relu', strides=2, padding='same')(x)
    decoder_outputs = keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    decoder_model = keras.Model(latent_inputs, decoder_outputs, name='decoder')

    return decoder_model

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
    
    def train_step(self, data):
        with tf.GradientTape as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'total_loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }

if __name__ == '__main__':
    latent_dim = 2
    input_img_shape = (300,300,3)

    # encoder
    encoder_model = get_encoder(input_img_shape=input_img_shape, latent_dim=latent_dim)
    encoder_final_conv_shape = encoder_model.get_layer(name='encoder_last_conv_layer').output_shape[1:]

    # decoder
    decoder_model = get_decoder(encoder_final_conv_shape=encoder_final_conv_shape, latent_dim=latent_dim)

    vae = VAE(encoder_model, decoder_model)
    vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
    