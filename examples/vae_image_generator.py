import tensorflow as tf
import numpy as np
import random
import keras
import os
import glob
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# https://www.kaggle.com/datasets/biaiscience/dogs-vs-cats
base_dir = Path(os.path.dirname(__file__))
dog_vs_cats_dir = base_dir / 'dogs_vs_cats'

def resize_to_square_with_center_crop(image, max_dim):
    # height, width, channels
    width = image.shape[1]
    height = image.shape[0]

    width_offset = None
    height_offset = None
    if width > max_dim:
        width_offset = int((width - max_dim) / 2)
    if height > max_dim:
        height_offset = int((height - max_dim) / 2)

    if width_offset is not None:
        if width_offset * 2 < width - max_dim:
            image = image[:, width_offset:-(width_offset+1)]
        else:
            image = image[:, width_offset:-width_offset]
    if height_offset is not None:
        if height_offset * 2 < height - max_dim:
            image = image[height_offset:-(height_offset+1), :]
        else:
            image = image[height_offset:-height_offset, :]

    return cv2.resize(image, (max_dim, max_dim), interpolation=cv2.INTER_AREA)

class Sampler(keras.layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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
    x = keras.layers.Flatten()(x)
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
    decoder_outputs = keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    decoder_model = keras.Model(latent_inputs, decoder_outputs, name='decoder')

    return decoder_model

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
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
        with tf.GradientTape() as tape:
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
    max_images = 10000
    latent_dim = 2
    max_size = 256
    num_channels = 3
    input_img_shape = (max_size,max_size,num_channels)
    load_model = False
    modelckpt = str(base_dir / 'vae_weights')

    # get some fun images
    all_cat_imgs = glob.glob(str(dog_vs_cats_dir / 'train/cat.*.jpg'))
    all_dog_imgs = glob.glob(str(dog_vs_cats_dir / 'train/dog.*.jpg'))
    random.shuffle(all_cat_imgs)
    random.shuffle(all_dog_imgs)
    all_dog_imgs = all_dog_imgs[:int(max_images / 2)]
    all_cat_imgs = all_cat_imgs[:int(max_images / 2)]
    all_cat_imgs = [resize_to_square_with_center_crop(cv2.imread(img), max_size) for img in all_cat_imgs]
    all_dog_imgs = [resize_to_square_with_center_crop(cv2.imread(img), max_size) for img in all_dog_imgs]
    all_imgs = all_cat_imgs + all_dog_imgs
    all_imgs = np.asarray(all_imgs, dtype='float32') / 255

    # encoder
    encoder_model = get_encoder(input_img_shape=input_img_shape, latent_dim=latent_dim)
    encoder_final_conv_shape = encoder_model.get_layer(name='encoder_last_conv_layer').output_shape[1:]
    print(encoder_model.summary())

    # decoder
    decoder_model = get_decoder(encoder_final_conv_shape=encoder_final_conv_shape, latent_dim=latent_dim)
    print(decoder_model.summary())

    vae = VAE(encoder_model, decoder_model)
    vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

    if not load_model:
        try:
            vae.fit(
                all_imgs,
                epochs=5000,
                batch_size=128,
                callbacks=[
                    # keras.callbacks.EarlyStopping(
                    #     monitor='kl_loss',
                    #     patience=50,
                    #     restore_best_weights=True
                    # ),
                    keras.callbacks.TensorBoard(
                        log_dir=str(base_dir / 'vae_tensorbaord_logs')
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='kl_loss',
                        patience=20,
                        factor=0.5
                    )
                ]
            )
        except KeyboardInterrupt:
            pass
        vae.save_weights(modelckpt)
    else:
        vae.load_weights(modelckpt)

    n = 4
    img_to_sample_size = max_size
    figure = np.zeros((img_to_sample_size * n, img_to_sample_size * n, num_channels))

    grid_x = np.linspace(-1, 1, n)
    grid_y = np.linspace(-1, 1, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            img = x_decoded[0].reshape(img_to_sample_size, img_to_sample_size, num_channels)
            figure[
                i * img_to_sample_size : (i + 1) * img_to_sample_size,
                j * img_to_sample_size : (j + 1) * img_to_sample_size,
            ] = img

    plt.figure(figsize=(15, 15))
    start_range = img_to_sample_size // 2
    end_range = n * img_to_sample_size + start_range
    pixel_range = np.arange(start_range, end_range, img_to_sample_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.axis("off")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()