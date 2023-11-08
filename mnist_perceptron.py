import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

print("TensowFlow version: " + tf.__version__)
tf.config.set_visible_devices([], 'GPU')

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
	layers.Dense(10, activation="softmax")
])

model.compile(
	optimizer="rmsprop",
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"]
)

# Reshape the data into a 28*28 = 784 vector
train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

# The image data ranges from 0 - 255 integers, so this
# reformats it to floating point values between 0.0 and 1.0
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

# small batch size                                  --> very slow, averaging and applying weights for each sample, each epoch, good result
# batch_size == sample_size (train_images.shape[0]) --> very fast, averaging and applying weights only once per epoch, poor result
# moderate batch size, such as 128                  --> quick and approaches answer quickly, nice happy medium, good result
model.fit(train_images, train_labels, epochs=5, batch_size=128)
