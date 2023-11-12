import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("TensowFlow version: " + tf.__version__)
tf.config.set_visible_devices([], 'GPU')

num_epochs = 5
batch_size = 128
layer_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the data into a 28*28 = 784 vector
train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

# The image data ranges from 0 - 255 integers, so this
# reformats it to floating point values between 0.0 and 1.0
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

# Do some hyperparameter tuning of layer size
layer_size_vs_accuracy = {}
for layer_size in layer_sizes:
	print(f"Fitting model with layer size: {layer_size}")
	# Create a sequential model with one hidden layer of 512 neurons
	# and one output layer of 10 digits for classifying digits 0 - 9
	model = keras.Sequential([
		layers.Dense(layer_size, activation="relu"),
		layers.Dense(10, activation="softmax")
	])

	# Compile the model tracking accuracy
	model.compile(
		optimizer=keras.optimizers.RMSprop(1e-2),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"]
	)

	# Effect of batch size
	#   small batch size                                  --> very slow, averaging and applying weights for each sample, each epoch, good result
	#   batch_size == sample_size (train_images.shape[0]) --> very fast, averaging and applying weights only once per epoch, poor result
	#   moderate batch size, such as 128                  --> quick and approaches answer quickly, nice happy medium, good result
	history = model.fit(
		train_images,
		train_labels,
		epochs=num_epochs,
		batch_size=batch_size,
		validation_split=0.2
	)
	layer_size_vs_accuracy[layer_size] = history.history

# Summarize the results of the experiment
fig = plt.figure()
plt.title("Hidden layer size vs. model accuracy")
x = layer_sizes
y = []
for layer_size in layer_size_vs_accuracy:
	y.append(layer_size_vs_accuracy[layer_size]['accuracy'][-1])

accuracy_vs_layer_size_plot = plt.plot(x, y, color='red', marker='o')
plt.show()
