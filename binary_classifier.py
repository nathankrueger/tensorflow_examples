import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("TensowFlow version: " + tf.__version__)
tf.config.set_visible_devices([], 'GPU')

number_unique_words = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=number_unique_words)

# Decode the review into it's words
def decode_a_review(seq):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    result = " ".join([reverse_word_index.get(i - 3, "?") for i in seq])
    return result

# Multi-hot encode the sequences
def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        for j in seq:
            results[i, j] = 1
    return results

# Convert the training and destination data into 10000 dimensional
# vectors with multi-hot encoding.  It holds 1s for each word which
# appears in then input sequence -- no counts of a word, no order
x_train = vectorize_sequences(train_data, number_unique_words)
x_test = vectorize_sequences(test_data, number_unique_words)

# Convert the labels to a format appropriate for the model
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Set aside some of the training data to use for validation while training each epoch
x_train_val = x_train[:10000]
y_train_val = y_train[:10000]
x_train = x_train[10000:]
y_train = y_train[10000:]

best_weights_filename = "best_weights"

fit_callbacks = [
    # Stop after 2 consecutive decreases of validation value
    tf.keras.callbacks.EarlyStopping(patience=2),

    # Save a checkpoint file for each step
    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_weights_filename,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
]

# Train the model, stopping once the validation accuracy starts declining
model.fit(
    x_train,
    y_train,
    epochs=12,
    batch_size=512,
    validation_data=(x_train_val, y_train_val),
    callbacks=fit_callbacks
)

# Load in the best weights as determined by the validation data
model.load_weights(best_weights_filename)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Final loss: {test_loss}, accuracy: {test_accuracy}")