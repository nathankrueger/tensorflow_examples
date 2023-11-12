import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras import layers

# hyperparameters
num_epochs = 10
batch_size = 512
num_words_max = 10000

def to_onehot(labels, dimension=46):
    result = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        result[i, label] = 1
    return result

# create an array of shape (numsamples, num_words_max)
def to_multihot(data, dimension=num_words_max):
    result = np.zeros((len(data), dimension))
    for i, sequence in enumerate(data):
        for j in sequence:
            result[i, j] += 1
    return result

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
train_labels = to_onehot(train_labels)
test_labels = to_onehot(test_labels)
train_data = to_multihot(train_data)
test_data = to_multihot(test_data)

val_data = train_data[2000:]
val_labels = train_labels[2000:]
train_data = train_data[:2000]
train_labels = train_labels[:2000]

model = tf.keras.models.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="rmsprop",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    train_labels,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
    validation_data=(val_data, val_labels)
)

(loss, accuracy) = model.evaluate(
    test_data,
    test_labels,
    batch_size=batch_size      
)

# Predict one sample
prediction = model.predict(np.array([test_data[0],]))
print(prediction)
print(np.argmax(prediction))

# Predict all samples
predictions = model.predict(test_data)