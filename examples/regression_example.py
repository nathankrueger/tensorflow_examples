import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(f"Data type: {type(test_data)}")
print(f"Data shape: {test_data.shape}")
for col in test_data[0]:
    print(col)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    model = tf.keras.models.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

# Use k-fold vaidaiton to compensate for a small data set.
# This helps mitigate the issue of using too small a validation set
# where an arbitrary choice of validation data could lead to a 
# result with too high a variance
k=4
num_epochs = 500

# // is floor division
num_val_samples = len(train_data) // k
all_scores = []
all_histories = []
for i in range(k):
    print(f"Processing fold {i}")
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0
    )

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
         axis=0
    )

    model = build_model()
    hist = model.fit(
        partial_train_data,
        partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs,
        batch_size=16,
        verbose=0 # prevents all the printing -- faster!
    )

    mae_history = hist.history["val_mae"]
    all_histories.append(mae_history)
    #val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    #all_scores.append(val_mae)

#print(all_scores)
#kfold_mean = np.mean(all_scores)
#print(f"K fold result: {kfold_mean}")

# create a list of mean average error for all epochs, per fold
average_mae_history = [
    np.mean([x[i] for x in all_histories]) for i in range(num_epochs)
]

plt.plot(range(10, len(average_mae_history) + 1), average_mae_history[9:])
plt.xlabel("Epochs")
plt.ylabel("Validation Mean Accuracy Error")
plt.show()