import tensorflow as tf
import numpy as np
import os
import shutil

# For small batch sizes with RNN, CPU is much faster
#tf.config.set_visible_devices([], 'GPU')

tensorboard_logs_folder = os.path.join(os.path.dirname(__file__), 'tensorboard_logs')

csv_fname = 'jena_climate_2009_2016.csv'
csv_zip_fname = csv_fname + '.zip'
csv_path = os.path.join(os.path.dirname(__file__), csv_fname)

# get the data from the web
if not os.path.exists(csv_path):
    print(f'Downloading csv file: {csv_fname}...')
    os.system(f'wget https://s3.amazonaws.com/keras-datasets/{csv_zip_fname}')
    os.system(f'unzip {csv_zip_fname}')
    shutil.move(csv_fname, csv_path)
    os.remove(csv_zip_fname)

with open(csv_path) as f:
    data = f.read()

# get all data, sans header
lines = data.split('\n')
lines = lines[1:]

header = lines[0].split(',')

temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]

# train, val, test split
num_train_samples = int(len(raw_data) * 0.5)
num_val_samples = int(len(raw_data) * 0.25)
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

# normalize the data based on the training samples
mean = raw_data[:num_train_samples].mean(axis = 0)
raw_data -= mean
std_dev = raw_data[:num_train_samples].std(axis = 0)
raw_data /= std_dev

# the raw data is provided every 10 minutes, by setting a sampling rate of 6,
# we are using one sample per hour
sampling_rate = 6

# we are looking at sequneces 120 hours in length
sequence_length = 120

# the delay between the sequence data and its target is 24 hours in the future
delay = sampling_rate * (sequence_length + 24 - 1)

# we will load batches of 256 into memory for processing by the model
batch_sz = 256

train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_sz,
    start_index=0,
    end_index=num_train_samples
)

val_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_sz,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples
)

test_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_sz,
    start_index=num_train_samples + num_val_samples
)

def get_simple_rnn():
    num_features = len(raw_data[0])
    inputs = tf.keras.layers.Input(shape=(None, num_features))
    x = tf.keras.layers.SimpleRNN(16, return_sequences=True, activation='tanh')(inputs)
    x = tf.keras.layers.SimpleRNN(16, return_sequences=True, activation='tanh')(x)
    x = tf.keras.layers.SimpleRNN(16, return_sequences=False, activation='tanh')(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

def get_ltsm(timesteps):
    num_features = len(raw_data[0])
    inputs = tf.keras.layers.Input(shape=(timesteps, num_features))
    x = tf.keras.layers.LSTM(32, recurrent_dropout=0.25, return_sequences=False, activation='tanh')(inputs)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model  

# simple RNN
#model = get_simple_rnn()
model = get_ltsm(sequence_length)

# compile & train the model
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.fit(
    x=train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.abspath(tensorboard_logs_folder)
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=2,
            min_lr=0.00005
        )
    ]
)