import os

# disable some of the tensorflow chatter
#   0 = all messages are logged (default behavior)
#   1 = INFO messages are not printed
#   2 = INFO and WARNING messages are not printed
#   3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import string
import re
import random
import numpy as np
from pathlib import Path

# wget http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
# unzip -q spa-eng.zip

# read in the raw data into pairs
textfile = Path(os.path.dirname(__file__)) / 'spa-eng/spa.txt'
with open(textfile, 'r') as fh:
    lines = fh.read().split('\n')[:-1]
text_pairs = []

for line in lines:
    english, spanish = line.split('\t')
    spanish = '[start] ' + spanish + ' [end]'
    text_pairs.append((english, spanish))

# create the datasets
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples # 2x test samples as val samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

# string punctuation, keep [start] & [end]
strip_chars = string.punctuation + '¿'
strip_chars = strip_chars.replace('[', '')
strip_chars = strip_chars.replace(']', '')

print('Some example english-spanish pairs:')
print(train_pairs[0])
print(val_pairs[0])
print(test_pairs[0])

# define a text standardization method for use by our Spanish text vectorization layer
def custom_standardization(input_str):
    lowercase = tf.strings.lower(input_str)
    return tf.strings.regex_replace(lowercase, f'[{re.escape(strip_chars)}]', '')

# some hyperparams
vocab_sz = 15000
seq_len = 20
batch_sz = 64

source_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vocab_sz,
    output_mode='int',
    output_sequence_length=seq_len
)

# output sequence length is +1 in size since it is offset by 1 during training
target_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vocab_sz,
    output_mode='int',
    output_sequence_length=seq_len + 1,
    standardize=custom_standardization
)

# initialize our vectorization layers
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({'english': eng, 'spanish': spa[:, :-1]}, spa[:, 1:])

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_sz)
    dataset = dataset.map(format_dataset, num_parallel_calls=os.cpu_count())
    return dataset.shuffle(2048).prefetch(16).cache()

def simple_gru_example(load_model_if_available=True):
    embed_dim = 256
    latent_dim = 1024

    # encoder RNN
    source_layer = tf.keras.Input(shape=(None,), dtype='int64', name='english')
    x = tf.keras.layers.Embedding(
        input_dim=vocab_sz,
        output_dim=embed_dim,
        mask_zero=True
    )(source_layer)
    encoded_source_layers = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(latent_dim),
        merge_mode='sum'
    )(x)

    # decoder RNN
    past_target = tf.keras.Input(shape=(None,), dtype='int64', name='spanish')
    x = tf.keras.layers.Embedding(
        input_dim=vocab_sz,
        output_dim=embed_dim,
        mask_zero=True
    )(past_target)
    decoder_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True)
    x = decoder_gru(x, initial_state=encoded_source_layers)
    x = tf.keras.layers.Dropout(0.5)(x)
    target_next_step = tf.keras.layers.Dense(vocab_sz, activation='softmax')(x)
    seq2seq_rnn = tf.keras.Model([source_layer, past_target], target_next_step)

    seq2seq_rnn.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # dump out a nice graph
    tf.keras.utils.plot_model(model=seq2seq_rnn, to_file=str(Path(os.path.dirname(__file__)) / 'seq2seq_rnn.png'), show_shapes=True)

    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)
    test_ds = make_dataset(test_pairs)

    modelchkpt = 'seq2seq_gru.hdf5'

    # if requested, or the model doesn't exist on disk, train it!
    if not (load_model_if_available and os.path.exists(modelchkpt)):
        try:
            seq2seq_rnn.fit(
                x=train_ds,
                validation_data=val_ds,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=modelchkpt,
                        save_best_only=True,
                        save_weights_only=False
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        patience=3
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        patience=2,
                        factor=0.5
                    ),
                    tf.keras.callbacks.TensorBoard(
                        log_dir=os.path.abspath('tensorboard_logs')
                    ),
                ],
                epochs=20
            )
            print(f'Test accuracy: {seq2seq_rnn.evaluate(test_ds)[1]:.3f}')
        except KeyboardInterrupt:
            print(os.linesep)

    # load and evaluate the best model
    best_model = tf.keras.models.load_model(modelchkpt)
    print(f'Test accuracy: {best_model.evaluate(test_ds)[1]:.3f}')

    # do some inference for fun
    spa_vocab = target_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

    def decode_sequence(input_sentence):
        tokenized_input_sentence = source_vectorization([input_sentence])
        decoded_sentence = '[start]'

        for i in range(seq_len):
            tokenized_target_sentence = target_vectorization([decoded_sentence])
            next_token_predictions = best_model.predict([tokenized_input_sentence, tokenized_target_sentence])

            # [the first item in the batch, the current token, all probabilities in vocab_sz]
            sampled_token_index = np.argmax(next_token_predictions[0, i, :])
            samped_token = spa_index_lookup[sampled_token_index]
            decoded_sentence += ' ' + samped_token
            if samped_token == '[end]':
                break

        return decoded_sentence
    
    sentences_to_convert = [
        'I love riding motorcycles.',
        'I am enjoying practicing machine learning on my computer.',
        'Time to move on to transformer models!'
    ]

    for sentence in sentences_to_convert:
        print('---')
        print(f'[Input] English: {sentence}')
        print(f'[Output] Spanish: {decode_sequence(sentence)}')

if __name__ == '__main__':
    simple_gru_example(False)
