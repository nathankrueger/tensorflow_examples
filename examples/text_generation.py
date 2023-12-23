import os

# disable some of the tensorflow chatter
#   0 = all messages are logged (default behavior)
#   1 = INFO messages are not printed
#   2 = INFO and WARNING messages are not printed
#   3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import keras
import string
import re
import random
import numpy as np
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta

from transfomers_for_text import TransformerDecoder, PositionalEmbedding

base_dir = Path(os.path.dirname(__file__))

# https://www.kaggle.com/datasets/kewagbln/shakespeareonline/data
shakespearean_text_file = base_dir / 't8.shakespeare.txt'

def get_shakespearean_training_text(text_path):
    result = ''
    with open(text_path, 'r') as fh:
        lines = fh.readlines()
        record_line = False
        for line in lines:
            if record_line:
                result += ' ' + line
            if 'THE SONNETS' in line:
                record_line = True
    return result

def get_dataset(
    text: str,
    text_vectorization: keras.layers.TextVectorization,
    seq_len: int
):
    batch_sz = 64

    # convert string data into tensor of shape (num_seq, seq_len)
    text_entries = []
    ds_entries = len(text) // seq_len
    for i in range(ds_entries):
        text_entries.append(text[i*seq_len:(i+1)*seq_len])
    text_vectorization.adapt(text_entries)

    def prepare_text_dataset_for_generation(text_batch):
        vectorized_sequences = text_vectorization(text_batch)
        x = vectorized_sequences[:, :-1]
        y = vectorized_sequences[:, 1:]
        return x, y
    
    lm_dataset = tf.data.Dataset.from_tensor_slices(text_entries)
    lm_dataset = lm_dataset.batch(batch_sz, num_parallel_calls=tf.data.AUTOTUNE)
    lm_dataset = lm_dataset.map(prepare_text_dataset_for_generation, num_parallel_calls=tf.data.AUTOTUNE)

    return lm_dataset.shuffle(2048).prefetch(32).cache()

def sample_next(predictions, temperature: float=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

if __name__ == '__main__':
    modelckpt = 'transformer_seq2seq.hdf5'
    seq_len = 100
    vocab_sz = 30000
    embed_dim = 1024
    dense_dim = 2048
    num_heads = 6

    inputs = keras.Input(shape=(None,), dtype='int64')
    x = PositionalEmbedding(sequence_length=seq_len, input_dim=vocab_sz, output_dim=embed_dim)(inputs)
    x = TransformerDecoder(embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads)(x, x)
    outputs = keras.layers.Dense(vocab_sz, activation='softmax')(x)
    transformer_generative_model = keras.Model(inputs, outputs)
    transformer_generative_model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=['accuracy'],
        loss='sparse_categorical_crossentropy'
    )

    text_vec_layer = keras.layers.TextVectorization(
        max_tokens=vocab_sz,
        output_mode='int',
        output_sequence_length=seq_len
    )
    text_corpus = get_shakespearean_training_text(shakespearean_text_file)
    lm_dataset = get_dataset(text_corpus, text_vec_layer, seq_len)
    token_index = dict(enumerate(text_vec_layer.get_vocabulary()))

    load_model_if_available = False

    # if requested, or the model doesn't exist on disk, train it!
    if not (load_model_if_available and os.path.exists(modelckpt)):
        try:
            transformer_generative_model.fit(
                lm_dataset,
                epochs=50,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        filepath=modelckpt,
                        save_weights_only=False,
                        save_best_only=True
                    ),
                    keras.callbacks.EarlyStopping(
                        patience=5,
                        monitor='val_accuracy',
                        mode='max'
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=3,
                        factor=0.5,
                        montior='val_accuracy',
                        mode='max'
                    ),
                    keras.callbacks.TensorBoard(
                        log_dir='tensorboard_logs'
                    ),
                ]
            )
        except KeyboardInterrupt:
            print(os.linesep)

    # load and evaluate the best model
    best_model = keras.models.load_model(
        modelckpt,
        custom_objects={
            "TransformerDecoder": TransformerDecoder,
            "PositionalEmbedding": PositionalEmbedding
        }
    )

    default_prompt = 'I'
    generate_len = 100
    for _ in range(100):
        print('Generating context: ')
        sentence = default_prompt
        for i in range(seq_len - generate_len):
            tokenized_sentence = text_vec_layer([sentence])
            predictions = transformer_generative_model.predict(tokenized_sentence)
            next_token = sample_next(predictions[0, i, :])
            sampled_token = token_index[next_token]
            sentence += " " + sampled_token
        print(sentence)
