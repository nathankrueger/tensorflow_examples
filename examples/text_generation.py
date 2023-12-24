import os

# disable some of the tensorflow chatter
#   0 = all messages are logged (default behavior)
#   1 = INFO messages are not printed
#   2 = INFO and WARNING messages are not printed
#   3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import keras
import numpy as np
from pathlib import Path

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
    seq_len: int,
    batch_size: int=64,
    validation_split: float=0.2
):
    # convert string data into tensor of shape (num_seq, seq_len)
    text_entries = []
    total_entries = len(text) // seq_len
    for i in range(total_entries):
        text_entries.append(text[i*seq_len:(i+1)*seq_len])
    text_vectorization.adapt(text_entries)

    def prepare_text_dataset_for_generation(text_batch):
        vectorized_sequences = text_vectorization(text_batch)
        x = vectorized_sequences[:, :-1]
        y = vectorized_sequences[:, 1:]
        return x, y
    
    num_train = int((1 - validation_split) * total_entries)

    train_dataset = tf.data.Dataset.from_tensor_slices(text_entries[:num_train])
    train_dataset = train_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(prepare_text_dataset_for_generation, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(2048).prefetch(128).cache()

    val_dataset = tf.data.Dataset.from_tensor_slices(text_entries[num_train:])
    val_dataset = val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(prepare_text_dataset_for_generation, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.shuffle(2048).prefetch(128).cache()

    return train_dataset, val_dataset

def sample_next(predictions, temperature: float=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

if __name__ == '__main__':
    modelckpt = 'transformer_generative.hdf5'
    seq_len = 128
    vocab_sz = 30000
    embed_dim = 512
    dense_dim = 2048
    num_heads = 8
    num_decoders = 3

    inputs = keras.Input(shape=(None,), dtype='int64')
    x = PositionalEmbedding(sequence_length=seq_len, input_dim=vocab_sz, output_dim=embed_dim)(inputs)
    for _ in range(num_decoders):
        x = TransformerDecoder(embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads)(x, x)
    outputs = keras.layers.Dense(vocab_sz, activation='softmax')(x)
    transformer_generative_model = keras.Model(inputs, outputs)
    transformer_generative_model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(transformer_generative_model.summary())

    text_vec_layer = keras.layers.TextVectorization(
        max_tokens=vocab_sz,
        output_mode='int',
        output_sequence_length=seq_len
    )
    text_corpus = get_shakespearean_training_text(shakespearean_text_file)
    token_index = dict(enumerate(text_vec_layer.get_vocabulary()))
    train_dataset, val_dataset = get_dataset(
                                    text_corpus,
                                    text_vec_layer,
                                    seq_len,
                                    batch_size=128,
                                    validation_split=0.2
                                )

    load_model_if_available = False

    # if requested, or the model doesn't exist on disk, train it!
    if not (load_model_if_available and os.path.exists(modelckpt)):
        try:
            transformer_generative_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=200,
                callbacks=[
                    keras.callbacks.TensorBoard(
                        log_dir='tensorboard_logs'
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath=modelckpt,
                        save_weights_only=False,
                        save_best_only=True
                    ),
                    keras.callbacks.EarlyStopping(
                        patience=10,
                        monitor='val_accuracy'
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=3,
                        factor=0.5,
                        montior='val_accuracy'
                    ),
                ]
            )
        except KeyboardInterrupt:
            print(os.linesep)
    
    # load the model
    transformer_generative_model = keras.models.load_model(
        modelckpt,
        custom_objects={
            "TransformerDecoder": TransformerDecoder,
            "PositionalEmbedding": PositionalEmbedding
        }
    )

    default_prompt = 'I'
    temperature = 0.7
    generate_len = 100
    for _ in range(100):
        print(os.linesep + '--- Sentence ---')
        sentence = default_prompt
        print(default_prompt, end=' ')
        for i in range(generate_len):
            tokenized_sentence = text_vec_layer([sentence])
            predictions = transformer_generative_model.predict(tokenized_sentence, verbose=0)
            next_token = sample_next(predictions[0, i, :], temperature=temperature)
            sampled_token = token_index[next_token]
            print(sampled_token, end=' ')
            sentence += " " + sampled_token
        print(os.linesep)