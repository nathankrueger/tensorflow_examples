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
import re
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
    batch_size: int=64,
    validation_split: float=0.2
) -> (tf.data.Dataset, tf.data.Dataset):
    # filter out repeated legal disclaimer
    text = re.sub(r'<<[^>]+>>', r'', text)

    # filter out newlines
    text = re.sub(r'\n\s+', '', text)

    # attempt to use regex to find complete sentences
    sentences = []
    matches = re.finditer(r'(?<=[\.!?\]])\s+(.+?[\.!?])', text, re.DOTALL)
    for match in matches:
        sentence = match[1]
        if len(sentence.split()) > 1:
            sentences.append(sentence)
    total_sentences = len(sentences)
    print(f'Total sentences found: {total_sentences}')

    # print out some metrics on sentence length
    total_sentence_len = 0
    for sentence in sentences:
        total_sentence_len += len(sentence)
    print(f'Avg. sentence len: {total_sentence_len / total_sentences}')

    # teach the vocabulary
    text_vectorization.adapt(sentences)

    # function for producing vectorized inputs & outputs
    def prepare_text_dataset_for_generation(text_batch):
        vectorized_sequences = text_vectorization(text_batch)
        # inputs start at the first token, and have the last token removed
        x = vectorized_sequences[:, :-1]

        # outputs start at the second token, and include the last token
        y = vectorized_sequences[:, 1:]

        # in this way, the outputs are shifted by one from the inputs,
        # e.g. the i-th output (prediction) is the (i-1)-th input
        return x, y
    
    num_train = int((1 - validation_split) * total_sentences)

    # convert string data into tensor of shape (num_seq, seq_len)
    train_dataset = tf.data.Dataset.from_tensor_slices(sentences[:num_train])
    train_dataset = train_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(prepare_text_dataset_for_generation, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(4096).prefetch(128).cache()

    if validation_split > 0.0:
        val_dataset = tf.data.Dataset.from_tensor_slices(sentences[num_train:])
        val_dataset = val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(prepare_text_dataset_for_generation, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.shuffle(4096).prefetch(128).cache()
    else:
        val_dataset = None

    return train_dataset, val_dataset

"""
Sample the next token with a given amount of randomness aka 'temperature'
"""
def sample_next(predictions, temperature: float=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

"""
Store intersting hyperparmeters
"""
class LanguageModelParams:
    def __init__(
            self,
            seq_len: int=64,
            vocab_sz: int=30000,
            embed_dim: int=512,
            dense_dim: int=2048,
            num_heads: int=8,
            num_decoders: int=3,
            batch_size: int=64,
            validation_split: float=0.5,
            dropout_amt: float=0.1,
            modelckpt: str='transformer_generative.hdf5'
        ):
        self.seq_len = seq_len
        self.vocab_sz = vocab_sz
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.num_decoders = num_decoders
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.dropout_amt = dropout_amt
        self.modelckpt = modelckpt

quick_test_params = LanguageModelParams(
    seq_len=32,
    vocab_sz=20000,
    embed_dim=256,
    dense_dim=512,
    num_heads=2,
    num_decoders=1,
    validation_split=0.0
)

accurate_test_params = LanguageModelParams(
    seq_len=32,
    vocab_sz=40000,
    embed_dim=512,
    dense_dim=2048,
    num_heads=6,
    num_decoders=4,
    validation_split=0.3
)

if __name__ == '__main__':
    load_model_if_available = False
    continue_training = False
    total_epochs = 500
    params = accurate_test_params

    # build the transformer encoder stack
    inputs = keras.Input(shape=(params.seq_len,), dtype='int64')
    x = PositionalEmbedding(
        sequence_length=params.seq_len,
        input_dim=params.vocab_sz,
        output_dim=params.embed_dim
    )(inputs)
    x = keras.layers.Dropout(params.dropout_amt)(x)
    for _ in range(params.num_decoders):
        x = TransformerDecoder(
            embed_dim=params.embed_dim,
            dense_dim=params.dense_dim,
            num_heads=params.num_heads,
            dropout_amt=params.dropout_amt
        )(x, x)

    outputs = keras.layers.Dense(params.vocab_sz, activation='softmax')(x)
    transformer_generative_model = keras.Model(inputs, outputs)
    transformer_generative_model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(transformer_generative_model.summary())

    # token index 0: MASK token (not a word) represented as ''
    # token index 1: OOV (out of vocabulary) token represented as '[UNK]'
    text_vec_layer = keras.layers.TextVectorization(
        max_tokens=params.vocab_sz,
        output_mode='int',
        output_sequence_length=params.seq_len + 1,
        standardize='lower'
    )
    text_corpus = get_shakespearean_training_text(shakespearean_text_file)
    train_dataset, val_dataset = get_dataset(
                                    text=text_corpus,
                                    text_vectorization=text_vec_layer,
                                    batch_size=params.batch_size,
                                    validation_split=params.validation_split
                                )
    token_index = dict(enumerate(text_vec_layer.get_vocabulary()))

    monitor = 'val_accuracy' if params.validation_split > 0.0 else 'accuracy'
    model_on_disk = os.path.exists(params.modelckpt)

    # load in the model to continue training where we left off, if requested
    if continue_training and model_on_disk:
        # load the model
        transformer_generative_model = keras.models.load_model(
            filepath=params.modelckpt,
            custom_objects={
                "TransformerDecoder": TransformerDecoder,
                "PositionalEmbedding": PositionalEmbedding
            }
        )

    # if requested, or the model doesn't exist on disk, train it!
    if not (load_model_if_available and model_on_disk) or continue_training:
        try:
            transformer_generative_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=total_epochs,
                callbacks=[
                    keras.callbacks.TensorBoard(
                        log_dir='tensorboard_logs'
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath=params.modelckpt,
                        save_weights_only=False,
                        save_best_only=True,
                        monitor=monitor
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor=monitor,
                        factor=0.5,
                        patience=5
                    )
                ]
            )
        except KeyboardInterrupt:
            print(os.linesep)
    
    # load the model
    transformer_generative_model = keras.models.load_model(
        filepath=params.modelckpt,
        custom_objects={
            "TransformerDecoder": TransformerDecoder,
            "PositionalEmbedding": PositionalEmbedding
        }
    )

    # generate some texts
    default_prompt = 'I'
    temperature = 0.7
    for _ in range(5):
        print(os.linesep + '--- Sentence ---')
        sentence = default_prompt
        print(default_prompt, end=' ')
        for i in range(params.seq_len):
            tokenized_sentence = text_vec_layer([sentence])
            predictions = transformer_generative_model.predict(tokenized_sentence, verbose=0)
            next_token = sample_next(predictions[0, i, :], temperature=temperature)
            sampled_token = token_index[next_token]
            print(sampled_token, end=' ')
            sentence += " " + sampled_token
        print('')