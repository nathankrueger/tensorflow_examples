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
    train_dataset = train_dataset.shuffle(4096).prefetch(128).cache()

    if validation_split > 0.0:
        val_dataset = tf.data.Dataset.from_tensor_slices(text_entries[num_train:])
        val_dataset = val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(prepare_text_dataset_for_generation, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.shuffle(4096).prefetch(128).cache()
    else:
        val_dataset = None

    return train_dataset, val_dataset

def sample_next(predictions, temperature: float=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

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
    seq_len=100,
    vocab_sz=30000,
    embed_dim=512,
    dense_dim=2048,
    num_heads=6,
    num_decoders=3,
    validation_split=0.0
)

if __name__ == '__main__':
    load_model_if_available = False
    params = accurate_test_params

    # build the transformer encoder stack
    inputs = keras.Input(shape=(None,), dtype='int64')
    x = PositionalEmbedding(
        sequence_length=params.seq_len,
        input_dim=params.vocab_sz,
        output_dim=params.embed_dim
    )(inputs)
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

    text_vec_layer = keras.layers.TextVectorization(
        max_tokens=params.vocab_sz,
        output_mode='int',
        output_sequence_length=params.seq_len
    )
    text_corpus = get_shakespearean_training_text(shakespearean_text_file)
    train_dataset, val_dataset = get_dataset(
                                    text_corpus,
                                    text_vec_layer,
                                    params.seq_len,
                                    batch_size=params.batch_size,
                                    validation_split=params.validation_split
                                )
    token_index = dict(enumerate(text_vec_layer.get_vocabulary()))

    monitor = 'val_accuracy' if params.validation_split > 0.0 else 'accuracy'

    def lr_scheduler(epoch, lr):
        if epoch < 30:
            return 0.001
        else:
            return 0.0005

    # if requested, or the model doesn't exist on disk, train it!
    if not (load_model_if_available and os.path.exists(params.modelckpt)):
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
                        filepath=params.modelckpt,
                        save_weights_only=False,
                        save_best_only=True,
                        monitor=monitor
                    ),
                    keras.callbacks.LearningRateScheduler(
                        schedule=lr_scheduler
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
    generate_len = 100
    for _ in range(100):
        print(os.linesep + '--- Sentence ---')
        sentence = default_prompt
        print(default_prompt, end=' ')
        for i in range(generate_len):
            i = min(i, params.seq_len - 1)
            tokenized_sentence = text_vec_layer([sentence])

            if len(tokenized_sentence) > params.seq_len:
                end_idx = -params.seq_len
            else:
                end_idx = len(tokenized_sentence)
            predictions = transformer_generative_model.predict(tokenized_sentence[:end_idx], verbose=0)
            next_token = sample_next(predictions[0, i, :], temperature=temperature)
            sampled_token = token_index[next_token]
            print(sampled_token, end=' ')
            sentence += " " + sampled_token
        print('')