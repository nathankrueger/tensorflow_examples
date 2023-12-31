import os, shutil, random
from pathlib import Path
import keras
import tensorflow as tf
from transfomers_for_text import TransformerEncoder, PositionalEmbedding

# get the data
#   wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
#   tar -xf aclImdb_v1.tar.gz
#   rm -r aclImdb/train/unsup

# Constants
batch_sz = 32
vocab_sz = 20000
base_dir = Path(os.path.dirname(__file__)) / 'aclImdb'
train_dir = base_dir / 'train'
val_dir = base_dir / 'val'
test_dir = base_dir / 'test'

assert train_dir.exists()
assert test_dir.exists()

# split off 20% of the training data and create a validation set (if needed)
if not os.path.exists(val_dir):
    for category in ('neg', 'pos'):
        os.makedirs(val_dir / category)
        files = os.listdir(train_dir / category)
        random.Random(1337).shuffle(files)
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname, val_dir / category / fname)

assert val_dir.exists()

# create the data sets
print(train_dir)
train_ds = keras.utils.text_dataset_from_directory(str(train_dir), batch_size=batch_sz)
print(val_dir)
val_ds = keras.utils.text_dataset_from_directory(str(val_dir), batch_size=batch_sz)
print(test_dir)
test_ds = keras.utils.text_dataset_from_directory(str(test_dir), batch_size=batch_sz)

def get_model_for_multihot(max_tokens=vocab_sz, hidden_layer_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = keras.layers.Dense(hidden_layer_dim, activation='relu')(inputs)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def multihot_example():
    # take one sample and display some info
    for inputs, targets in train_ds:
        print(f'inputs.shape: {inputs.shape}')
        print(f'inputs.dtype: {inputs.dtype}')
        print(f'targets.shape: {targets.shape}')
        print(f'targets.dtype: {targets.dtype}')
        print(f'inputs[0]: {inputs[0]}')
        print(f'targets[0]: {targets[0]}')
        break

    # get only the string content to be used for initializing the TextVectorization layer
    text_only_train_ds = train_ds.map(lambda x, y: x)

    for ngram in [1, 2, 3]:
        # limit the vocab to 20,000
        text_vec_layer = keras.layers.TextVectorization(
            max_tokens=vocab_sz,
            output_mode='multi_hot', # 'tf_idf' Term Frequency, Inverse Document Frequency -- divide the number of times a token appears in a given sample by the log of frequency across all samples
            ngrams=ngram
        )
        print('Creating the mapping from tokens / n-grams to multi-hot encoding...')
        text_vec_layer.adapt(text_only_train_ds)

        # convert to mapping of multihot vector to binary pos / neg classification
        print(f'Creating {ngram} ngram datasets...')
        binary_ngram_train_ds = train_ds.map(lambda x, y: (text_vec_layer(x), y), num_parallel_calls=os.cpu_count())
        binary_ngram_val_ds = val_ds.map(lambda x, y: (text_vec_layer(x), y), num_parallel_calls=os.cpu_count())
        binary_ngram_test_ds = test_ds.map(lambda x, y: (text_vec_layer(x), y), num_parallel_calls=os.cpu_count())

        # take one sample and display some info
        for inputs, targets in binary_ngram_train_ds:
            print(f'inputs.shape: {inputs.shape}')
            print(f'inputs.dtype: {inputs.dtype}')
            print(f'targets.shape: {targets.shape}')
            print(f'targets.dtype: {targets.dtype}')
            print(f'inputs[0]: {inputs[0]}')
            print(f'targets[0]: {targets[0]}')
            break

        model = get_model_for_multihot()
        model.summary()

        # 'cache()' on the dataset holds it in memory -- only works if it can fit in memory
        print(f'Fitting model for {ngram} ngram...')
        model.fit(
            binary_ngram_train_ds.cache(),
            validation_data=binary_ngram_val_ds.cache(),
            epochs=8
        )

def word_embeddings_example(embed_sz=256):
    max_seq_len = 600
    text_vec_layer = keras.layers.TextVectorization(
        max_tokens=vocab_sz,
        output_mode='int',
        output_sequence_length=max_seq_len
    )

    # get only the string content to be used for initializing the TextVectorization layer
    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vec_layer.adapt(text_only_train_ds)

    int_train_ds = train_ds.map(lambda x, y: (text_vec_layer(x), y), num_parallel_calls=os.cpu_count())
    int_val_ds = val_ds.map(lambda x, y: (text_vec_layer(x), y), num_parallel_calls=os.cpu_count())
    int_test_ds = test_ds.map(lambda x, y: (text_vec_layer(x), y), num_parallel_calls=os.cpu_count())

    inputs = keras.Input(shape=(None,), dtype='int64')
    embedded = keras.layers.Embedding(input_dim=vocab_sz, output_dim=embed_sz, mask_zero=True)(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(embedded)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(int_train_ds, validation_data=int_val_ds, epochs=10)

def transformer_encoder_example():
    model_checkpt = 'transformer_encoder_model.hdf5'
    vocab_sz = 20000
    seq_len = 600
    embed_dim = 256
    num_heads = 2
    dense_dim = 32

    text_vec_layer = keras.layers.TextVectorization(
        max_tokens=vocab_sz,
        output_mode='int',
        output_sequence_length=seq_len
    )

    # get only the string content to be used for initializing the TextVectorization layer
    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vec_layer.adapt(text_only_train_ds)

    int_train_ds = train_ds.map(lambda x, y: (text_vec_layer(x), y), num_parallel_calls=os.cpu_count())
    int_val_ds = val_ds.map(lambda x, y: (text_vec_layer(x), y), num_parallel_calls=os.cpu_count())
    int_test_ds = test_ds.map(lambda x, y: (text_vec_layer(x), y), num_parallel_calls=os.cpu_count())

    inputs = keras.layers.Input(shape=(None,), dtype='int64')
    x = PositionalEmbedding(
        sequence_length=seq_len,
        input_dim=vocab_sz,
        output_dim=embed_dim
    )(inputs)
    x = TransformerEncoder(
        embed_dim=embed_dim,
        dense_dim=dense_dim,
        num_heads=num_heads
    )(x)
    x = keras.layers.GlobalMaxPool1D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    try:
        model.fit(
            x=int_train_ds,
            validation_data=int_val_ds,
            callbacks=[
                keras.callbacks.ModelCheckpoint(model_checkpt, save_best_only=True, save_weights_only=False)
            ],
            epochs=20
        )
    except KeyboardInterrupt:
        print(os.linesep)

    model = keras.models.load_model(
                                model_checkpt,
                                custom_objects={
                                    "TransformerEncoder": TransformerEncoder,
                                    "PositionalEmbedding": PositionalEmbedding
                                    }
                            )
    
    print(f'Test accuracy: {model.evaluate(int_test_ds)[1]:.3f}')

if __name__ == '__main__':
    #multihot_example()
    #word_embeddings_example()
    transformer_encoder_example()
