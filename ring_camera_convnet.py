import tensorflow as tf
import numpy as np
import csv
import keras
from PIL import Image

best_weights_filename = "best_weights"
np_images_file = 'all_images.npy'
np_labels_file = 'all_labels.npy'

#tf.config.set_visible_devices([], 'GPU')

csv_path = '/Users/nathankrueger/Documents/Programming/ML/tensorflow_examples/ring_downloader/ring_data/sept_through_nov_2023/frames/test.csv'

def sort_func(pair):
    key = pair[0]
    filenum = key.split('__')[1]
    rest = key.split('__')[2].split('_')
    framenum = rest[1].split('.jpg')[0]
    return int(filenum) * 20000 + int(framenum)

load_from_npy = True
def load_data():
    if load_from_npy:
        all_images = np.load(np_images_file)
        all_labels = np.load(np_labels_file)
        return all_images, all_labels
    else:
        all_images = []
        all_labels = []

        img_dict = {}
        unique_labels = {}
        label_num = 0
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                img_path = row[0]
                label = row[1]
                label_int = None

                if label in unique_labels:
                    label_int = unique_labels[label]
                else:
                    unique_labels[label] = label_num
                    print(f"{label_num} == {label}")
                    label_int = label_num
                    label_num += 1                
                
                img_dict[img_path] = label_int

        img_dict = dict(sorted(img_dict.items(), key=sort_func))

        img_idx = 0
        for key, val in img_dict.items():
            print(f'Converting img: {img_idx}...')
            img_idx += 1
            img = Image.open(key)
            img_tensor = tf.convert_to_tensor(img)
            all_images.append(img_tensor)
            all_labels.append(val)

        all_images = np.array(all_images)
        all_images = all_images.astype("float32") / 255
        all_labels = np.array(all_labels)

        np.save(np_images_file, all_images)
        np.save(np_labels_file, all_labels)

        return (all_images, all_labels)

def main():
    all_images, all_labels = load_data()

    # how many unique labels are there?
    lbl_dict = {}
    for lbl in all_labels:
        lbl_dict[lbl] = lbl
    label_num = len(lbl_dict)

    print(f"all_images: {all_images.shape}")
    print(f"all_labels: {all_labels.shape}")

    cutoff_idx = 2500

    train_imgs = all_images[:cutoff_idx]
    train_labels = all_labels[:cutoff_idx]

    val_imgs = train_imgs[2000:]
    val_labels = train_labels[2000:]
    train_imgs = train_imgs[:2000]
    train_labels = train_labels[:2000]

    test_imgs = all_images[cutoff_idx:]
    test_labels = all_labels[cutoff_idx:]

    print(f"train_imgs: {train_imgs.shape}")
    print(f"train_labels: {train_labels.shape}")

    img_shape = all_images[0].shape
    print('img shape: ' + str(img_shape))
    inputs = keras.Input(shape=img_shape)
    x = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    #x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(label_num, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_imgs,
        train_labels,
        validation_data=(val_imgs, val_labels),
        epochs=100,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5),
            keras.callbacks.ModelCheckpoint(
                filepath=best_weights_filename,
                save_weights_only=True,
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        ],
        batch_size=16
    )

    model.save('ring_camera_convnet.keras')

    #model.evaluate()


if __name__ == '__main__':
    main()
