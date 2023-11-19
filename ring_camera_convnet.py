import tensorflow as tf
import numpy as np
import csv
import os
import cv2
import keras
from PIL import Image

weights_filename = "ring_convnet_weights/checkpt.ckpt"
np_train_images_file = 'train_images.npy'
np_train_labels_file = 'train_labels.npy'
np_val_images_file = 'val_images.npy'
np_val_labels_file = 'val_labels.npy'
np_test_images_file = 'test_images.npy'
np_test_labels_file = 'test_labels.npy'
all_numpy_files = [
    np_train_images_file,
    np_train_labels_file,
    np_val_images_file,
    np_val_labels_file,
    np_test_images_file,
    np_test_labels_file
]

use_cpu = False
max_images = 1000

labels_to_consider = ['none', 'car', 'deer', 'turkey']

csv_path = '/Users/nathankrueger/Documents/Programming/ML/tensorflow_examples/ring_downloader/ring_data/sept_through_nov_2023/frames/test.csv'

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_num_unique_labels(all_labels):
    lbl_dict = {}
    for lbl in all_labels:
        lbl_dict[lbl] = lbl
    label_num = len(lbl_dict)
    return label_num

def split_data_into_groups_seq(all_images, all_labels):
    train_percentage = 0.6
    val_percentage = 0.2
    total_data_items = len(all_images)

    print(f"all_images: {all_images.shape}")
    print(f"all_labels: {all_labels.shape}")
    
    train_end_idx = int(train_percentage * total_data_items)
    val_end_idx = int(train_end_idx + (val_percentage * total_data_items))

    train_imgs = all_images[:train_end_idx]
    train_labels = all_labels[:train_end_idx]

    val_imgs = all_images[train_end_idx:val_end_idx]
    val_labels = all_labels[train_end_idx:val_end_idx]

    test_imgs = all_images[val_end_idx:]
    test_labels = all_labels[val_end_idx:]

    print(f"train_imgs: {train_imgs.shape}")
    print(f"train_labels: {train_labels.shape}")
    print(f"val_imgs: {val_imgs.shape}")
    print(f"val_labels: {val_labels.shape}")
    print(f"test_imgs: {test_imgs.shape}")
    print(f"test_labels: {test_labels.shape}")

    return (train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels)

def split_data_into_groups_bucketize(all_images, all_labels, max_images=None):
    train_percentage = 0.6
    val_percentage = 0.2

    # randomize, then reduce the sample size
    if max_images is not None and max_images < len(all_images):
        all_images, all_labels = unison_shuffled_copies(all_images, all_labels)
        all_images = all_images[:max_images]
        all_labels = all_labels[:max_images]

    print(f"all_images: {all_images.shape}")
    print(f"all_labels: {all_labels.shape}")

    # bucketize by group
    label_buckets = {}
    for idx, label in enumerate(all_labels):
        if label in label_buckets:
            label_buckets[label].append(all_images[idx])
        else:
            label_buckets[label] = [all_images[idx]]

    train_imgs = None
    train_labels = None
    val_imgs = None
    val_labels = None
    test_imgs = None
    test_labels = None

    # Concatenate the percentage of each class into each set (train, val, test)
    for label in label_buckets:
        instances = label_buckets[label]
        num_instances = len(instances)
        train_end_idx = int(train_percentage * num_instances)
        val_end_idx = int(train_end_idx + (val_percentage * num_instances))

        curr_train_images = np.array(instances[:train_end_idx])
        curr_train_labels =  np.full((train_end_idx,), label, dtype=int)
        if train_imgs is None:
            train_imgs = curr_train_images
            # Fill with the current label
            train_labels = curr_train_labels
        else:
            train_imgs = np.concatenate((train_imgs, curr_train_images))
            train_labels = np.concatenate((train_labels, curr_train_labels))

        curr_val_images = np.array(instances[train_end_idx:val_end_idx])
        curr_val_labels =  np.full((val_end_idx - train_end_idx,), label, dtype=int)
        if val_imgs is None:
            val_imgs = curr_val_images
            # Fill with the current label
            val_labels = curr_val_labels
        else:
            val_imgs = np.concatenate((val_imgs, curr_val_images))
            val_labels = np.concatenate((val_labels, curr_val_labels))

        curr_test_images = np.array(instances[val_end_idx:])
        curr_test_labels =  np.full((num_instances - val_end_idx,), label, dtype=int)
        if test_imgs is None:
            test_imgs = curr_test_images
            # Fill with the current label
            test_labels = curr_test_labels
        else:
            test_imgs = np.concatenate((test_imgs, curr_test_images))
            test_labels = np.concatenate((test_labels, curr_test_labels))

    print(f"train_imgs: {train_imgs.shape}")
    print(f"train_labels: {train_labels.shape}")
    print(f"val_imgs: {val_imgs.shape}")
    print(f"val_labels: {val_labels.shape}")
    print(f"test_imgs: {test_imgs.shape}")
    print(f"test_labels: {test_labels.shape}")

    (train_imgs, train_labels) = unison_shuffled_copies(train_imgs, train_labels)
    (val_imgs, val_labels) = unison_shuffled_copies(val_imgs, val_labels)
    (test_imgs, test_labels) = unison_shuffled_copies(test_imgs, test_labels)

    return (train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels)

def get_compiled_model(img_shape, num_outputs):
    inputs = keras.Input(shape=img_shape)

    x = keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu')(inputs)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(num_outputs, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def load_data(force_reload_images=False, max_images=None):
    all_files_on_disk = True
    for file in all_numpy_files:
        if not os.path.exists(file):
            all_files_on_disk = False
            break

    # If all the files are present, or if we are forcing the image load (forcing introduces a new shuffle of the data)
    if all_files_on_disk and not force_reload_images:
        train_imgs = np.load(np_train_images_file)
        train_labels = np.load(np_train_labels_file)
        val_imgs = np.load(np_val_images_file)
        val_labels = np.load(np_val_labels_file)
        test_imgs = np.load(np_test_images_file)
        test_labels = np.load(np_test_labels_file)
        return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels
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

                # Convert labels to integer values
                if label in unique_labels:
                    label_int = unique_labels[label]
                else:
                    # Only look at certain classes if desired
                    if labels_to_consider is not None and label in labels_to_consider:
                        unique_labels[label] = label_num
                        print(f"{label_num} == {label}")
                        label_int = label_num
                        label_num += 1                
                
                # if we have an image we wish to consider in our dataset, record it
                if label_int is not None:
                    img_dict[img_path] = label_int

        # Convert images to tensors
        img_idx = 0
        for key, val in img_dict.items():
            print(f'Converting img to tensor format and normalizing: {img_idx}...')
            img_idx += 1
            img = Image.open(key)
            img_tensor = tf.convert_to_tensor(img)
            all_images.append(img_tensor)
            all_labels.append(val)

        # Convert image sequence to np array, normalize image channels from 0.0 to 1.0
        all_images = np.array(all_images)
        all_images = all_images.astype("float32") / 255
        all_labels = np.array(all_labels)

        (train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels) = split_data_into_groups_bucketize(all_images, all_labels, max_images)

        # Save the results for next time!
        np.save(np_train_images_file, train_imgs)
        np.save(np_train_labels_file, train_labels)
        np.save(np_val_images_file, val_imgs)
        np.save(np_val_labels_file, val_labels)
        np.save(np_test_images_file, test_imgs)
        np.save(np_test_labels_file, test_labels)

        return (train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels)

def main():
    if use_cpu:
        tf.config.set_visible_devices([], 'GPU')

    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels  = load_data(force_reload_images=True, max_images=max_images)

    # how many unique labels are there?
    num_labels = get_num_unique_labels(train_labels)

    # what's the image shape for all images in this dataset
    img_shape = train_imgs[0].shape

    model = get_compiled_model(img_shape=img_shape, num_outputs=num_labels)
    print(model.summary())

    try:
        model.fit(
            train_imgs,
            train_labels,
            validation_data=(val_imgs, val_labels),
            epochs=100,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5),
                keras.callbacks.ModelCheckpoint(
                    filepath=weights_filename,
                    save_weights_only=True,
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=3,
                    min_lr=0.00001
                ),
                keras.callbacks.TensorBoard(
                    log_dir=os.path.abspath("./tensorboard_logs")
                )
            ],
            batch_size=64
        )
    except KeyboardInterrupt:
        print(os.linesep + "Killed fit early via CTRL-C...")
        pass

    model.load_weights(weights_filename)
    (loss, accuracy) = model.evaluate(
        test_imgs,
        test_labels,
        batch_size=64
    )

    print(f"Loss: {loss}, Accuracy: {accuracy}")

def evaluate_only(show_predict_loop=False):
    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels  = load_data(force_reload_images=False, max_images=max_images)

    # how many unique labels are there?
    num_labels = get_num_unique_labels(train_labels)

    # what's the image shape for all images in this dataset
    img_shape = train_imgs[0].shape

    model = get_compiled_model(img_shape=img_shape, num_outputs=num_labels)
    print(model.summary())

    model.load_weights(weights_filename)

    model.evaluate(
        test_imgs,
        test_labels,
        batch_size=64
    )

    if show_predict_loop:
        esc_key = 27
        image_review_ms = 2000
        label_dict = {0:"car", 1:"none", 2:"deer", 3:"turkey"}
        predictions = model.predict(test_imgs)
        for idx, img in enumerate(test_imgs):
            prediction = predictions[idx]
            int_label = tf.argmax(prediction).numpy()
            if int_label < 2:
                continue
            label = label_dict[int_label]
            cv2.imshow(label, img)
            keyPressed = cv2.waitKeyEx(image_review_ms) & 0xFF
            if keyPressed == esc_key:
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()

if __name__ == '__main__':
    #main()
    evaluate_only(True)