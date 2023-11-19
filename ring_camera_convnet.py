import tensorflow as tf
import numpy as np
import random
import csv
import os
import shutil
import cv2
import keras
import glob
from PIL import Image
from pathlib import Path

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

labels_to_consider = ['none', 'car', 'dog', 'turkey', 'deer']

img_extensions = ['.jpg']
csv_path = './ring_downloader/ring_data/sept_through_nov_2023/frames/test.csv'

def get_unison_shuffled_np_array_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_unison_shuffled_list_copies(a, b):
    zipped = list(zip(a, b))
    random.shuffle(zipped)
    a, b = zip(*zipped)
    return a, b

def get_img_dict_from_csv(csv_path):
    img_dict = {}
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            img_path = row[0]
            label = row[1]
            img_dict[img_path] = label
    return img_dict

def save_img_dict_to_csv(img_dict, csv_path):
    # copy just in case we've messed something up...
    shutil.copy(csv_path, str(csv_path) + '.bak')

    # overwrites any pre-existing file
    with open(os.path.abspath(csv_path), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img in img_dict:
            writer.writerow([img, img_dict[img]])

def get_all_images_in_folder(folder, recursive=False):
    all_images = []
    for img_type in img_extensions:
        all_images.extend(glob.glob(str(Path(os.path.abspath(folder)) / f'*{img_type}'), recursive=recursive))
    all_images.sort()
    return all_images

def get_unlabeled_imgs(img_dict, folder):
    result = []
    all_imgs = get_all_images_in_folder(folder)
    for path in all_imgs:
        if not path in img_dict:
            result.append(path)
    return path

def get_num_unique_labels(all_labels):
    lbl_dict = {}
    for lbl in all_labels:
        lbl_dict[lbl] = lbl
    label_num = len(lbl_dict)
    return label_num

def convert_img_to_tensor(img_path):
    img = Image.open(img_path)
    img_tensor = tf.convert_to_tensor(img)
    img_tensor = img_tensor.numpy().astype("float32") / 255
    return img_tensor

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

def split_data_into_groups_bucketize(img_dict, max_images=None):
    # test_percentage = 1 - (train_percentage + val_percentage)
    train_percentage = 0.6
    val_percentage = 0.2

    all_images = []
    all_labels = []
    for k, v in img_dict.items():
        all_images.append(k)
        all_labels.append(v)

    # randomize the order
    all_images, all_labels = get_unison_shuffled_list_copies(all_images, all_labels)

    # reduce the sample size if requested
    if max_images is not None and max_images < len(all_images):
        all_images = all_images[:max_images]
        all_labels = all_labels[:max_images]

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

    # concatenate the percentage of each class into each set (train, val, test)
    for label in label_buckets:
        instances = label_buckets[label]
        num_instances = len(instances)
        train_end_idx = int(train_percentage * num_instances)
        val_end_idx = int(train_end_idx + (val_percentage * num_instances))

        # train
        curr_train_images = instances[:train_end_idx]
        curr_train_labels = [label] * train_end_idx
        if train_imgs is None:
            train_imgs = curr_train_images
            train_labels = curr_train_labels
        else:
            train_imgs = train_imgs + curr_train_images
            train_labels = train_labels + curr_train_labels

        # val
        curr_val_images = instances[train_end_idx:val_end_idx]
        curr_val_labels =  [label] * (val_end_idx - train_end_idx)
        if val_imgs is None:
            val_imgs = curr_val_images
            val_labels = curr_val_labels
        else:
            val_imgs = val_imgs + curr_val_images
            val_labels = val_labels + curr_val_labels

        # test
        curr_test_images = instances[val_end_idx:]
        curr_test_labels =  [label] * (num_instances - val_end_idx)
        if test_imgs is None:
            test_imgs = curr_test_images
            test_labels = curr_test_labels
        else:
            test_imgs = test_imgs + curr_test_images
            test_labels = test_labels + curr_test_labels

    # shuffle the train, val, and test content
    train_imgs, train_labels = get_unison_shuffled_list_copies(train_imgs, train_labels)
    val_imgs, val_labels = get_unison_shuffled_list_copies(val_imgs, val_labels)
    test_imgs, test_labels = get_unison_shuffled_list_copies(test_imgs, test_labels)

    return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels

def get_compiled_model(img_shape, num_outputs):
    inputs = keras.Input(shape=img_shape)

    x = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
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

def load_data(csv_path, force_reload_images=False, max_images=None):
    # only allow load from file if we have all the necessary content
    all_files_on_disk = True
    for file in all_numpy_files:
        if not os.path.exists(file):
            all_files_on_disk = False
            break

    # if all the files are present, or if we are forcing the image load (forcing introduces a new shuffle of the data)
    if all_files_on_disk and not force_reload_images:
        train_imgs = np.load(np_train_images_file)
        train_labels = np.load(np_train_labels_file)
        val_imgs = np.load(np_val_images_file)
        val_labels = np.load(np_val_labels_file)
        test_imgs = np.load(np_test_images_file)
        test_labels = np.load(np_test_labels_file)
        return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels
    else:
        # load in csv file (file_path --> label)
        img_dict = get_img_dict_from_csv(csv_path)

        # map labels to consider to integer values used by net
        unique_labels = {str_label: int_label for int_label, str_label in enumerate(labels_to_consider)}

        # convert labels to integer values
        img_dict = {img_path: unique_labels[str_label] for img_path, str_label in img_dict.items() if str_label in labels_to_consider}

        # get lists of equal order of (train, val, test) images and labels that are split by a requested percentage on a class by class
        # basis.  this way, if you ask for 60% training data, you'll get 60% of each class (which aren't going to be equally represented)
        train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = split_data_into_groups_bucketize(img_dict, max_images)

        # convert list of images to list of tensors
        print(f"Coverting {len(train_imgs) + len(val_imgs) + len(test_imgs)} images to tensor...")
        train_imgs = list(map(convert_img_to_tensor, train_imgs))
        val_imgs = list(map(convert_img_to_tensor, val_imgs))
        test_imgs = list(map(convert_img_to_tensor, test_imgs))

        # convert lists to np array
        train_imgs = np.array(train_imgs)
        train_labels = np.array(train_labels)
        val_imgs = np.array(val_imgs)
        val_labels = np.array(val_labels)
        test_imgs = np.array(test_imgs)
        test_labels = np.array(test_labels)

        # save the results for next time!
        np.save(np_train_images_file, train_imgs)
        np.save(np_train_labels_file, train_labels)
        np.save(np_val_images_file, val_imgs)
        np.save(np_val_labels_file, val_labels)
        np.save(np_test_images_file, test_imgs)
        np.save(np_test_labels_file, test_labels)

        return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels

def train_and_evaluate(force_reload_images):
    if use_cpu:
        tf.config.set_visible_devices([], 'GPU')

    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels  = load_data(csv_path, force_reload_images=force_reload_images, max_images=max_images)

    # how many unique labels are there?
    num_labels = get_num_unique_labels(train_labels)

    # what's the image shape for all images in this dataset
    img_shape = train_imgs[0].shape

    # get the model and show a brief summary of it
    model = get_compiled_model(img_shape=img_shape, num_outputs=num_labels)
    print(model.summary())

    # train the model, allowing user CTRL-C to quit the process early
    try:
        model.fit(
            train_imgs,
            train_labels,
            validation_data=(val_imgs, val_labels),
            epochs=100,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=15),
                keras.callbacks.ModelCheckpoint(
                    filepath=weights_filename,
                    save_weights_only=False,
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.5,
                    patience=6,
                    min_lr=0.00005
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

    # load in the weights from the epoch with the maximum accuracy
    # and evaluate the model on the test data
    model.load_weights(weights_filename)
    loss, accuracy = model.evaluate(
        test_imgs,
        test_labels,
        batch_size=64
    )
    print(f"Loss: {loss}, Accuracy: {accuracy}")

def evaluate_only(show_predict_loop=False):
    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels  = load_data(csv_path, force_reload_images=False, max_images=max_images)

    # how many unique labels are there?
    num_labels = get_num_unique_labels(train_labels)

    # what's the image shape for all images in this dataset
    img_shape = train_imgs[0].shape

    model = get_compiled_model(img_shape=img_shape, num_outputs=num_labels)
    print(model.summary())

    # evaluate the model from the best epoch checkpoint
    model.load_weights(weights_filename)
    model.evaluate(
        test_imgs,
        test_labels,
        batch_size=64
    )

    # show the labeled sample for fun & confirming the labels are correct
    if show_predict_loop:
        esc_key = 27
        image_review_ms = 1500
        label_dict = {k: v for k, v in enumerate(labels_to_consider)}
        predictions = model.predict(test_imgs)

        # loop over each prediction made on the test data
        for idx, img in enumerate(test_imgs):
            prediction = predictions[idx]
            int_label = tf.argmax(prediction).numpy()
            label = label_dict[int_label]

            # review just the most troublesome, least common classes
            if label in ['car', 'none']:
                continue

            # show the image for a moment before moving on to the next
            cv2.imshow(label, img)
            keyPressed = cv2.waitKeyEx(image_review_ms) & 0xFF
            if keyPressed == esc_key:
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()

if __name__ == '__main__':
    train_and_evaluate(force_reload_images=True)
    #evaluate_only(show_predict_loop=True)