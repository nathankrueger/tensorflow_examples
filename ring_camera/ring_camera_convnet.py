import tensorflow as tf
import numpy as np
import math, random
import os, shutil, glob
import csv
import cv2

from itertools import repeat
from pathlib import Path

THIS_FOLDER_PATH = Path(os.path.dirname(__file__))
ESC_KEY = 27
IMG_EXTENSIONS = ['.jpg']

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
        reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if len(row) == 2:
                img_path = row[0]
                label = row[1]
                img_dict[img_path] = label
    return img_dict

def save_img_dict_to_csv(img_dict, csv_path):
    # copy just in case we've messed something up...
    if os.path.exists(csv_path):
        shutil.copy(csv_path, str(csv_path) + '.bak')

    # overwrites any pre-existing file
    with open(os.path.abspath(csv_path), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img in img_dict:
            writer.writerow([img, img_dict[img]])

def get_all_images_in_folder(folder, recursive=False):
    all_images = []
    for img_type in IMG_EXTENSIONS:
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

def convert_img_to_tensor(img_path, img_size=None):
    img = cv2.imread(img_path)
    if img_size is not None:
        # remove color channel for resizing if present
        if len(img_size) == 3:
            img_size = img_size[:2]
        img = cv2.resize(img, img_size)
    img_tensor = tf.convert_to_tensor(img)
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

def split_data_into_groups_bucketize(img_dict, max_images=None, max_images_per_class=False, force_even_distribution=False):
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

    # force an even distribution by finding the lowest represented class
    # and only choosing that many samples for every other class
    if force_even_distribution:
        label_buckets_force_even_dist = {}
        min_rep_class = min(map(len, label_buckets.values()))
        for label in label_buckets:
            class_list = label_buckets[label]
            random.shuffle(class_list)
            label_buckets_force_even_dist[label] = class_list[:min_rep_class]
        label_buckets = label_buckets_force_even_dist

    if max_images_per_class:
        label_buckets_per_class_limit = {}
        for label in label_buckets:
            class_list = label_buckets[label]
            random.shuffle(class_list)
            label_buckets_per_class_limit[label] = class_list[:max_images_per_class]
        label_buckets = label_buckets_per_class_limit

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

def display_class_histogram(msg, imgs, img_dict, int_to_str_label_dict):
    histogram = {}
    for img in imgs:
        int_label = img_dict[img]
        str_label = int_to_str_label_dict[int_label]
        if str_label in histogram:
            histogram[str_label] += 1
        else:
            histogram[str_label] = 1

    if msg is not None:
        print(msg)
    for str_label, count in sorted(histogram.items()):
        print(f'{str_label}: {count}')

"""
Get a dictionary of integer label to string label
"""
def get_label_dict(filepath):
    img_dict = {}
    with open(filepath, 'r', newline='') as labelfile:
        line = labelfile.readline()
        for idx, label in enumerate(line.split(',')):
            img_dict[idx] = label
    return img_dict

class RingCameraConvnet:
    max_images = 15000
    use_cpu = False

    model_folder_path = THIS_FOLDER_PATH / "ring_convnet_model"
    model_folder = str(model_folder_path)
    keras_model_folder = str(model_folder_path / "keras")
    label_file = str(model_folder_path / "labels.csv")

    np_train_images_file = str(model_folder_path / 'train_images.npy')
    np_train_labels_file = str(model_folder_path / 'train_labels.npy')
    np_val_images_file = str(model_folder_path / 'val_images.npy')
    np_val_labels_file = str(model_folder_path / 'val_labels.npy')
    np_test_images_file = str(model_folder_path / 'test_images.npy')
    np_test_labels_file = str(model_folder_path / 'test_labels.npy')

    all_numpy_files = [
        np_train_images_file,
        np_train_labels_file,
        np_val_images_file,
        np_val_labels_file,
        np_test_images_file,
        np_test_labels_file
    ]

    def __init__(self, labeled_csv_images=None, labels_to_consider=None):
        self.__labeled_images_csv = labeled_csv_images

        # if no explicit list of labels to consider is passed in, use the default label file
        if labels_to_consider == None:
            if os.path.exists(RingCameraConvnet.label_file):
                self.__label_dict = get_label_dict(RingCameraConvnet.label_file)
            else:
                raise Exception(f'No labels provided, and model csv file {RingCameraConvnet.label_file} is missing')
        else:
            self.__label_dict = {int_label: str_label for int_label, str_label in enumerate(labels_to_consider)}

        self.__train_imgs = None
        self.__train_labels = None
        self.__val_imgs = None
        self.__val_labels = None
        self.__test_imgs = None
        self.__test_labels = None
        self.__data_loaded = False

        self.__model = None
        self.__conv_base = None

    def __confirm_data_is_loaded(self, ctx):
        if not self.__data_loaded:
            raise Exception(f'{ctx} requires loaded data.')

    def __confirm_model_is_loaded(self, ctx):
        if self.__model is None:
            raise Exception(f'{ctx} requires a loaded model')

    def __confirm_labeled_csv_is_defined(self, ctx):
        if self.__labeled_images_csv is None:
            raise Exception(f'{ctx} requires a labeled images and data CSV file')

    def __confirm_data_and_model_are_valid(self, ctx):
        self.__confirm_data_is_loaded(ctx)
        self.__confirm_model_is_loaded(ctx)

    def __report_on_mispredictions(self, predictions):
        self.__confirm_data_is_loaded('Reporting mispredictions')

        mis_predicts = {}

        for idx, int_pred_label in enumerate(predictions):
            int_expect_label = self.__test_labels[idx]
            if int_pred_label != int_expect_label:
                str_pred_label = self.__label_dict[int_pred_label]
                str_expect_label = self.__label_dict[int_expect_label]

                key = (str_expect_label, str_pred_label)
                if key in mis_predicts:
                    mis_predicts[key] += 1
                else:
                    mis_predicts[key] = 1

        # sort by value, we want to see the most often mistakes first
        mis_predicts = dict(sorted(mis_predicts.items(), key=lambda item: item[1], reverse=True))

        for mp in mis_predicts.items():
            comp = mp[0]
            num = mp[1]

            expect = comp[0]
            actual = comp[1]

            print(f'{expect} predicted as {actual} {num} times...')

    """
    Return the keras model -- use with caution
    """
    def get_model(self):
        return self.__model

    """
    Load data either from numpy array on disk (fast), or image files directly (slow)
    """
    def load_data(self, img_size=None, force_reload_images=False, max_images=None, max_images_per_class=None, force_even_distribution=False):
        # only allow load from file if we have all the necessary content
        all_files_on_disk = True
        for file in RingCameraConvnet.all_numpy_files:
            if not os.path.exists(file):
                all_files_on_disk = False
                break

        # if all the files are present, or if we are forcing the image load (forcing introduces a new shuffle of the data)
        if all_files_on_disk and not force_reload_images:
            self.__train_imgs = np.load(RingCameraConvnet.np_train_images_file)
            self.__train_labels = np.load(RingCameraConvnet.np_train_labels_file)
            self.__val_imgs = np.load(RingCameraConvnet.np_val_images_file)
            self.__val_labels = np.load(RingCameraConvnet.np_val_labels_file)
            self.__test_imgs = np.load(RingCameraConvnet.np_test_images_file)
            self.__test_labels = np.load(RingCameraConvnet.np_test_labels_file)
            self.__data_loaded = True
        else:
            self.__confirm_labeled_csv_is_defined('Loading image and label data from CSV')

            # load in csv file (file_path --> label)
            img_dict = get_img_dict_from_csv(self.__labeled_images_csv)

            # map labels to consider to integer values used by net
            str_to_int_label_dict = {str_label: int_label for int_label, str_label in self.__label_dict.items()}

            # convert labels to integer values
            img_dict = {img_path: str_to_int_label_dict[str_label] for img_path, str_label in img_dict.items() if str_label in str_to_int_label_dict}

            # get lists of equal order of (train, val, test) images and labels that are split by a requested percentage on a class by class
            # basis.  this way, if you ask for 60% training data, you'll get 60% of each class (which aren't going to be equally represented)
            train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = split_data_into_groups_bucketize(img_dict, max_images, max_images_per_class, force_even_distribution)

            # dump out histograms
            display_class_histogram(os.linesep + 'Histogram for training data:', train_imgs, img_dict, self.__label_dict)
            display_class_histogram(os.linesep + 'Histogram for validation data:', val_imgs, img_dict, self.__label_dict)
            display_class_histogram(os.linesep + 'Histogram for test data:', test_imgs, img_dict, self.__label_dict)

            # convert list of images to list of tensors
            print(f'Coverting {len(train_imgs) + len(val_imgs) + len(test_imgs)} images to tensor...')
            train_imgs = list(map(convert_img_to_tensor, train_imgs, repeat(img_size)))
            val_imgs = list(map(convert_img_to_tensor, val_imgs, repeat(img_size)))
            test_imgs = list(map(convert_img_to_tensor, test_imgs, repeat(img_size)))

            # convert lists to np array
            self.__train_imgs = np.array(train_imgs)
            self.__train_labels = np.array(train_labels)
            self.__val_imgs = np.array(val_imgs)
            self.__val_labels = np.array(val_labels)
            self.__test_imgs = np.array(test_imgs)
            self.__test_labels = np.array(test_labels)
            self.__data_loaded = True

            # save the results for next time!
            np.save(RingCameraConvnet.np_train_images_file, self.__train_imgs)
            np.save(RingCameraConvnet.np_train_labels_file, self.__train_labels)
            np.save(RingCameraConvnet.np_val_images_file, self.__val_imgs)
            np.save(RingCameraConvnet.np_val_labels_file, self.__val_labels)
            np.save(RingCameraConvnet.np_test_images_file, self.__test_imgs)
            np.save(RingCameraConvnet.np_test_labels_file, self.__test_labels)

    """
    Sets the final convolution block 'trainable', makes assumptions about the shape / layer index of the model.
    """
    def configure_pretrained_model_for_finetuning(self):
        # Pick a low learning rate such as to not alter the learned representations too much...
        self.__model.optimizer.learning_rate.assign(1e-5)
        self.__conv_base.trainable = True
        for layer in self.__conv_base.layers[:-4]:
            layer.trainable = False
    
    """
    Compile a model which reuses a well trained convolutional base with a custom classifier
    """
    def compile_model_pretrained(self, img_shape=None):
        use_augmentation = True

        data_augmentation = tf.keras.layers.RandomFlip('horizontal')

        self.__conv_base = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        self.__conv_base.trainable = False

        input_shape = None
        if img_shape is None:
            self.__confirm_data_is_loaded('Creating a model with no explicit input shape')
            input_shape = self.__train_imgs[0].shape
        else:
            input_shape = img_shape
        
        # input layer
        input_layer = tf.keras.layers.Input(shape=input_shape)

        # optional data augmentation
        x = data_augmentation(input_layer) if use_augmentation else input_layer

        # preprocess the image tensors to what vgg prefers
        x = tf.keras.applications.vgg16.preprocess_input(x)

        # the main convolutional network
        x = self.__conv_base(x)

        # flatten to 1 dim
        x = tf.keras.layers.Flatten()(x)

        # apply drop-out to help with overfitting
        x = tf.keras.layers.Dropout(0.5)(x)

        # output layer
        num_outputs = len(self.__label_dict)
        output_layer = tf.keras.layers.Dense(num_outputs, activation='softmax')(x)

        # create a model utilizing our feature-extraced conv network
        self.__model = tf.keras.Model(input_layer, output_layer, name='pretrained_cnn_model')

        self.__model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.95),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    """
    Compile the internal keras model from scratch
    """
    def compile_model(self, img_shape=None, use_augmentation=False):
        input_shape = None
        use_resizing = False

        if img_shape is None:
            # if no explicit input shape is requested, there better be data to tell us how it should be sized!
            self.__confirm_data_is_loaded('Creating a model with no explicit input shape')
            input_shape = self.__train_imgs[0].shape
        else:
            if self.__data_loaded:
                # we will use resizing if we can see that the loaded data does not match the requested size
                use_resizing = self.__train_imgs[0].shape[:2] != img_shape[:2]
                input_shape = self.__train_imgs[0].shape
            else:
                # if no data is loaded, make the input shape match the specification
                input_shape = img_shape

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip('horizontal'),
                #tf.keras.layers.RandomRotation(0.1),
                #tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomContrast(0.5)
            ],
            name='data_augmentation'
        )

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = inputs

        # resize things if need be
        if use_resizing:
            x = tf.keras.Sequential(
                [
                    tf.keras.layers.Resizing(*img_shape[:2]),
                    tf.keras.layers.Rescaling(1./255)
                ],
                name='rescaling_and_resizing'
            )(inputs)
        
        # augment data to increase resistance to overfitting if needed
        if use_augmentation:
            x = data_augmentation(x)

        # convolutional layers
        x = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3)(x)
        x = tf.keras.layers.SeparableConv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=3, activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=3, activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x = tf.keras.layers.SeparableConv2D(filters=512, kernel_size=3, activation='relu')(x)

        # ouput classification layer(s)
        x = tf.keras.layers.Flatten()(x)
        #x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        num_outputs = len(self.__label_dict)
        outputs = tf.keras.layers.Dense(num_outputs, activation='softmax')(x)

        # define a keras model
        self.__model = tf.keras.Model(inputs=inputs, outputs=outputs, name='turkeynet')

        # compile the model with chosen loss metrics and optimizer algorithm
        self.__model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    """
    Load model from disk using the best epoch checkpoint
    """
    def load_model(self):
        self.__model = tf.keras.models.load_model(RingCameraConvnet.keras_model_folder)

    """
    Train the model, and evaluate its performance on the test data
    """
    def train_and_evaluate(self):
        self.__confirm_data_and_model_are_valid('Training')

        if RingCameraConvnet.use_cpu:
            tf.config.set_visible_devices([], 'GPU')
        
        # get the model and show a brief summary of it
        print(self.__model.summary())
        tf.keras.utils.plot_model(model=self.__model, to_file=str(RingCameraConvnet.model_folder_path / 'model.png'), show_shapes=True)

        # train the model, allowing user CTRL-C to quit the process early
        try:
            self.__model.fit(
                self.__train_imgs,
                self.__train_labels,
                validation_data=(self.__val_imgs, self.__val_labels),
                epochs=100,
                callbacks=[
                    #tf.keras.callbacks.EarlyStopping(patience=15),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=RingCameraConvnet.keras_model_folder,
                        save_weights_only=False,
                        save_best_only=True,
                        monitor='val_accuracy',
                        mode='max'
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_accuracy',
                        factor=0.8,
                        patience=5,
                        min_lr=0.00001
                    ),
                    tf.keras.callbacks.TensorBoard(
                        log_dir=os.path.abspath(str(RingCameraConvnet.model_folder_path / "tensorboard_logs"))
                    )
                ],
                batch_size=64
            )
        except KeyboardInterrupt:
            print(os.linesep + "Killed fit early via CTRL-C...")
            pass

        # load in the weights from the epoch with the maximum accuracy
        self.load_model()

        # evaluate the performance on the never-before-seen test data,
        # measuring the inference capability of the model
        loss, accuracy = self.__model.evaluate(
            self.__test_imgs,
            self.__test_labels,
            batch_size=64
        )
        print(f"Loss: {loss}, Accuracy: {accuracy}")

    """
    Evaluate a trained model
    """
    def evaluate_only(self, show_predict_loop=False):
        self.__confirm_model_is_loaded('Model evaluation')

        print(self.__model.summary())

        # evaluate the performance on the never-before-seen test data,
        # measuring the inference capability of the model
        self.__model.evaluate(
            self.__test_imgs,
            self.__test_labels,
            batch_size=64
        )

        # show the labeled sample for fun, confirming the inference is (mostly) correct
        if show_predict_loop:
            image_review_ms = 1500
            predictions = self.__model.predict(self.__test_imgs)

            predictions = [tf.argmax(pred).numpy() for pred in predictions]
            self.__report_on_mispredictions(predictions)

            # loop over each prediction made on the test data
            for idx, img in enumerate(self.__test_imgs):
                int_label = predictions[idx]
                label = self.__label_dict[int_label]

                # show the image for a moment before moving on to the next
                cv2.imshow(label, img)
                keyPressed = cv2.waitKeyEx(image_review_ms) & 0xFF
                if keyPressed == ESC_KEY:
                    cv2.destroyAllWindows()
                    break
                cv2.destroyAllWindows()

    """
    Create prediciton(s) for a batch of images
    """
    def predict(self, images):
        self.__confirm_model_is_loaded('Prediction')

        # For small batches, use model.call
        if images.shape[0] < 10:
            predictions = self.__model(images)
        else:
            predictions = self.__model.predict(images)

        predictions = [tf.argmax(pred).numpy() for pred in predictions]
        predictions = [self.__label_dict[int_pred] for int_pred in predictions]
        return predictions

    """
    Create predictions for all images in a given folder which are not present in the labeled
    csv file.  This effectively can be used to bootstrap training data by prediciting new labels.
    All of the predicitons would need to have their labels confirmed before they can
    be incorporated with the (train, val, test) set.
    """
    def prediction_on_unlabeled_data(self, img_folder):
        self.__confirm_labeled_csv_is_defined('Prediction')
        self.__confirm_data_and_model_are_valid('Predicition')

        prediction_batch_size = 1000

        labeled_imgs = get_img_dict_from_csv(self.__labeled_images_csv)
        all_img_paths = get_all_images_in_folder(img_folder)
        all_img_paths = list(filter(lambda x: x not in labeled_imgs, all_img_paths))
        total_imgs_to_predict = len(all_img_paths)

        # load the model from the best epoch checkpoint
        print(self.__model.summary())

        output_dict = {}

        # make predicitons on batches so we don't run out of memory!
        total_batches = math.ceil(total_imgs_to_predict / prediction_batch_size)
        for batch_num in range(0, total_batches):
            print(f'Predicting batch [{batch_num + 1} / {total_batches}]...')
            start_idx = batch_num * prediction_batch_size
            end_idx = (batch_num + 1) * prediction_batch_size
            batch_img_paths = all_img_paths[start_idx:end_idx]
            batch_imgs = list(map(convert_img_to_tensor, batch_img_paths))
            batch_imgs = np.array(batch_imgs)
            predictions = self.__model.predict(batch_imgs)

            # loop over each prediction made on the test data
            for idx, img_path in enumerate(batch_img_paths):
                prediction = predictions[idx]
                int_label = tf.argmax(prediction).numpy()
                label = self.__label_dict[int_label]
                output_dict[img_path] = label

        save_img_dict_to_csv(output_dict, 'unlabeled_imgs.csv')

if __name__ == '__main__':
    labeled_csv_path = './ring_camera/ring_data/sept_through_nov_2023/frames/400max/labeled_unique_0p999.csv'
    net = RingCameraConvnet(labeled_csv_path)

    #load in data
    net.load_data(force_reload_images=False)

    # create or load a model
    net.compile_model(use_augmentation=True)#, img_shape=(169, 300, 3))
    #net.compile_model_pretrained()
    #net.load_model()

    # train the model
    net.train_and_evaluate()
    #net.configure_pretrained_model_for_finetuning()

    # conduct inference
    #net.evaluate_only(show_predict_loop=True)
    #net.predict_on_unlabeled_data('./ring_camera/ring_data/sept_through_nov_2023/frames/400max')