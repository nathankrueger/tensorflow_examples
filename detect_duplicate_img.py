import tensorflow as tf
import numpy as np
from scipy.spatial import distance
import cv2
import csv
import argparse
import os, glob
import math, random
import concurrent.futures
import queue, threading
from timeit import default_timer as timer
from datetime import timedelta
from pathlib import Path
from PIL import Image

img_extensions = ['.jpg']
imagenet_sz = (224, 224, 3)

progress_lock = threading.Lock()
total_comparisons = 0
completed = 0
print_q = queue.Queue()

def get_progress_msg(num_completed):
    global completed
    with progress_lock:
        completed += num_completed
    percentage = (float(completed) / float(total_comparisons)) * 100.0
    return f'Comparing [{completed} {percentage:0,.2f}% of {total_comparisons}] cosine similarities...'

def print_thread():
    while True:
        msg = print_q.get()
        print(msg)

def get_feature_vector_img(img, model):
    img = cv2.imread(img)
    img = cv2.resize(img, imagenet_sz[0:2])
    vector = model.predict(img.reshape((1,) + imagenet_sz))
    return vector

def get_feature_vectors_dict(img_paths, folder, model):
    loaded_imgs = []
    result = {}
    for img_path in img_paths:
        img = cv2.imread(str(Path(folder) / img_path))
        img = cv2.resize(img, imagenet_sz[0:2])
        loaded_imgs.append(img)

    feature_vectors = model.predict(np.array(loaded_imgs))
    for idx, img_path in enumerate(img_paths):
        result[img_path] = feature_vectors[idx]

    return result

def get_cosine_similarity_vec(vec1, vec2):
    return 1 - distance.cosine(vec1, vec2)

def get_cosine_similarity_img(img1, img2, model):
    v1 = np.squeeze(get_feature_vector_img(img1, model))
    v2 = np.squeeze(get_feature_vector_img(img2, model))
    return get_cosine_similarity_vec(v1, v2)

def get_all_combinations(array):
    idx = np.stack(np.triu_indices(len(array), k=1), axis=-1)
    return np.array(array)[idx]

def gen_batch(iterable, batch_sz):
    l = len(iterable)
    for ndx in range(0, l, batch_sz):
        yield iterable[ndx:min(ndx+batch_sz, l)]

def calculate_batch_of_cosine_similarities(batch, img_path_to_feature_vec, sim_tup_list, lock):
    print_interval = 10000
    try:
        batch_tup_list = []
        batch_sz = len(batch)
        for idx, combo in enumerate(batch):
            path1 = combo[0]
            path2 = combo[1]

            # this reading should be thread-safe as we're not modifying anything
            img1_fv = np.squeeze(img_path_to_feature_vec[path1])
            img2_fv = np.squeeze(img_path_to_feature_vec[path2])
            sim = get_cosine_similarity_vec(img1_fv, img2_fv)
            batch_tup_list.append((path1, path2, sim))

            # give the outside world some details on our progress
            if (idx % print_interval == 0 and idx != 0):
                print_q.put(get_progress_msg(print_interval))
        
        # report the last chunk
        print_q.put(get_progress_msg(batch_sz % print_interval))

        # don't trust the GIL here...
        with lock:
            sim_tup_list.extend(batch_tup_list)

    except Exception as ex:
        print(ex)

def catalog_duplicates(input_csv, output_csv_path, thresholds):
    sim_tup_list = []

    start = timer()
    if os.path.exists(input_csv):
        with open(input_csv, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                sim_tup_list.append((row[0], row[1], row[2]))

    print(f"Read in {len(sim_tup_list)} image pairs in {str(timedelta(seconds=timer() - start))}")

    for threshold in thresholds:
        output_unique_csv = Path(output_csv_path) / f"unique_{str(threshold).replace('.', 'p')}.csv"
        output_clone_csv = Path(output_csv_path) / f"clones_{str(threshold).replace('.', 'p')}.csv"
        write_similarities_for_threshold(sim_tup_list, output_unique_csv, output_clone_csv, threshold)

def write_similarities_for_threshold(sim_tup_list, output_unique_csv, output_clone_csv, threshold):
    all_images = {}
    sim_dict = {}
    clones = {}

    print_interval = 100000
    total_pairs = len(sim_tup_list)
    print(f"Determining duplicates among {total_pairs} pairs...")

    for idx, tup in enumerate(sim_tup_list):
        imageA = tup[0]
        imageB = tup[1]
        similarity = float(tup[2])

        all_images[imageA] = None
        all_images[imageB] = None

        if idx % print_interval == 0:
            percent_complete = (float(idx) / float(total_pairs)) * 100.0
            print(f"[{percent_complete:0.2f}% of {total_pairs}] pairs processed...")

        # only consider images that are sufficiently similar
        if similarity >= threshold:
            imageA_is_clone = imageA in clones
            imageB_is_clone = imageB in clones

            # only one is a clone
            if imageA_is_clone != imageB_is_clone:
                old_clone = imageA if imageA_is_clone else imageB
                new_clone = imageA if imageB_is_clone else imageB

                # be careful not to record an image as a clone of itself
                if new_clone not in sim_dict:
                    relative = clones[old_clone]
                    sim_dict[relative][new_clone] = None
                    clones[new_clone] = relative

            # neither are clones (yet)
            elif not imageA_is_clone and not imageB_is_clone:
                # imageB is the clone by convention

                # be careful not to record an image as a clone of itself
                if imageB not in sim_dict:
                    clones[imageB] = imageA
                    if imageA not in sim_dict:
                        sim_dict[imageA] = {imageB:None}
                    else:
                        sim_dict[imageA][imageB] = None
                else:
                    sim_dict[imageB][imageA] = None
                    clones[imageA] = imageB

    print(f'{len(clones)} similar images identified for removal...')

    # keep all images that are not in the to remove list
    print(f"Writing images that are unique with threshold of {threshold} to file {output_unique_csv}")
    sorted_images = sorted(all_images.keys())
    with open(output_unique_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img in sorted_images:
            if img in clones:
                continue
            else:
                writer.writerow([img])

    # write only the images that should be removed
    print(f"Writing images that are similar with threshold of {threshold} to file {output_clone_csv}")
    sorted_images = sorted(sim_dict.keys())
    with open(output_clone_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for original_dup in sorted_images:
            for clone in sorted(sim_dict[original_dup]):
                writer.writerow([clone])

def review_duplicates(csv_path, img_folder):
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                img_path = os.path.abspath(str(Path(img_folder) / row[0]))
                cv2.imshow(row[0], cv2.imread(img_path))
                keypress = cv2.waitKeyEx(delay=0) & 0xFF

                # quit early!
                if keypress == 27:
                    cv2.destroyAllWindows()
                    break
                cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Determine the pairwise cosine similarity of a batch of images')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True, help='Path relative to input dir where the similarity csv file will be written')
    parser.add_argument('--size_limit', type=int, default=-1)
    args=parser.parse_args()

    # find all images
    all_images = []
    for img_type in img_extensions:
        full_img_paths = glob.glob(str(Path(os.path.abspath(args.input_dir)) / f'*{img_type}'), recursive=False)
        for img_path in full_img_paths:
            rel_path = os.path.split(img_path)[1]
            all_images.append(rel_path)

    # load a pretrained model adept at feature extraction
    vgg16 = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=True,
        pooling='max',
        input_shape=imagenet_sz
    )

    # remove the classification layer, we only want the feature vector
    basemodel = tf.keras.Model(
        inputs=vgg16.input,
        outputs=vgg16.get_layer('fc2').output
    )

    # reduce the size for testing or truncated results
    if args.size_limit > 0:
        random.shuffle(all_images)
        all_images = all_images[:args.size_limit]

    # get mapping of each image to each feature vector
    total_images = len(all_images)
    img_path_to_feature_vec = {}
    batch_size = 1000
    num_batches = math.ceil(total_images / batch_size)
    for batch_num in range(0, num_batches):
        print(f'Calculating feature vectors for batch [{batch_num + 1} / {num_batches}]')
        start_idx = batch_num * batch_size
        end_idx = (batch_num + 1) * batch_size
        curr_batch_paths = all_images[start_idx:end_idx]
        curr_dict = get_feature_vectors_dict(curr_batch_paths, args.input_dir, basemodel)
        img_path_to_feature_vec.update(curr_dict)

    # determine all the combinations, this is going to hurt.
    print(f"Calculating all pairwise combinations of {len(all_images)} images...")
    start = timer()
    combos = get_all_combinations(all_images)
    total_combos = combos.shape[0]
    global total_comparisons
    total_comparisons = total_combos
    print(f"Got {total_combos} in {str(timedelta(seconds=timer() - start))}")

    # Create a print thread
    t = threading.Thread(target=print_thread, daemon=True)
    t.start()

    # calculate cosine similarities among each pair
    # launch the workers as this is cpu-bound
    sim_tup_list = []
    num_threads = os.cpu_count() / 2
    lock = threading.Lock()
    start = timer()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as exec:
        for batch in gen_batch(combos, math.ceil(total_combos / num_threads)):
            exec.submit(calculate_batch_of_cosine_similarities, batch, img_path_to_feature_vec, sim_tup_list, lock)
        
        # Wait for all futures to be completed
        exec.shutdown(wait=True)
    end = timer()
    print(f"Calculating cosine similarities took {str(timedelta(seconds=end - start))}")

    # sort the list such that the pairs are ordered by similarity
    # with most similar at the top
    sim_tup_list.sort(key=lambda x: x[2], reverse=True)

    # write the results
    csv_path  = Path(args.input_dir) / args.output_csv
    print(f"Writing {total_combos} to {csv_path}...")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for tup in sim_tup_list:
            writer.writerow(list(tup))

if __name__ == '__main__':
    #duplicate_csv_output_folder = './ring_downloader/ring_data/sept_through_nov_2023/frames/400max'
    #catalog_duplicates('./ring_downloader/ring_data/sept_through_nov_2023/frames/400max/similarities.csv.full', duplicate_csv_output_folder, [0.99, 0.995, 0.999, 0.9995, 0.9999])
    #review_duplicates(csv_path=str(Path(duplicate_csv_output_folder) / 'unique_0p99.csv'), img_folder='./ring_downloader/ring_data/sept_through_nov_2023/frames/originals')
    main()