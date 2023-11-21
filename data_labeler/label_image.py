import os, glob, shutil
import argparse, configparser
import cv2
import csv
from pathlib import Path

image_extensions = ['.jpg']
config_section = 'DEFAULT'
default_keymap_file = 'keys.ini'

# how long a chosen label be flashed on the screen
review_amt_ms = 250

font_size = 4
font_thickness = 5
text_offset = (100, 100)
font_color_bgr = (0, 0 , 255)

# could be system dependent; these work on macOS
esc_key = 27
del_key = 40
left_key = 2
right_key = 3
up_key = 0
down_key = 1

def get_key_dict(ini_file):
    config = configparser.RawConfigParser()
    config.read(os.path.abspath(ini_file))

    key_dict = {}
    for key in config[config_section]:
        assert(len(key) == 1)
        key_dict[ord(key)] = config[config_section][key]
    
    return key_dict

def resize_image(image, max_dim):
    resized = None
    height = image.shape[0]
    width = image.shape[1]

    if max_dim <= height and max_dim <= width:
        return image

    if width > height:
        ratio = max_dim / width
        resized = cv2.resize(image, (max_dim, int(height * ratio)))
    else:
        ratio = max_dim / height
        resized = cv2.resize(image, (int(width * ratio), max_dim))

    return resized

def show_and_label_images(output_csv, keymap_file, images, start_idx, label_dict, only_show_labels, min_dim):
    key_dict = get_key_dict(keymap_file)

    esc_requested = False
    last_image = False
    total_images = len(images)
    print(f'Total images: {total_images}')
    img_idx = start_idx

    while not last_image:
        cv2.destroyAllWindows()

        img = images[img_idx]
        cv2_image = cv2.imread(img)
        cv2_image = resize_image(cv2_image, min_dim)

        # reshow a pre-labeled image, as is needed
        # during navigation
        if img in label_dict:
            old_label = label_dict[img]

            if only_show_labels is not None and old_label not in only_show_labels:
                img_idx += 1
                continue
            else:
                cv2.putText(cv2_image, f'[{img_idx} / {total_images - 1}]: {old_label}', text_offset, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color_bgr, font_thickness, cv2.LINE_AA)
        
        cv2.imshow(img, cv2_image)
        
        img_navigate = False
        needs_valid_key = True
        label = None
        while needs_valid_key:
            keyPressed = cv2.waitKeyEx(0) & 0xFF

            # support quitting early
            if keyPressed == esc_key:
                esc_requested = True
                break

            # navigate forward
            elif keyPressed == right_key:
                if img_idx < total_images - 1:
                    img_idx += 1
                    img_navigate = True
                    break

            # navigate backward
            elif keyPressed == left_key:
                if img_idx > 0:
                    img_idx -= 1
                    if only_show_labels is not None:
                        while True:
                            img = images[img_idx]
                            if img in label_dict:
                                if label_dict[img] not in only_show_labels:
                                    img_idx -= 1
                                else:
                                    break
                            else:
                                break

                    img_navigate = True
                    break

            # navigate to start
            elif keyPressed == up_key:
                img_idx = 0
                img_navigate = True
                break

            # naviate to end
            elif keyPressed == down_key:
                img_idx = total_images - 1
                img_navigate = True
                break

            # remove a label
            elif keyPressed == del_key:
                if img in label_dict:
                    del label_dict[img]
                    img_idx += 1
                    img_navigate = True
                    break

            # image was labeled!
            if keyPressed in key_dict:
                label = key_dict[keyPressed]
                label_dict[img] = label
                needs_valid_key = False

                # hide the image, then show it briefly with the selected label, for review
                cv2.destroyAllWindows()
                cv2_image = cv2.imread(img)
                cv2_image = resize_image(cv2_image, min_dim)
                cv2.putText(cv2_image, f'[{img_idx} / {total_images - 1}]: {label}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color_bgr, 5, cv2.LINE_AA)
                cv2.imshow(img, cv2_image)
                cv2.waitKey(review_amt_ms)

        if img_navigate:
            continue

        if esc_requested:
            break

        if img_idx == total_images - 1:
            last_image = True
        
        img_idx += 1

    # make a backup in case
    output_abspath = os.path.abspath(output_csv)
    if os.path.exists(output_abspath):
        shutil.copy(output_abspath, output_abspath + '.bak')
    
    # serialize the CSV file,
    with open(output_abspath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img in label_dict:
            writer.writerow([img, label_dict[img]])

def main():
    parser = argparse.ArgumentParser(description='Label images into a single class')
    parser.add_argument('--input_dir', type=str, required=False)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--input_csv', type=str, required=False, help='If defined, the labels will be loaded from input_csv instead of output_csv')
    parser.add_argument('--keymap_file', type=str, default=default_keymap_file)
    parser.add_argument('--resume', action='store_true', default=False, help='If set, labeling will start at the first unlabeled image')
    parser.add_argument('--only_show_labels', type=str, required=False, help='Supply a comma separated list of labels that we should restrict to reviewing')
    parser.add_argument('--recursive', action='store_true', default=False)
    parser.add_argument('--min_dimension', type=int, default=1920)
    args=parser.parse_args()

    output_csv_path = os.path.abspath(args.output_csv)
    keymap_file_path = os.path.abspath(args.keymap_file)

    # build up any pre-existing labeling in the dictionary
    labeled_data_start_idx = 0
    label_dict = {}
    csv_to_preload_with = os.path.abspath(args.input_csv) if args.input_csv is not None else output_csv_path
    if os.path.exists(csv_to_preload_with):
        with open(csv_to_preload_with, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                filepath = row[0]
                label = row[1]
                label_dict[filepath] = label
                labeled_data_start_idx += 1
    
    # find all images and sort them, so the order that files are labeled
    # will be repeatable (normally, if the user hasn't skipped ahead)
    if args.input_dir:
        all_images = []
        for img_type in image_extensions:
            all_images.extend(glob.glob(str(Path(os.path.abspath(args.input_dir)) / f'*{img_type}'), recursive=args.recursive))
        all_images.sort()
    else:
        all_images = list(label_dict.keys())

    show_and_label_images(
        output_csv_path,
        keymap_file_path,
        all_images,
        labeled_data_start_idx - 1 if args.resume else 0,
        label_dict,
        args.only_show_labels,
        args.min_dimension
    )
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()