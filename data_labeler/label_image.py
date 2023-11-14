import os
import cv2
import glob
import argparse
import configparser
import csv
from pathlib import Path

image_extensions = ['.jpg']
config_section = 'DEFAULT'

review_amt_ms = 250
font_size = 4
font_color_bgr = (0, 0 , 255) 

esc_key = 27
left_key = 2
right_key = 3

def get_key_dict(ini_file):
    config = configparser.RawConfigParser()
    config.read(os.path.abspath(ini_file))

    key_dict = {}
    for key in config[config_section]:
        assert(len(key) == 1)
        key_dict[ord(key)] = config[config_section][key]
    
    return key_dict

def show_and_label_images(output_csv, keymap_file, images, start_idx, label_dict):
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

        # Reshow a pre-labeled image, as is needed
        # during navigation
        if img in label_dict:
            old_label = label_dict[img]
            cv2.putText(cv2_image, f'[{img_idx} / {total_images}]: {old_label}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color_bgr, 5, cv2.LINE_AA)

        cv2.imshow(img, cv2_image)
        
        img_navigate = False
        needs_valid_key = True
        label = None
        while needs_valid_key:
            keyPressed = cv2.waitKeyEx(0) & 0xFF

            # Support quiting early
            if keyPressed == esc_key:
                esc_requested = True
                break

            # Navigate forward
            if keyPressed == right_key:
                if img_idx < total_images - 2:
                    img_idx += 1
                    img_navigate = True
                    break
            
            # Navigate backward
            if keyPressed == left_key:
                if img_idx > 0:
                    img_idx -= 1
                    img_navigate = True
                    break

            # Image was labeled!
            if keyPressed in key_dict:
                label = key_dict[keyPressed]
                label_dict[img] = label
                needs_valid_key = False

                # Hide the image, then show it briefly with the selected label, for review
                cv2.destroyAllWindows()
                cv2_image = cv2.imread(img)
                cv2.putText(cv2_image, f'[{img_idx} / {total_images}]: {label}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color_bgr, 5, cv2.LINE_AA)
                cv2.imshow(img, cv2_image)
                cv2.waitKey(review_amt_ms)

        if img_navigate:
            continue

        if esc_requested:
            last_image = True
            break

        if img_idx == total_images - 1:
            last_image = True
        
        img_idx += 1

    # Serialize the CSV file
    with open(os.path.abspath(output_csv), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img in label_dict:
            writer.writerow([img, label_dict[img]])

def main():
    parser = argparse.ArgumentParser(description='Label images into a single class')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--keymap_file', type=str, required=True)
    parser.add_argument('--recursive', action='store_true', default=False)
    args=parser.parse_args()

    # Find all images and sort them, so the order that files are labeled
    # will be repeatable (normally, if the user hasn't skipped ahead)
    all_images = []
    for img_type in image_extensions:
        all_images.extend(glob.glob(str(Path(os.path.abspath(args.input_dir)) / f'*{img_type}'), recursive=args.recursive))
    all_images.sort()

    output_csv_path = os.path.abspath(args.output_csv)
    keymap_file_path = os.path.abspath(args.keymap_file)

    # Build up any pre-existing labeling in the dictionary
    start_idx = 0
    label_dict = {}
    if os.path.exists(output_csv_path):
        with open(output_csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                label_dict[row[0]] = row[1]
                start_idx += 1

    show_and_label_images(output_csv_path, keymap_file_path, all_images, start_idx, label_dict)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()