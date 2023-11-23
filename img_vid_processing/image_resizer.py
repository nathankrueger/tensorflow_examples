import os
from pathlib import Path
import cv2
import argparse
import concurrent.futures
import queue
import threading
import glob

image_extensions = ['.jpg']
print_q = queue.Queue()

def print_thread():
    while True:
        msg = print_q.get()
        print(msg)

def resize_img(image_path, total_images, image_num, output_dir, max_dim):
    print_q.put(f"[{image_num} / {total_images}] -- Resizing image {image_path}...")

    image = cv2.imread(image_path)

    # Height, Width, Channels
    width = image.shape[1]
    height = image.shape[0]

    filename = str(Path(os.path.abspath(output_dir)) / str(os.path.split(image_path)[-1]))

    resized = None
    if width > height:
        ratio = max_dim / width
        resized = cv2.resize(image, (max_dim, int(height * ratio)), interpolation=cv2.INTER_AREA)
    else:
        ratio = max_dim / height
        resized = cv2.resize(image, (int(width * ratio), max_dim), interpolation=cv2.INTER_AREA)

    cv2.imwrite(filename, resized)

def main():
    parser = argparse.ArgumentParser(description='Resize images')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--recursive', action='store_true', default=False)
    parser.add_argument('--max_dimension', type=int, required=True)
    args=parser.parse_args()

    all_images = []
    for img_type in image_extensions:
        all_images.extend(glob.glob(str(Path(args.input_dir) / f'*{img_type}'), recursive=args.recursive))

    # Create a print thread
    t = threading.Thread(target=print_thread, daemon=True)
    t.start()

    os.makedirs(args.output_dir, exist_ok=True)

    # Launch the workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(all_images))) as exec:
        for i, img_path in enumerate(all_images):
            exec.submit(resize_img, img_path, len(all_images), i + 1, args.output_dir, args.max_dimension)
        
        # Wait for all futures to be completed
        exec.shutdown(wait=True)

if __name__ == "__main__":
    main()