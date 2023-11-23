import os
import time
from pathlib import Path
import numpy as np
import cv2
import argparse
import concurrent.futures
import queue
import threading
import glob

video_extensions = ['.mp4']
print_q = queue.Queue()

def print_thread():
    while True:
        msg = print_q.get()
        print(msg)

def resize_img(image, max_dim):
    # height, width, channels
    width = image.shape[1]
    height = image.shape[0]

    if width > height:
        ratio = max_dim / width
        return cv2.resize(image, (max_dim, int(height * ratio)), interpolation=cv2.INTER_AREA)
    else:
        ratio = max_dim / height
        return cv2.resize(image, (int(width * ratio), max_dim), interpolation=cv2.INTER_AREA)

def extract_frames(video_path, total_videos, video_num, output_dir, frame_interval, resize=None):
    print_q.put(f"[{video_num} / {total_videos}] -- Extracting frames for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_num = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        frame_num += frame_interval
        ret, frame = cap.read()

        if resize:
            frame = resize_img(frame, resize)

        # quit if we've loaded the last frame
        if frame_num > total_frames:
            break

        frame_filename = Path(os.path.abspath(output_dir)) / str(os.path.split(os.path.splitext(video_path)[0])[1] + f'__frame_{frame_num:06d}.jpg')
        cv2.imwrite(str(frame_filename), frame)

    cap.release()

def main():
    parser = argparse.ArgumentParser(description='Extract images from a movie')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--recursive', action='store_true', default=False)
    parser.add_argument('--frame_interval', type=int, default=100)
    parser.add_argument('--max_dimension', type=int, required=False)
    args=parser.parse_args()

    all_videos = []
    for vid_type in video_extensions:
        all_videos.extend(glob.glob(str(Path(args.input_dir) / f'*{vid_type}'), recursive=args.recursive))

    # create a print thread
    t = threading.Thread(target=print_thread, daemon=True)
    t.start()

    os.makedirs(args.output_dir, exist_ok=True)

    # launch the workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(all_videos))) as exec:
        for i, video_path in enumerate(all_videos):
            exec.submit(extract_frames, video_path, len(all_videos), i + 1, args.output_dir, args.frame_interval, args.max_dimension)
        
        # wait for all futures to be completed
        exec.shutdown(wait=True)

if __name__ == "__main__":
    main()