import json
import os
import time
import getpass
from pathlib import Path
import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import argparse

from ring_doorbell import Ring, Auth, exceptions
from oauthlib.oauth2 import MissingTokenError

max_retries=5
file_download_sleep_amt=5
user_input = False
cache_file = Path("test_token.cache")
ring_ini_file = os.path.abspath('ring_api.ini')

def token_updated(token):
    cache_file.write_text(json.dumps(token))

def otp_callback():
    auth_code = input("2FA code: ")
    return auth_code

# unused
def show_snapshot(ring, camera_name, filename=None):
    device = ring.get_device_by_name(camera_name)
    if filename is not None:
        successs = device.get_snapshot(filename='test.jpg', delay=1)
        if successs:
            img = mpimg.imread('test.jpg')
            plt.imshow(img)
            plt.show()
        else:
            img_content = device.get_snapshot()
            image_bytes = np.asarray(bytearray(img_content), dtype="uint8")
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            cv2.imshow(None, image)
            cv2.waitKey(0)

def get_history(ring, camera_name, limit=100):
    device = ring.get_device_by_name(camera_name)
    hist = device.history(limit=limit)
    return hist, device

def main():
    parser = argparse.ArgumentParser(description='Label images into a single class')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--limit', type=int, required=False, default=10000)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--camera_name', type=str, required=True)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--per_day_limit', type=int, default=10)

    args=parser.parse_args()    

    if cache_file.is_file():
        auth = Auth("NEW_PROJECT/1.0", json.loads(cache_file.read_text()), token_updated)
    else:
        username=''
        password=''
        if user_input:
            username = input("Username: ")
            password = getpass.getpass("Password: ")
        else:
            config = configparser.RawConfigParser()
            print(config.read(ring_ini_file))
            username = config['DEFAULT']['username']
            password = config['DEFAULT']['password']

        auth = Auth("NEW_PROJECT/1.0", None, token_updated)
        try:
            auth.fetch_token(username, password)
        except (MissingTokenError, exceptions.Requires2FAError):
            auth.fetch_token(username, password, otp_callback())

    ring = Ring(auth)
    ring.update_data()

    hist, device = get_history(ring, 'Garage Cam', limit=args.limit)    

    items_by_day = {}
    for item in hist:
        id = item['id']
        dt = item['created_at']

        date_key = dt.strftime('%Y-%m-%d')
        if date_key in items_by_day:
            items_by_day[date_key].append(item)
        else:
            items_by_day[date_key] = [item]

    if args.download:
        os.makedirs(args.output_dir, exist_ok=True)

    # Limit the items by day to 'per_day_limit' and attempt to get
    # a relatively even distribution across the 24hrs of the day by
    # taking a random sample.
    matched_items = []
    for date_key in items_by_day:
        curr_day_items = items_by_day[date_key]
        random_selected_items = np.random.choice(curr_day_items, size=min(args.per_day_limit, len(curr_day_items)), replace=False)
        matched_items.extend(random_selected_items)

    total_items = len(matched_items)
    for i, item in enumerate(matched_items):
        id = item['id']
        duration = item['duration']
        dt = item['created_at']
        event_type = item['cv_properties']['detection_type']
        date_key = dt.strftime('%Y-%m-%d')

        if not args.silent:
            print(f"[{i+1}/{total_items}] -- ID: {id} | time: {dt} | duration: {duration} | event_type:{event_type}")
        if args.download:
            filename = os.path.abspath(str(Path(args.output_dir) / f'{date_key}__{i}.mp4'))
            if os.path.exists(filename):
                os.remove(filename)

            retries = max_retries
            while retries > 0:
                try:
                    time.sleep(file_download_sleep_amt)
                    device.recording_download(recording_id=id, filename=filename)
                    retries = -1
                except Exception as e:
                    print(e)
                    retries -= 1


if __name__ == "__main__":
    main()