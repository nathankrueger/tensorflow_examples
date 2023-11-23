import os
import time
import json
import getpass
import configparser, argparse
import concurrent.futures
import threading
import queue
import numpy as np

from pathlib import Path
from ring_doorbell import Ring, Auth, exceptions
from oauthlib.oauth2 import MissingTokenError

num_download_threads=5
max_retries=5
file_download_sleep_amt=5
user_input = False
cache_file = Path("test_token.cache")
ring_ini_file = os.path.abspath('ring_api.ini')

print_q = queue.Queue()

def token_updated(token):
    cache_file.write_text(json.dumps(token))

def otp_callback():
    auth_code = input("2FA code: ")
    return auth_code

def print_thread():
    while True:
        msg = print_q.get()
        print(msg)

def get_history(ring, camera_name, limit=100):
    device = ring.get_device_by_name(camera_name)
    hist = device.history(limit=limit)
    return hist, device

def get_download_msg(idx, total_items, id, dt, duration, event_type):
    return f"[{idx+1}/{total_items}] -- ID: {id} | time: {dt} | duration: {duration} | event_type:{event_type}"

def download_thread(output_dir, device, item, idx, total_items, silent):
    try:
        id = item['id']
        duration = item['duration']
        dt = item['created_at']
        event_type = item['cv_properties']['detection_type']
        date_key = dt.strftime('%Y-%m-%d')

        if not silent:
            print_q.put(get_download_msg(idx, total_items, id, dt, duration, event_type))

        filename = os.path.abspath(str(Path(output_dir) / f'{idx:05d}__{date_key}.mp4'))
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
    except Exception as ex:
        print(f'An error occurred in the dowload worker: {ex}')

def main():
    parser = argparse.ArgumentParser(description='Download ring camera data up to a specified limit')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--limit', type=int, required=False, default=10000)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--camera_name', type=str, default='Garage Cam')
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--per_day_limit', type=int, default=10)
    parser.add_argument('--start_id', type=int, default=0)

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

    hist, device = get_history(ring, args.camera_name, limit=args.limit)    

    items_by_day = {}
    for item in hist:
        dt = item['created_at']
        id = item['id']

        if int(id) > args.start_id:
            date_key = dt.strftime('%Y-%m-%d %h-%m-%s')
            if date_key in items_by_day:
                items_by_day[date_key].append(item)
            else:
                items_by_day[date_key] = [item]

    # limit the items by day to 'per_day_limit' and attempt to get
    # a relatively even distribution across the 24hrs of the day by
    # taking a random sample.
    matched_items = []
    for date_key in items_by_day:
        curr_day_items = items_by_day[date_key]
        random_selected_items = np.random.choice(curr_day_items, size=min(args.per_day_limit, len(curr_day_items)), replace=False)
        random_selected_items.sort()
        matched_items.extend(random_selected_items)
    total_items = len(matched_items)

    if args.download:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # create a print thread
        t = threading.Thread(target=print_thread, daemon=True)
        t.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_download_threads) as exec:
            for idx, item in enumerate(matched_items):
                exec.submit(download_thread, args.output_dir, device, item, idx, total_items, args.silent)
        
            # wait for all downloads to complete
            exec.shutdown(wait=True)
    else:
        # print what items are matched
        for idx, item in enumerate(matched_items):
            id = item['id']
            duration = item['duration']
            dt = item['created_at']
            event_type = item['cv_properties']['detection_type']
            print(get_download_msg(idx, total_items, id, dt, duration, event_type))

if __name__ == "__main__":
    main()