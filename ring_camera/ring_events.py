import time
import configparser
import json
import os
import tempfile
import cv2
import queue
import playsound

import tensorflow as tf
import numpy as np

from pathlib import Path
from threading import Thread, Lock
from datetime import datetime

from ring_doorbell import Ring, Auth, exceptions
from oauthlib.oauth2 import MissingTokenError
from gtts import gTTS

from ring_camera_convnet import RingCameraConvnet

# this is a critical parameter... the motion event vs. snaphot content update are
# unfortunately asynchronous in ring's API.  if we set this param too large, then
# the delay to report the prediciton witll be large, and effectively the inference 
# latency will be increased.  if we set this param too small, then we can get a stale
# snapshot from 30s to 5min old, and the inference will be out of sync with the current evt.
evt_to_snapshot_delay = 5

img_review_delay = 5

cache_file = Path(os.path.dirname(__file__)) / 'test_token.cache'
ring_ini_file = Path(os.path.dirname(__file__)) / 'ring_api.ini'
ring = None

callback_queue = queue.Queue()
evt_queue = queue.Queue()
imshow_queue = queue.Queue()
callback_lock = Lock()

# a dictionary of event label to list of tuples
# each tuple is a pair of (function, (arg1, .. argN))
callbacks_dict = {}

def token_updated(token):
    cache_file.write_text(json.dumps(token))

def otp_callback():
    auth_code = input("2FA code: ")
    return auth_code

def resize_img(image, max_dim):
    # Height, Width, Channels
    width = image.shape[1]
    height = image.shape[0]

    if width > height:
        ratio = max_dim / width
        return cv2.resize(image, (max_dim, int(height * ratio)), interpolation=cv2.INTER_AREA)
    else:
        ratio = max_dim / height
        return cv2.resize(image, (int(width * ratio), max_dim), interpolation=cv2.INTER_AREA)

"""
Download the most recent snapshot from the Ring API.  Motion events trigger snapshot
updates, but you must be careful, the motion event can easily come sooner than the snapshot
is available, resulting in a race condition.  Snapshots are sized 640x360x3.
"""
def retrieve_image(camera_name):
    result = None
    try:
        device = ring.get_device_by_name(camera_name)
        img_content = device.get_snapshot()
        image_bytes = np.asarray(bytearray(img_content), dtype="uint8")
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        result = image
    except Exception as ex:
        print(f'Failed to retrieve snapshot: {ex}')

    return result

def on_motion(event):
    print(f"Motion detected at {datetime.now()}")
    print(f"Event details: {event}")
    evt_queue.put(event.device_name)

def speak(evt, tld='en'):
    label = evt[1]
    device_name = evt[2]

    text = f"There's a {label} at your {device_name}"
    with tempfile.NamedTemporaryFile() as tmpfile:
        try:
            gTTS(text=text, lang='en', tld=tld, slow=False).save(tmpfile.name)
            playsound.playsound(tmpfile.name)
        except Exception as ex:
            print(f'Failed to play sound {ex}')

def callback_thread():
    while True:
        # format is callback, img, label, device_name
        q_item = callback_queue.get()
        callback_fn_tup = q_item[0]
        img = q_item[1]
        label = q_item[2]
        device_name = q_item[3]

        # function args are optional
        fn = callback_fn_tup[0]
        if len(callback_fn_tup) == 1:
            args = ((img, label, device_name),)
        else:
            args = callback_fn_tup[1]
            args = ((img, label, device_name), *args)

        # callback signature is expected to be (img, label, device_name), optional_args0, ... optional_argsN
        fn(*args)

def predict_thread():
    global callback_lock

    # initialize the tensorflow model
    net = RingCameraConvnet()
    net.load_model()
    kerasmodel = net.get_model()

    img_height = kerasmodel.input.shape[1]
    img_width = kerasmodel.input.shape[2]

    while True:
        print('Waiting for next ring event...')
        device_name = evt_queue.get()
        time.sleep(evt_to_snapshot_delay)
        img = retrieve_image(device_name)

        # in case Ring has an issue
        if img is None:
            continue

        img = resize_img(img, max(img_height, img_width))
        imshow_queue.put(img)
        predicted_label = net.predict(np.expand_dims(img, axis=0))[0]

        # guard access to the callback dictionary
        with callback_lock:
            callbacks_to_run = []
            if device_name in callbacks_dict:
                callbacks_to_run += callbacks_dict[device_name]
            
            # None entries imply run for any camera type
            if None in callbacks_dict:
                callbacks_to_run += callbacks_dict[None]

            # queue up any defined callbacks
            for cb in callbacks_to_run:
                callback_queue.put((cb, img, predicted_label, device_name))

def initialize_ring(install_listener):
    if cache_file.is_file():
        auth = Auth("NEW_PROJECT/1.0", json.loads(cache_file.read_text()), token_updated)
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

    global ring
    ring = Ring(auth)
    ring.update_data()

    # register the callback function for motion events
    if install_listener:
        ring.start_event_listener()
        ring.add_event_listener_callback(on_motion)

def main():
    # don't use GPU for running one prediction at a time...
    tf.config.set_visible_devices([], 'GPU')

    # initialize the ring library
    try:
        install_listener = True
        print(f'Initializing Ring (install_listener={install_listener})...')
        initialize_ring(install_listener)
        print('Finished initializing Ring...')
    except Exception as ex:
        print(f'Failed to initialize Ring with exception: {ex}')

    # add a thread which runs the inference and triggers
    # callbacks to occur on the callback_thread as needed
    pt = Thread(target=predict_thread, daemon=True)
    pt.start()

    # respond to any callbacks
    ct = Thread(target=callback_thread, daemon=True)
    ct.start()

    evt_queue.put('Garage Cam')

    # add some useful callbacks
    with callback_lock:
        # image name generator
        def get_next_img():
            i = 0
            while True:
                yield f'{i}.jpg'
                i += 1

        # add a callback which saves an image 0.jpg, 1.jpg, etc for each 'Garage Cam' event
        callbacks_dict['Garage Cam'] = [(lambda args: cv2.imwrite('img.jpg', args[0]),)]

        # add a callback which prints out whatever label was detected on any camera
        callbacks_dict[None] = [(lambda args: print(f'{args[1].upper()} detected on {args[2]}'),)]

        # add a callback which speaks whatever label was detected by any camera
        callbacks_dict[None] += [(speak,('ie',))]

    # keep the main thread running to listen for image review events.
    # the cv2 library functions must be executed from the main thread.
    try:
        idx = 0
        img_shown = False
        while True:
            print('Main thread waiting for imshow events...')

            try:
                img = imshow_queue.get(timeout=img_review_delay)
            except queue.Empty as ex:
                if img_shown:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    img_shown = False
                continue

            cv2.destroyAllWindows()
            cv2.imshow(f'motion_capture_{idx}', img)
            img_shown = True
            cv2.waitKey(1)

            idx += 1

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()