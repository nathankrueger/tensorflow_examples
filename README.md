# tensorflow_examples

This repository mostly captures my learning of the TensorFlow framework, and some common utilities for handling video and image data.

There also exist a handful of scripts for downloading Ring camera videos, extracting frames, resizing images, and detecting similar images
by computing cosine similarity of feature vectors provided by a pretrained CNN.

# Flow for Ring camera detection:
* Download .mp4 videos via [ring_downloader.py](ring_camera/ring_downloader.py)
* Extract frames into .jpg via [frame_extractor.py](img_vid_processing/frame_extractor.py)
* Detect duplicate images via [detect_duplicate_image.py](img_vid_processing/detect_duplicate_image.py)
* Label images via [label_image.py](data_labeler/label_image.py)
* Train CNN via [ring_camera_convnet.py](ring_camera/ring_camera_convnet.py)
* Deploy a script [ring_events.py](ring_camera/ring_events.py) which listens for ring events, grabs the latest snapshot from the camera associated with the event, runs it through the CNN, generates a prediction, and executes any user-defined callbacks associated with the event.
