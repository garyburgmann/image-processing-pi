import pickle
from typing import List, Tuple, Dict
import time
import os

import numpy as np
import tensorflow as tf

from .config import (
    API_SERVER,
)

print("loading model ...")
start_time = time.time()
clf = tf.saved_model.load(API_SERVER.model_path)
print(f"model loaded in {time.time() - start_time} seconds")


def predict_frame(frame: np.ndarray) -> Dict:
    global clf

    converted_img  = tf.image.convert_image_dtype(
        frame,
        tf.uint8
    )[tf.newaxis, ...]
    
    print("starting inference ...")
    start_time = time.time()
    res = clf(converted_img)
    print(f"inference time: {time.time() - start_time} seconds")

    # .numpy() will error on num_detections, pop and add back in
    num_detections = int(res.pop('num_detections'))

    predictions = {
        key: value[0, :num_detections].numpy()
        for key, value in res.items()
    }
    predictions['num_detections'] = num_detections

    return predictions
