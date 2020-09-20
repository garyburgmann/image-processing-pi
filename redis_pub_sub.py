import time
import pickle
from typing import List, Tuple, Dict

import numpy as np

from app.redis import get_redis
from app.config import DEFAULT_ARGS
from app.utils import (
    get_classifier,
    run_classifier
)

server_args = DEFAULT_ARGS
server_args.lite = False

clf = get_classifier(server_args)

r = get_redis()
p = r.pubsub(ignore_subscribe_messages=True)
count = 0


def classify_frame(frame: np.ndarray) -> Tuple[List[Dict], int]:
    global clf
    results, num_boxes = run_classifier(img=frame, clf=clf)
    print(f'{__name__}: {results}, {num_boxes}')
    return results, num_boxes


def my_handler(message):
    global count
    count += 1
    frame = pickle.loads(message['data'])
    results, num_boxes = classify_frame(frame)
    print('my_handler: ', count)
    r.publish('ack', pickle.dumps([results, num_boxes]))


# channel_map = {'det': my_handler}

p.subscribe('det')

# time = time.time()
while True:
    message = p.get_message()
    if message:
        my_handler(message)
        # time.sleep(0.0001)
