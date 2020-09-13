import pickle
from typing import List, Tuple, Dict
import time

import numpy as np
import falcon

from app.utils import (
    parse_args,
    get_classifier,
    run_classifier
)
from app.config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    FLASK_ENDPOINT,
    DEFAULT_ARGS
)

api = falcon.API()

server_args = DEFAULT_ARGS
server_args.lite = False

clf = get_classifier(server_args)


def classify_frame(frame: np.ndarray) -> Tuple[List[Dict], int]:
    results, num_boxes = run_classifier(img=frame, clf=clf)
    print(f'{__name__}: {results}, {num_boxes}')
    return results, num_boxes


class PingResource:
    def on_get(self, req, resp):
        """ Handles GET request """
        resp.media = {'message': 'pong'}


class ClassifierResource:
    def on_post(self, req, resp):
        """ Handles POST request """
        body = req.stream.read()
        frame = pickle.loads(body)
        results, num_boxes = classify_frame(frame)
        resp.body = pickle.dumps([results, num_boxes])


api.add_route(f'/{FLASK_ENDPOINT}', ClassifierResource())
api.add_route('/ping', PingResource())
