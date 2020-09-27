import pickle
from typing import List, Tuple, Dict
import time
import os

import numpy as np
import tensorflow as tf

from app.utils import (
    parse_args,
    get_classifier,
    get_quadrant_results,
    compare_quadrants_from_results
)
from app.config import (
    APP_SERVER,
)

# mimic args, as argparse will clash with gunicorn
# server_args = APP_SERVER.default_args
# server_args.lite = False

# bootstrap here to reduce latency on each request
# clf = get_classifier(server_args)

print("loading model ...")
start_time = time.time()
clf = tf.saved_model.load(APP_SERVER.default_args.model_path)
print(f"model loaded in {time.time() - start_time} seconds")


def predict_frame(frame: np.ndarray) -> Tuple[List[Dict], int]:
    global clf
    # res = clf.exec(frame)

    converted_img  = tf.image.convert_image_dtype(
        frame,
        tf.uint8
    )[tf.newaxis, ...]
    
    print("starting inference ...")
    start_time = time.time()
    res = clf(converted_img)
    print(f"inference time: {time.time() - start_time} seconds")

    # print(f'{__name__} | res: {res}')

    # .numpy() will error on num_detections, pop and add back in
    num_detections = int(res.pop('num_detections'))

    predictions = {
        key: value[0, :num_detections].numpy()
        for key, value in res.items()
    }
    predictions['num_detections'] = num_detections

    return predictions


print(f'starting {APP_SERVER.type} server')

if APP_SERVER.type == 'flask':
    from flask import Flask, request, jsonify, send_file

    app = Flask(__name__)


    @app.route('/ping', methods=['GET'])
    def ping():
        return {'message': 'pong'}


    @app.route(f'/{APP_SERVER.endpoint}', methods=['POST'])
    def predict():
        frame = pickle.loads(request.data)
        results = predict_frame(frame)

        return pickle.dumps(results)


    application = app

    if __name__ == '__main__':
        app.run(
            host=APP_SERVER.host,
            port=APP_SERVER.port,
            debug=APP_SERVER.debug
        )


elif APP_SERVER.type == 'falcon':
    import falcon

    api = falcon.API()


    class PingResource:
        def on_get(self, req, resp):
            """ Handles GET request """
            resp.media = {'message': 'pong'}


    class PredictionResource:
        def on_post(self, req, resp):
            """ Handles POST request """
            body = req.stream.read()
            frame = pickle.loads(body)
            results = predict_frame(frame)
            resp.body = pickle.dumps(results)

    
    class QuadrantsResource:
        def on_post(self, req, resp):
            """ Handles POST request """
            body = req.stream.read()
            frame, onboard, offboard = pickle.loads(body)
            
            onboard_res, offboard_res = \
                get_quadrant_results(frame, onboard, offboard)

            # print(f'{__name__} | idx: {idx} | onboard_res: {onboard_res.__dict__}')
            # print(f'{__name__} | idx: {idx} | offboard_res: {offboard_res.__dict__}')

            # updated_quadrants = compare_quadrants_from_results(
            #     onboard_res,
            #     offboard_res,
            #     original_thresholds
            # )

            resp.body = pickle.dumps([onboard_res, offboard_res])


    api.add_route('/ping', PingResource())
    api.add_route(f'/{APP_SERVER.endpoint}', PredictionResource())
    api.add_route('/quadrants', QuadrantsResource())

    application = api

print(f'started {APP_SERVER.type} server')
