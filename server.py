import pickle
from typing import List, Tuple, Dict

import numpy as np

from app.utils import (
    parse_args,
    get_classifier,
)
from app.config import (
    APP_SERVER,
    APP_SERVER_OPTIONS,
)

# mimic args, as argparse will clash with gunicorn
server_args = APP_SERVER.default_args
server_args.lite = False

# bootstrap here to reduce latency on each request
clf = get_classifier(server_args)


def predict_frame(frame: np.ndarray) -> Tuple[List[Dict], int]:
    global clf
    res = clf.exec(frame)
    print(f'{__name__} | res: {res}')
    return res


assert APP_SERVER.type in APP_SERVER_OPTIONS, '\n\ninvalid APP_SERVER.type\n\n'

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


    class ClassifierResource:
        def on_post(self, req, resp):
            """ Handles POST request """
            body = req.stream.read()
            frame = pickle.loads(body)
            results = predict_frame(frame)
            resp.body = pickle.dumps(results)


    api.add_route('/ping', PingResource())
    api.add_route(f'/{APP_SERVER.endpoint}', ClassifierResource())

    application = api

print(f'started {APP_SERVER.type} server')
