import pickle

import falcon

from .config import API_SERVER
from .utils import predict_frame

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


api.add_route('/ping', PingResource())
api.add_route(f'/{API_SERVER.endpoint}', PredictionResource())
