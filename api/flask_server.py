import pickle

from flask import Flask, request

from .config import API_SERVER
from .utils import predict_frame

app = Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    return {'message': 'pong'}


@app.route(f'/{API_SERVER.endpoint}', methods=['POST'])
def predict():
    frame = pickle.loads(request.data)
    results = predict_frame(frame)

    return pickle.dumps(results)
