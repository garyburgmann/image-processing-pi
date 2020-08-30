import pickle
from typing import List, Tuple, Dict

import numpy as np
from flask import Flask, request, jsonify, send_file

from app.utils import (
    parse_args,
    get_classifier,
    run_classifier
)
from app.config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    FLASK_ENDPOINT
)

app = Flask(__name__)

clf = get_classifier(parse_args())


def classify_frame(frame: np.ndarray) -> Tuple[List[Dict], int]:
    results, num_boxes = run_classifier(img=frame, clf=clf)
    print(f'{__name__}: {results}, {num_boxes}')
    return results, num_boxes


@app.route(f'/{FLASK_ENDPOINT}', methods=['POST'])
def classify():
    frame = pickle.loads(request.data)
    results, num_boxes = classify_frame(frame)

    return pickle.dumps([results, num_boxes])


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
