from types import SimpleNamespace
import os

API_SERVER_OPTIONS = ['flask', 'falcon']
API_SERVER = SimpleNamespace(
    type=os.getenv('SERVER_TYPE', API_SERVER_OPTIONS[1]).lower(),
    port=8000,
    host='0.0.0.0',
    debug=False,
    endpoint=os.getenv('ENDPOINT', 'predict'),
    model_path=os.getenv(
        'SAVED_MODEL_PATH',
        'srv/models/efficientdet_d3_coco17_tpu-32/saved_model'
    )
)
assert API_SERVER.type in API_SERVER_OPTIONS
