from types import SimpleNamespace

FLASK_SERVER = 'http://localhost'  # 'http://192.168.42.174'
FLASK_PORT = 5000
FLASK_HOST = '0.0.0.0'
FLASK_DEBUG = False
FLASK_ENDPOINT = ''
DEFAULT_ARGS = SimpleNamespace(
    model_path='/tmp/detect.tflite',
    labels_path='/tmp/labels_corrected.txt',
    target='person',
    threshold=0.51,
    num_threads=4,
    camera=0
)
