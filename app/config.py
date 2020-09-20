from types import SimpleNamespace

FLASK_SERVER = 'http://mb.local'  # 'http://localhost'  # 'http://192.168.42.174'
FLASK_PORT = 8000
FLASK_HOST = '0.0.0.0'
FLASK_DEBUG = False
FLASK_ENDPOINT = ''
DEFAULT_ARGS = SimpleNamespace(
    model_path='/tmp/detect.tflite',
    labels_path='./labels/coco.txt',
    target='person',
    threshold=0.51,
    num_threads=4,
    camera=0
)
CLASSIFIER_INPUT_SHAPE = SimpleNamespace(
    width=300,
    height=300
)    

CELERY_CONFIG = SimpleNamespace(
    broker_url='redis://localhost:6379/1',
    task_serializer='pickle',
    accept_content=['pickle']
)
