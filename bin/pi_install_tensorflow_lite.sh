#! /usr/bin/env bash
VENV_DIR=/tmp
VENV=${VENV_DIR}/.venv
cd /tmp
wget https://github.com/PINTO0309/TensorflowLite-bin/raw/master/2.3.0/download_tflite_runtime-2.3.0-py3-none-linux_armv7l.whl.sh
chmod +x download_tflite_runtime-2.3.0-py3-none-linux_armv7l.whl.sh
./download_tflite_runtime-2.3.0-py3-none-linux_armv7l.whl.sh
${VENV}/bin/pip install tflite_runtime-2.3.0-py3-none-linux_armv7l.whl
