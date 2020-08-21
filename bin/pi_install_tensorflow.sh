#! /usr/bin/env bash
VENV_DIR=${PWD}
VENV=${VENV_DIR}/.venv
cd /tmp
wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh
chmod +x tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh
./tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh
${VENV}/bin/pip install tensorflow-2.3.0-cp37-none-linux_armv7l.whl
