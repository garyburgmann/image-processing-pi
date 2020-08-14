#! /usr/bin/env bash
VENV_DIR=/tmp
VENV=${VENV_DIR}/.venv

python3 -m venv ${VENV}
${VENV}/pip install --upgrade pip setuptools
${VENV}/bin/pip install -r pi.requirements.txt
