#! /usr/bin/env bash
VENV_DIR=/tmp
VENV=${VENV_DIR}/.venv

python3 -m venv ${VENV}
${VENV}/bin/pip install -r pi.requirements.txt
