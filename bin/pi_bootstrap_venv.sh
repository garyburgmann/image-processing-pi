#! /usr/bin/env bash
VENV_DIR=${PWD}
VENV=${VENV_DIR}/venv

python3.7 -m venv ${VENV} --clear
${VENV}/bin/pip install --upgrade pip setuptools wheel
${VENV}/bin/pip install -r pi.requirements.txt
