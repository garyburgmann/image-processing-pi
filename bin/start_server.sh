#!/usr/bin/env bash
# large timeout for model loading
gunicorn -w 4 -b 0.0.0.0:8000 server:application --reload --timeout 120
