#!/usr/bin/env bash
celery -A app.tasks worker -l info --pool=eventlet --concurrency=100
