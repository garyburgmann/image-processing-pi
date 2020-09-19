#!/usr/bin/env bash
celery -A app.tasks worker -l info
