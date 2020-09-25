#!/usr/bin/env bash
docker kill redis
docker run --rm --name redis -p 6379:6379 -d redis:6
docker logs -f redis
