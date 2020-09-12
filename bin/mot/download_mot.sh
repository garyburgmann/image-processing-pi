#!/usr/bin/env bash

# exec with args <MOTxx> <OUTDIR>

MOT=$1
DATA_FILE="$MOT.zip"
DATA_URL="https://motchallenge.net/data/$DATA_FILE"

if [ $# -eq 1 ]; then
  DATA_DIR="/tmp"
else
  DATA_DIR="$2"
fi

MOT_DIR="$DATA_DIR/$MOT"

mkdir -p $MOT_DIR

cd $MOT_DIR

if [ -f "$DATA_FILE" ]; then
  echo "$DATA_FILE already exists "
else 
  echo "$DATA_FILE does not exist in $MOT_DIR"
  curl -O $DATA_URL
fi

unzip -o $DATA_FILE
rm $DATA_FILE

echo -e "$DATA_FILE extracted to $MOT_DIR"
