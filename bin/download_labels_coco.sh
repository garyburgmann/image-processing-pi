#!/usr/bin/env bash

#### in repo ow anyway, here for brevity #####
#### assumes lables/ dir in project root #####
#### args 1 (optional) - LABELS_DIR ####

LABELS_FILE=coco_labels.txt
LABELS_URI="https://dl.google.com/coral/canned_models/$LABELS_FILE"

LABELS_DIR="./labels"
if [ $# -gt 0 ]; then
  LABELS_DIR="$1"
fi

mkdir -p $LABELS_DIR
cd $LABELS_DIR

# Get a labels file with corrected indices
curl -O $LABELS_URI

echo -e "Labels downloaded to $LABELS_URI"
