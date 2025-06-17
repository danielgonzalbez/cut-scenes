#!/bin/bash

KAGGLE_USERNAME=""
KAGGLE_KEY=""
DATASET="perrotedan/concerts-scenes"
LOCAL_PATH="../data"

mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

mkdir -p "$LOCAL_PATH"
kaggle datasets download -d "$DATASET" -p "$LOCAL_PATH"

unzip "$LOCAL_PATH/$(basename $DATASET).zip" -d "$LOCAL_PATH"


