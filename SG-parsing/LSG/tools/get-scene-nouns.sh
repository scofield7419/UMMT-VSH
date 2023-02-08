#! /bin/sh

set -x

curl https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt | \
    python3 process-scene-nouns.py > \
    ../sng_parser/_data/scene-nouns.txt
