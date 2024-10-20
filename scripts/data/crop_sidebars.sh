#!/bin/bash

for video in *.mp4; do
    # Detect crop
    crop=$(ffmpeg -i "$video" -vf cropdetect -f null - 2>&1 | awk '/crop/ { print $NF }' | tail -1)
    
    # Apply crop
    ffmpeg -i "$video" -vf "$crop" "cropped_$video"
    
    echo "Processed $video with crop $crop"
done