#!/bin/bash

# Convert OGG files
for video in *.ogg; do
    if [ -f "$video" ]; then
        output="${video%.ogg}.mp4"
        ffmpeg -i "$video" -c:v libx264 -c:a aac "$output"
        echo "Converted $video to $output"
    fi
done

# Convert OGV files
for video in *.ogv; do
    if [ -f "$video" ]; then
        output="${video%.ogg}.mp4"
        ffmpeg -i "$video" -c:v libx264 -c:a aac "$output"
        echo "Converted $video to $output"
    fi
done

# Convert WebM files
for video in *.webm; do
    if [ -f "$video" ]; then
        output="${video%.webm}.mp4"
        ffmpeg -i "$video" -c:v libx264 -c:a aac "$output"
        echo "Converted $video to $output"
    fi
done

echo "All conversions completed."