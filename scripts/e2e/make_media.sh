#!/bin/bash
# Makes the two small files the end-to-end tests upload: a textured 320x568 clip with sound, and a file that is not a video.
source "$(dirname "$0")/env.sh"
M="$CF_E2E_DIR/media"
ffmpeg -v error -y -f lavfi -i "testsrc2=size=320x568:rate=30:duration=2" -f lavfi -i "sine=frequency=440:duration=2" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a aac -shortest "$M/small.mp4"
printf 'this is not a video, it is text with a video name' > "$M/garbage.mp4"
ls -l "$M"
