#!/bin/bash
# The clip the load test uploads: the semi-transparent-logo test footage (60 frames) enlarged to 960x480, so a render
# costs more than the tiny unit-test clips do.
source "$(dirname "$0")/../e2e/env.sh"
ffmpeg -v error -y -i "$CF_E2E_DIR/media/overlay.mp4" -vf scale=960:480:flags=lanczos -c:v libx264 -crf 14 -pix_fmt yuv420p -c:a copy "$CF_E2E_DIR/media/load.mp4"
ls -l "$CF_E2E_DIR/media/load.mp4"
