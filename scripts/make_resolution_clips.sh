#!/usr/bin/env bash
# Builds test clips at 720p, 1080p, 1440p and 4K from one real source clip (landscape crop, lanczos upscale, grain),
# each as a "clean" version (the ground truth) and a "marked" version with an opaque logo plate drawn at the same
# fractional position. Audio is copied from the source. NOTE: the footage is upscaled, so it carries no genuine
# fine detail beyond the source resolution; it tests the pipeline's handling of size, not 4K sensor detail.
# Usage: scripts/make_resolution_clips.sh <source.mp4> <outdir> [seconds=2]
set -euo pipefail
SRC="${1:?source clip}"; OUT="${2:?outdir}"; SECS="${3:-2}"
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}"
mkdir -p "$OUT"
# name:width:height
for spec in 720p:1280:720 1080p:1920:1080 1440p:2560:1440 2160p:3840:2160; do
  IFS=: read -r name w h <<<"$spec"
  base="crop=iw:iw*9/16:0:ih*0.30,scale=${w}:${h}:flags=lanczos,noise=alls=6:allf=t,format=yuv420p"
  ffmpeg -y -v error -t "$SECS" -i "$SRC" -vf "$base" -r 30 -c:v libx264 -preset veryfast -crf 16 -c:a copy "$OUT/clean_${name}.mp4"
  # logo plate: x 0.70-0.94, y 0.06-0.19 of the frame
  ffmpeg -y -v error -i "$OUT/clean_${name}.mp4" -vf "drawbox=x=iw*0.70:y=ih*0.06:w=iw*0.24:h=ih*0.13:color=0xC0392B@1:t=fill,\
drawtext=fontfile=$FONT:text='LOGO':fontcolor=white:fontsize=h*0.07:x=w*0.745:y=h*0.085" \
    -c:v libx264 -preset veryfast -crf 16 -c:a copy "$OUT/marked_${name}.mp4"
done
ls -la "$OUT"
