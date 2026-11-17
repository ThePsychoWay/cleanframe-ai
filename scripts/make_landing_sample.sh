#!/usr/bin/env bash
# Builds the two clips on the landing page: a generated moving picture with a semi-transparent logo, and the same
# clip after the real CleanFrame pipeline removed it (scripts/drive_landing_sample.py).
# Usage: scripts/make_landing_sample.sh <workdir>   (needs ffmpeg with drawtext and geq)
set -euo pipefail
W="${1:?workdir}"; mkdir -p "$W"
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}"
# Moving colour field: every channel is a bounded sine (0..255), so nothing wraps or clips.
ffmpeg -y -v error -f lavfi -i "color=c=black:s=540x960:r=24:d=4" -vf "format=rgb24,geq=\
r='118+72*sin(X/70+T*1.7)*cos(Y/90-T*1.1)':\
g='118+72*sin((X+Y)/110-T*1.3)':\
b='118+72*cos(X/60-Y/130+T*0.9)',format=yuv420p" -c:v libx264 -crf 14 "$W/clean.mp4"
# The overlay: a see-through white plate with text, in the top-right corner (the region used by the driver script).
ffmpeg -y -v error -i "$W/clean.mp4" -vf "drawbox=x=324:y=61:w=205:h=79:color=white@0.22:t=fill,\
drawtext=fontfile=$FONT:text='LOGO':fontcolor=white@0.85:fontsize=44:x=352:y=82" \
  -c:v libx264 -crf 17 -pix_fmt yuv420p "$W/before.mp4"
echo "wrote $W/clean.mp4 and $W/before.mp4"
