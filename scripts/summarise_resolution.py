"""Turns the JSON files written by scripts/e2e_resolution.py into the markdown tables used in docs/RESOLUTION.md.

    python scripts/summarise_resolution.py /tmp/bench/res/out
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ["720p", "1080p", "1440p", "2160p"]


def main() -> None:
    folder = Path(sys.argv[1])
    reports = {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")}
    print("| Clip | Source | Output (ffprobe) | Frames | Audio | Render | fps | Peak server RAM | Mean CPU | PSNR in logo / outside | Checks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for label in ORDER:
        r = reports.get(label)
        if not r or "output" not in r:
            continue
        s, o, q, srv = r["source"], r["output"], r.get("quality", {}), r.get("server", {})
        seconds = r["render_seconds"]
        print(
            f"| {label} | {s['width']}×{s['height']} {s['codec']} {s['bitrate'] / 1e6:.0f} Mb/s | {o['width']}×{o['height']} {o['codec']} {o['bitrate'] / 1e6:.0f} Mb/s "
            f"| {o['frames']}/{s['frames']} | {o['audio']['codec'] if o['audio'] else '-'} copied | {seconds:.0f} s | {o['frames'] / seconds:.2f} "
            f"| {srv.get('peak_rss_mb', 0) / 1024:.1f} GB | {srv.get('mean_cpu_percent', 0)} % "
            f"| {q.get('psnr_inside_plate_db', '-')} / {q.get('psnr_outside_plate_db', '-')} dB | {'all passed' if not r['failures'] else str(len(r['failures'])) + ' failed'} |"
        )
    print()
    print("| Export from | To | Size (ffprobe) | Frames | Time | PSNR in logo / outside vs scaled truth |")
    print("|---|---|---|---|---|---|")
    for label in ORDER:
        r = reports.get(label)
        for e in (r or {}).get("exports", []):
            f, q = e["file"], e.get("quality", {})
            print(f"| {label} | {e['height']}p | {f['width']}×{f['height']} | {f['frames']} | {e['seconds']:.0f} s | {q.get('psnr_inside_plate_db', '-')} / {q.get('psnr_outside_plate_db', '-')} dB |")


if __name__ == "__main__":
    main()
