#!/bin/bash
# Double-click (macOS) or run (Linux): starts CleanFrame AI and opens it in your browser.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 run_local.py "$@"
elif command -v python >/dev/null 2>&1; then
  python run_local.py "$@"
else
  echo "Python 3.11 or newer was not found. Install it from https://www.python.org/downloads/ and run this again."
fi
echo
read -r -p "Press Enter to close this window."
