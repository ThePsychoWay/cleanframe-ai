#!/bin/bash
# Runs one end-to-end script against the running stack:  scripts/e2e/run.sh e2e_full.js
# Needs Playwright for Node (npm i -g playwright && npx playwright install chromium) and the stack from start.sh.
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")"
NODE_PATH="$(npm root -g)" exec node "${1:?script name}"
