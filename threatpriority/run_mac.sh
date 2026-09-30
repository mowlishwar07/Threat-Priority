#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  printf '%s\n' 'Run bash setup_mac.sh first.'
  exit 1
fi
.venv/bin/python app.py --port 5050
