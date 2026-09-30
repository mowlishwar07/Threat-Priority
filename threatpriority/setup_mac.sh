#!/bin/bash
set -e
cd "$(dirname "$0")"
python3.12 --version
if [ ! -x .venv/bin/python ]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
printf '%s\n' 'Setup complete. Run bash run_mac.sh and open http://127.0.0.1:5050'
