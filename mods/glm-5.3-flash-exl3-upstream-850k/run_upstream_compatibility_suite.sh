#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
uv run --with pytest --with torch --with Jinja2 --with numpy python -m pytest -q upstream/tests
python3 test_suppress_stops_multitoken.py
