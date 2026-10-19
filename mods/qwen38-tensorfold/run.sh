#!/usr/bin/env bash
# No engine overlay: only verify the SparkRun-to-TensorFold alias adapter.
set -euo pipefail
cd "$(dirname "$0")"
sha256sum --check SHA256SUMS
python -m py_compile serve.py
python stage_drafter.py
