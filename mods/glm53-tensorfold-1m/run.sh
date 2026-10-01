#!/usr/bin/env bash
# SparkRun copies this mod onto every rank and runs this before serving.
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS
bash -n serve.sh
python3 -m py_compile fabric.py
[[ -r "$TF_DRAFTER_SNAPSHOT/config.json" ]] || { echo 'missing pinned DFlash2 snapshot' >&2; exit 1; }
echo 'TensorFold SparkRun adapter integrity passed'
