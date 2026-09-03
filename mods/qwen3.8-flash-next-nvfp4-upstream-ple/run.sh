#!/usr/bin/env bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${VLLM_QWEN38_PLE_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py}"
ORIGINAL_SHA256="a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7"
PATCHED_SHA256="ecb7b554f565ca4fb9fc89f3d7fa583bf6bed89b624398e38c9c34872d5746d5"

cd "$MOD_DIR"
sha256sum -c SHA256SUMS

if [ ! -f "$DEST" ]; then
  echo "FATAL: pinned image PLE module not found: $DEST" >&2
  exit 1
fi

current_sha=$(sha256sum "$DEST" | cut -d' ' -f1)
case "$current_sha" in
  "$ORIGINAL_SHA256")
    install -m 0644 "$MOD_DIR/ple_layer_patched.py" "$DEST"
    ;;
  "$PATCHED_SHA256")
    echo "[OK] PLE override already installed"
    ;;
  *)
    echo "FATAL: unexpected base PLE module sha256: $current_sha" >&2
    exit 1
    ;;
esac

installed_sha=$(sha256sum "$DEST" | cut -d' ' -f1)
if [ "$installed_sha" != "$PATCHED_SHA256" ]; then
  echo "FATAL: installed PLE module sha256 mismatch: $installed_sha" >&2
  exit 1
fi

PYTHONPYCACHEPREFIX=/tmp/qwen38-pycache python3 -m py_compile "$DEST" "$MOD_DIR/postready_gate.py"
python3 - "$DEST" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
tree = ast.parse(path.read_text())
names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
if names.count("_get_ple_embedding_quant_method") != 2:
    raise SystemExit("FATAL: PLE resolver wrapper count is not exactly two")
if "_ORIG_PLE_QUANT_RESOLVER" not in path.read_text():
    raise SystemExit("FATAL: original PLE resolver alias missing")
print("[OK] PLE FP8 resolver override installed and statically verified")
PY
