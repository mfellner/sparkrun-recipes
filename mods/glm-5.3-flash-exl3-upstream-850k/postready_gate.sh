#!/usr/bin/env bash
# Fail-closed post-ready gate launched beside vLLM before shell exec.
set -u
BASE="${1:-http://127.0.0.1:8000}"
MODEL="${2:-GLM-5.3-Flash-EXL3}"
SERVE_PID="${SPARKRUN_SERVE_PID:?SPARKRUN_SERVE_PID is required}"
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
RC=/tmp/glm53-postready.rc
OK=/tmp/glm53-postready.ok
rm -f "$RC" "$OK"
SERVE_PGID="${SPARKRUN_SERVE_PGID:-$(ps -o pgid= -p "$SERVE_PID" 2>/dev/null | tr -d '[:space:]')}"
case "$SERVE_PGID" in
  ''|*[!0-9]*) printf 'postready gate FATAL: invalid serve PGID %q\n' "$SERVE_PGID" >&2; exit 127 ;;
esac
GATE_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]')"
[ "$SERVE_PGID" != "$GATE_PGID" ] || {
  printf 'postready gate FATAL: serve group is not separate from gate\n' >&2
  exit 127
}
process_group_alive() {
  ps -eo pgid=,stat= | awk -v pgid="$SERVE_PGID" '
    $1 == pgid && $2 !~ /^Z/ { alive=1 }
    END { exit(alive ? 0 : 1) }
  '
}
fail() {
  rc=${1:-1}; shift || true
  printf 'postready gate FAILED: %s\n' "$*" >&2
  printf '%s\n' "$rc" > "$RC"
  kill -TERM -- "-$SERVE_PGID" 2>/dev/null || true
  for _ in $(seq 1 10); do
    process_group_alive || exit "$rc"
    sleep 1
  done
  kill -KILL -- "-$SERVE_PGID" 2>/dev/null || true
  for _ in $(seq 1 50); do
    process_group_alive || exit "$rc"
    sleep 0.1
  done
  printf 'postready gate FAILED to terminate serve process group %s\n' "$SERVE_PGID" >&2
  printf '127\n' > "$RC"
  exit 127
}
[ "${POSTREADY_GATE_TEST_FAIL:-0}" = 1 ] && fail 92 "negative-control failure"
ready=0
for _ in $(seq 1 720); do
  if curl -fsS --max-time 5 "$BASE/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$SERVE_PID" 2>/dev/null || fail 90 "serve process exited before readiness"
  sleep 5
done
[ "$ready" = 1 ] || fail 91 "API not ready within 3600 seconds"
GLM53_WARMUP_MAX_CONCURRENCY=4 \
GLM53_WARMUP_REQ_TIMEOUT=240 \
GLM53_WARMUP_DFLASH_K=7 \
GLM53_WARMUP_TRITON_CACHE_DIR="${GLM53_WARMUP_TRITON_CACHE_DIR:-${TRITON_CACHE_DIR:-}}" \
  bash "$MOD_DIR/upstream/scripts/boot-shape-warmup.sh" "$BASE" "$MODEL" || fail 92 "boot shape warmup failed"
python3 - "$BASE" "$MODEL" <<'PY' || fail 93 "semantic completion gate failed"
import json,sys,urllib.request
base,model=sys.argv[1:3]
body=json.dumps({"model":model,"messages":[{"role":"user","content":"Reply with exactly GLM53_EXL3_OK and nothing else."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":False}}).encode()
req=urllib.request.Request(base+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
data=json.load(urllib.request.urlopen(req,timeout=300))
content=(data["choices"][0]["message"].get("content") or "").strip()
assert content=="GLM53_EXL3_OK",repr(content)
PY
printf '0\n' > "$RC"
date --iso-8601=seconds > "$OK"
printf 'postready gate OK\n'
