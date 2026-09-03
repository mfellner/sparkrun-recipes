#!/usr/bin/env bash
set -euo pipefail

HEAD="${HEAD:-192.168.178.48}"
CONTAINER="${CONTAINER:-sparkrun_763b8e94b6d2bf09_27aa44aef83a_node_0}"
ATTEMPTS="${ATTEMPTS:-240}"
INTERVAL="${INTERVAL:-15}"

[[ "$HEAD" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "FATAL: invalid HEAD" >&2; exit 64; }
[[ "$CONTAINER" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "FATAL: invalid CONTAINER" >&2; exit 64; }

for ((i=1; i<=ATTEMPTS; i++)); do
  now=$(date --iso-8601=seconds)
  running=$(ssh -o BatchMode=yes "$HEAD" "docker inspect -f '{{.State.Running}}' '$CONTAINER' 2>/dev/null || printf missing")
  printf '%s attempt=%d running=%s\n' "$now" "$i" "$running"
  if [ "$running" != "true" ]; then
    ssh -o BatchMode=yes "$HEAD" "docker logs --since 30m '$CONTAINER' 2>&1" || true
    exit 1
  fi
  if ssh -o BatchMode=yes "$HEAD" "docker exec '$CONTAINER' test -f /tmp/qwen38-postready.json"; then
    set +e
    receipt=$(ssh -o BatchMode=yes "$HEAD" "docker exec '$CONTAINER' python3 -c 'import json; from pathlib import Path; p=Path(\"/tmp/qwen38-postready.json\"); d=json.loads(p.read_text()); print(json.dumps(d, indent=2, sort_keys=True)); raise SystemExit(0 if d.get(\"ok\") is True else 1)'" 2>&1)
    receipt_rc=$?
    set -e
    printf '%s\n' "$receipt"
    if [ "$receipt_rc" -eq 0 ]; then
      exit 0
    fi
    echo "FATAL: post-readiness receipt did not contain ok=true" >&2
    exit 3
  fi
  if [ "$i" -ge 3 ] && ! ssh -o BatchMode=yes "$HEAD" "docker exec '$CONTAINER' pgrep -f 'vllm.*serve' >/dev/null"; then
    ssh -o BatchMode=yes "$HEAD" "docker exec '$CONTAINER' python3 -c 'from pathlib import Path; p=Path(\"/tmp/sparkrun_serve.log\"); print(\"\\n\".join(p.read_text(errors=\"replace\").splitlines()[-160:]) if p.exists() else \"serve log missing\")'" || true
    echo "FATAL: serve process exited before post-readiness receipt" >&2
    exit 2
  fi
  sleep "$INTERVAL"
done

echo "FATAL: timed out waiting for in-container post-readiness receipt" >&2
exit 1
