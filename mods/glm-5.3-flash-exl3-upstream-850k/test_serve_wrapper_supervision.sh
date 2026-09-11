#!/usr/bin/env bash
# Lifecycle controls: wrapper must reap both supervised process groups.
set -euo pipefail
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_ROOT="$(mktemp -d)"
active_wrapper=""
active_pgids=()

process_group_alive() {
  local pgid="$1"
  ps -eo pgid=,stat= | awk -v pgid="$pgid" '
    $1 == pgid && $2 !~ /^Z/ { alive=1 }
    END { exit(alive ? 0 : 1) }
  '
}

force_cleanup() {
  local pgid
  [ -z "$active_wrapper" ] || kill -KILL "$active_wrapper" 2>/dev/null || true
  for pgid in "${active_pgids[@]}"; do
    kill -KILL -- "-$pgid" 2>/dev/null || true
  done
  rm -rf "$TMP_ROOT"
}
trap force_cleanup EXIT

wait_file() {
  local path="$1" wrapper="$2"
  for _ in $(seq 1 200); do
    [ -s "$path" ] && return 0
    kill -0 "$wrapper" 2>/dev/null || return 1
    sleep 0.02
  done
  return 1
}

wait_absent() {
  local path="$1" wrapper="$2"
  for _ in $(seq 1 200); do
    [ ! -e "$path" ] && return 0
    kill -0 "$wrapper" 2>/dev/null || return 1
    sleep 0.02
  done
  return 1
}

assert_group_gone() {
  local name="$1" pgid="$2"
  for _ in $(seq 1 100); do
    process_group_alive "$pgid" || return 0
    sleep 0.05
  done
  printf '%s process group %s retained non-zombie members\n' "$name" "$pgid" >&2
  ps -eo pid=,ppid=,pgid=,stat=,args= | awk -v pgid="$pgid" '$3 == pgid' >&2 || true
  return 1
}

make_helpers() {
  local dir="$1"
  mkdir -p "$dir"
  cat > "$dir/fake-vllm" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[ "${1:-}" = serve ] || exit 64
printf '%s\n' "$$" > "$GLM53_TEST_SERVE_PID_FILE"
bash -c 'trap "" TERM INT; while :; do sleep 1; done' &
child=$!
printf '%s\n' "$child" > "$GLM53_TEST_SERVE_CHILD_FILE"
if [ "${GLM53_TEST_SERVE_MODE:-wait}" = crash ]; then
  # Make the crash deterministic only after both supervised groups exist.
  if [ "${GLM53_TEST_RANK:-0}" -eq 0 ]; then
    for _ in $(seq 1 200); do
      [ -s "$GLM53_TEST_GATE_CHILD_FILE" ] && break
      sleep 0.01
    done
    [ -s "$GLM53_TEST_GATE_CHILD_FILE" ] || exit 98
  fi
  exit 42
fi
trap 'exit 0' TERM INT
wait "$child"
SH
  cat > "$dir/fake-gate" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$$" > "$GLM53_TEST_GATE_PID_FILE"
bash -c 'trap "" TERM INT; while :; do sleep 1; done' &
child=$!
printf '%s\n' "$child" > "$GLM53_TEST_GATE_CHILD_FILE"
if [ "${GLM53_TEST_GATE_MODE:-wait}" = crash ]; then
  exit 42
fi
if [ "${GLM53_TEST_GATE_MODE:-wait}" = success ]; then
  kill -KILL "$child" 2>/dev/null || true
  wait "$child" 2>/dev/null || true
  exit 0
fi
trap 'exit 0' TERM INT
wait "$child"
SH
  chmod +x "$dir/fake-vllm" "$dir/fake-gate"
}

run_case() {
  local name="$1" rank="$2" action="$3" expected_rc="$4"
  local dir="$TMP_ROOT/$name"
  make_helpers "$dir"
  GLM53_VLLM_BIN="$dir/fake-vllm" \
  GLM53_POSTREADY_GATE_BIN="$dir/fake-gate" \
  GLM53_GATE_PID_FILE="$dir/published-gate.pid" \
  GLM53_TEST_SERVE_PID_FILE="$dir/serve.pid" \
  GLM53_TEST_SERVE_CHILD_FILE="$dir/serve-child.pid" \
  GLM53_TEST_GATE_PID_FILE="$dir/gate.pid" \
  GLM53_TEST_GATE_CHILD_FILE="$dir/gate-child.pid" \
  GLM53_TEST_RANK="$rank" \
  GLM53_TEST_SERVE_MODE="${CASE_SERVE_MODE:-wait}" \
  GLM53_TEST_GATE_MODE="${CASE_GATE_MODE:-wait}" \
  GLM53_TEST_FORCE_CLEANUP_FAILURE="${CASE_CLEANUP_FAILURE:-0}" \
    setsid bash "$MOD_DIR/serve_wrapper.sh" model --node-rank "$rank" \
    >"$dir/wrapper.out" 2>"$dir/wrapper.err" &
  active_wrapper=$!

  wait_file "$dir/serve.pid" "$active_wrapper" || { echo "$name: serve leader missing" >&2; return 1; }
  wait_file "$dir/serve-child.pid" "$active_wrapper" || { echo "$name: serve descendant missing" >&2; return 1; }
  local serve_pgid gate_pgid=""
  serve_pgid="$(<"$dir/serve.pid")"
  active_pgids=("$serve_pgid")
  if [ "$rank" -eq 0 ]; then
    wait_file "$dir/gate.pid" "$active_wrapper" || { echo "$name: gate leader missing" >&2; return 1; }
    wait_file "$dir/gate-child.pid" "$active_wrapper" || { echo "$name: gate descendant missing" >&2; return 1; }
    gate_pgid="$(<"$dir/gate.pid")"
    active_pgids+=("$gate_pgid")
  else
    [ ! -e "$dir/gate.pid" ] || { echo "$name: worker unexpectedly launched gate" >&2; return 1; }
  fi

  case "$action" in
    kill-gate) kill -KILL "$gate_pgid" ;;
    term-wrapper) kill -TERM "$active_wrapper" ;;
    gate-success)
      wait_absent "$dir/published-gate.pid" "$active_wrapper" || {
        echo "$name: completed gate PGID remained published" >&2
        return 1
      }
      kill -TERM "$active_wrapper"
      ;;
    none) ;;
    *) echo "unknown action $action" >&2; return 1 ;;
  esac

  set +e
  wait "$active_wrapper"
  local rc=$?
  set -e
  active_wrapper=""
  [ "$rc" -eq "$expected_rc" ] || {
    printf '%s: expected wrapper rc %s, got %s\n' "$name" "$expected_rc" "$rc" >&2
    printf '%s\n' '--- wrapper stderr ---' >&2
    while IFS= read -r line; do printf '%s\n' "$line" >&2; done < "$dir/wrapper.err"
    ps -eo pid=,ppid=,pgid=,stat=,args= | awk -v a="$serve_pgid" -v b="$gate_pgid" '$3 == a || $3 == b' >&2 || true
    return 1
  }
  assert_group_gone "$name serve" "$serve_pgid"
  [ -z "$gate_pgid" ] || assert_group_gone "$name gate" "$gate_pgid"
  active_pgids=()
  printf 'serve-wrapper lifecycle control OK: %s rc=%s\n' "$name" "$rc"
}

CASE_SERVE_MODE=wait CASE_GATE_MODE=wait CASE_CLEANUP_FAILURE=0 run_case killed-gate 0 kill-gate 137
CASE_SERVE_MODE=wait CASE_GATE_MODE=crash CASE_CLEANUP_FAILURE=0 run_case crashed-gate 0 none 42
CASE_SERVE_MODE=wait CASE_GATE_MODE=wait CASE_CLEANUP_FAILURE=0 run_case wrapper-term 0 term-wrapper 143
CASE_SERVE_MODE=wait CASE_GATE_MODE=success CASE_CLEANUP_FAILURE=0 run_case gate-success-clears-pgid 0 gate-success 143
CASE_SERVE_MODE=crash CASE_GATE_MODE=wait CASE_CLEANUP_FAILURE=0 run_case serving-leader-crash 0 none 42
CASE_SERVE_MODE=wait CASE_GATE_MODE=wait CASE_CLEANUP_FAILURE=1 run_case cleanup-helper-failure 0 term-wrapper 127
CASE_SERVE_MODE=crash CASE_GATE_MODE=wait CASE_CLEANUP_FAILURE=0 run_case worker-serving-leader-crash 1 none 42
CASE_SERVE_MODE=wait CASE_GATE_MODE=wait CASE_CLEANUP_FAILURE=0 run_case worker-wrapper-term 1 term-wrapper 143
CASE_SERVE_MODE=wait CASE_GATE_MODE=wait CASE_CLEANUP_FAILURE=1 run_case worker-cleanup-helper-failure 1 term-wrapper 127
python3 "$MOD_DIR/test_serve_wrapper_sigint.py"
