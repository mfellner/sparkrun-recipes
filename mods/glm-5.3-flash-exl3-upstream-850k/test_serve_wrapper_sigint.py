#!/usr/bin/env python3
"""Deterministic rank-0/rank-1 SIGINT process-group cleanup controls."""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path

MOD_DIR = Path(__file__).resolve().parent
WRAPPER = MOD_DIR / "serve_wrapper.sh"


def write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def wait_file(path: Path, process: subprocess.Popen, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size:
            return int(path.read_text())
        if process.poll() is not None:
            raise AssertionError(f"wrapper exited before {path.name}: rc={process.returncode}")
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path.name}")


def group_has_live_members(pgid: int) -> bool:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = proc.joinpath("stat").read_text()
            fields = stat[stat.rfind(") ") + 2 :].split()
            state, member_pgid = fields[0], int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        if member_pgid == pgid and state != "Z":
            return True
    return False


def wait_group_gone(pgid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not group_has_live_members(pgid):
            return
        time.sleep(0.02)
    raise AssertionError(f"process group {pgid} retained a non-zombie member")


def run_rank(rank: int) -> None:
    with tempfile.TemporaryDirectory(prefix=f"glm53-sigint-rank{rank}-") as tmp:
        root = Path(tmp)
        fake_vllm = root / "fake-vllm"
        fake_gate = root / "fake-gate"
        write_executable(
            fake_vllm,
            """#!/usr/bin/env bash
set -euo pipefail
[ "${1:-}" = serve ] || exit 64
printf '%s\n' "$$" > "$GLM53_TEST_SERVE_PID_FILE"
bash -c 'trap "" TERM INT; while :; do sleep 1; done' &
child=$!
printf '%s\n' "$child" > "$GLM53_TEST_SERVE_CHILD_FILE"
trap 'exit 0' TERM INT
wait "$child"
""",
        )
        write_executable(
            fake_gate,
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$$" > "$GLM53_TEST_GATE_PID_FILE"
bash -c 'trap "" TERM INT; while :; do sleep 1; done' &
child=$!
printf '%s\n' "$child" > "$GLM53_TEST_GATE_CHILD_FILE"
trap 'exit 0' TERM INT
wait "$child"
""",
        )
        env = os.environ.copy()
        env.update(
            {
                "GLM53_VLLM_BIN": str(fake_vllm),
                "GLM53_POSTREADY_GATE_BIN": str(fake_gate),
                "GLM53_GATE_PID_FILE": str(root / "published-gate.pid"),
                "GLM53_TEST_SERVE_PID_FILE": str(root / "serve.pid"),
                "GLM53_TEST_SERVE_CHILD_FILE": str(root / "serve-child.pid"),
                "GLM53_TEST_GATE_PID_FILE": str(root / "gate.pid"),
                "GLM53_TEST_GATE_CHILD_FILE": str(root / "gate-child.pid"),
                "GLM53_TEST_RANK": str(rank),
                "GLM53_TEST_SERVE_MODE": "wait",
                "GLM53_TEST_GATE_MODE": "wait",
                "GLM53_TEST_FORCE_CLEANUP_FAILURE": "0",
            }
        )
        wrapper_command = shlex.join(
            ["bash", str(WRAPPER), "model", "--node-rank", str(rank)]
        )
        process = subprocess.Popen(
            ["script", "-qefc", wrapper_command, "/dev/null"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        pgids: list[int] = []
        try:
            pgids.append(wait_file(root / "serve.pid", process))
            wait_file(root / "serve-child.pid", process)
            if rank == 0:
                pgids.append(wait_file(root / "gate.pid", process))
                wait_file(root / "gate-child.pid", process)
            elif (root / "gate.pid").exists():
                raise AssertionError("worker unexpectedly launched the post-readiness gate")
            assert process.stdin is not None
            process.stdin.write(b"\x03")
            process.stdin.flush()
            stdout, stderr = process.communicate(timeout=20)
            if process.returncode != 130:
                raise AssertionError(
                    f"rank {rank}: expected rc=130, got {process.returncode}; "
                    f"stdout={stdout.decode(errors='replace')!r} "
                    f"stderr={stderr.decode(errors='replace')!r}"
                )
            for pgid in pgids:
                wait_group_gone(pgid)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for pgid in pgids:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        print(f"SIGINT cleanup control OK: rank={rank} rc=130")


def main() -> int:
    for rank in (0, 1):
        run_rank(rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
