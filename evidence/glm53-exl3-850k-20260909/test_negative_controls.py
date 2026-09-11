#!/usr/bin/env python3
"""Adversarial controls for the evidence verifier."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import runpy
import shlex
import shutil
import subprocess
import tempfile
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT / "verify.py"
REPO = ROOT.parents[1]
RECIPE = REPO / "recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml"
MOD = REPO / "mods/glm-5.3-flash-exl3-upstream-850k"


def write_manifest(root: Path) -> None:
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts)
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}" for path in paths]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def mutate_json(path: Path, mutator) -> None:
    data = json.loads(path.read_text())
    mutator(data)
    path.write_text(json.dumps(data, indent=2) + "\n")


def verifier_failures(
    root: Path,
    *,
    verifier: Path = VERIFY,
    mod: Path = MOD,
    recipe: Path = RECIPE,
) -> list[str]:
    result = subprocess.run(
        [
            "python3", str(verifier), "--root", str(root),
            "--recipe", str(recipe), "--mod-dir", str(mod),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return []
    try:
        return json.loads(result.stdout)["failures"]
    except Exception as exc:
        raise AssertionError(f"verifier did not return JSON: {result.stdout!r}") from exc


def rejected(
    name: str,
    mutator,
    expected_failure: str,
    *,
    refresh_manifest: bool = True,
    prepare=None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="glm53-evidence-negative-") as tmp:
        copied = Path(tmp) / "evidence"
        shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns("__pycache__"))
        if prepare is not None:
            prepare(copied)
        write_manifest(copied)
        baseline = verifier_failures(copied)
        mutator(copied)
        if refresh_manifest:
            write_manifest(copied)
        failures = verifier_failures(copied)
        if failures.count(expected_failure) <= baseline.count(expected_failure):
            raise AssertionError(
                f"negative control did not add the expected failure: {name}; "
                f"wanted {expected_failure!r}; baseline={baseline!r}; mutated={failures!r}"
            )
        print(f"rejected: {name}")


def refresh_artifact_binding(root: Path, rel: str) -> None:
    verifier = root / "verify.py"
    expected = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    text, count = re.subn(
        rf'("{re.escape(rel)}":\s*")[^"]+("[,])',
        rf"\g<1>{expected}\2",
        verifier.read_text(),
        count=1,
    )
    if count != 1:
        raise AssertionError(f"artifact binding row missing: {rel}")
    verifier.write_text(text)


def rejected_with_refreshed_binding(
    name: str,
    mutator,
    expected_failure: str,
    *,
    rel: str,
    prepare=None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="glm53-honest-binding-negative-") as tmp:
        copied = Path(tmp) / "evidence"
        shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns("__pycache__"))
        if prepare is not None:
            prepare(copied)
        refresh_artifact_binding(copied, rel)
        write_manifest(copied)
        baseline = verifier_failures(copied, verifier=copied / "verify.py")
        mutator(copied)
        refresh_artifact_binding(copied, rel)
        write_manifest(copied)
        failures = verifier_failures(copied, verifier=copied / "verify.py")
        if failures.count(expected_failure) <= baseline.count(expected_failure):
            raise AssertionError(
                f"honest refreshed-binding control did not add expected failure: {name}; "
                f"wanted {expected_failure!r}; baseline={baseline!r}; mutated={failures!r}"
            )
        print(f"rejected: {name}")


def mutate_worker_image(data: dict) -> None:
    command = data["hosts"]["192.168.178.46"]["docker_inspect"]
    inspect = json.loads(command["stdout"])
    inspect[0]["Config"]["Image"] = "forged:latest"
    command["stdout"] = json.dumps(inspect) + "\n"


def forge_telemetry(root: Path) -> None:
    path = root / "load-telemetry.log"
    text = path.read_text()
    marker = "sample=after"
    prefix, after = text.split(marker, 1)
    after, count = re.subn(r"(rocep1s0f1 xmit=)\d+", r"\g<1>0", after, count=1)
    if count != 1:
        raise AssertionError("telemetry mutation anchor missing")
    path.write_text(prefix + marker + after)


def request_sha(body: dict) -> str:
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


def substitute_direct_request(data: dict) -> None:
    row = data["checks"]["direct_exact"]
    row["request"]["messages"][0]["content"] = "Reply with exactly FORGED and nothing else."
    row["request_sha256"] = request_sha(row["request"])


def replace_fixture_everywhere(root: Path) -> None:
    replacement = (root / "vision-quadrants.png").read_bytes() + b"forged"
    (root / "vision-quadrants.png").write_bytes(replacement)

    def mutate(data: dict) -> None:
        data["fixture"]["sha256"] = hashlib.sha256(replacement).hexdigest()
        data["fixture"]["size"] = len(replacement)
        url = "data:image/png;base64," + base64.b64encode(replacement).decode()
        for key in ("direct_vision", "proxy_vision"):
            row = data["checks"][key]
            row["request"]["messages"][0]["content"][1]["image_url"]["url"] = url
            row["request_sha256"] = request_sha(row["request"])

    mutate_json(root / "acceptance.json", mutate)


def install_ocr_receipts(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))
    fixture = verify["canonical_ocr_fixture"]()
    (root / "vision-ocr.png").write_bytes(fixture)
    body = verify["ocr_request"](fixture)
    acceptance = json.loads((root / "acceptance.json").read_text())
    template = acceptance["checks"]["direct_vision"]

    def receipt(route: str) -> dict:
        row = json.loads(json.dumps(template))
        url = f"http://127.0.0.1:{4000 if route == 'proxy' else 8000}/v1/chat/completions"
        row.update({
            "passed": True,
            "requested_url": url,
            "effective_url": url,
            "path": "/v1/chat/completions",
            "request": body,
            "request_sha256": request_sha(body),
        })
        row["response"]["model"] = verify["MODEL"]
        row["response"]["object"] = "chat.completion"
        row["response"]["choices"][0]["message"]["content"] = verify["OCR_TEXT"]
        return row

    acceptance["ocr_fixture"] = {
        "path": "vision-ocr.png",
        "sha256": hashlib.sha256(fixture).hexdigest(),
        "size": len(fixture),
        "description": "544x96 RGB PNG with black bitmap text GLM53 OCR 8429",
    }
    acceptance["checks"]["direct_ocr"] = receipt("direct")
    acceptance["checks"]["proxy_ocr"] = receipt("proxy")
    (root / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")


def install_video_receipts(root: Path) -> None:
    verify = runpy.run_path(str(root / "verify.py"))
    fixture = verify["canonical_video_fixture"]()
    (root / "video-tiny.gif").write_bytes(fixture)
    body = verify["video_rejection_request"](fixture)

    def mutate(data: dict) -> None:
        data["video_fixture"] = {
            "path": "video-tiny.gif",
            "sha256": hashlib.sha256(fixture).hexdigest(),
            "size": len(fixture),
            "mime_type": "image/gif",
            "description": "deterministic two-frame 224x224 inline GIF for video=0 rejection",
        }
        template = data["checks"]["direct_remote_media_rejected"]
        for key, port, error in (
            ("direct_video_rejected", 8000, verify["VIDEO_LIMIT_ERROR"]),
            ("proxy_video_rejected", 4000, verify["PROXY_VIDEO_LIMIT_ERROR"]),
        ):
            row = json.loads(json.dumps(template))
            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            row.update({
                "passed": True,
                "requested_url": url,
                "effective_url": url,
                "path": "/v1/chat/completions",
                "http": 400,
                "request": body,
                "request_sha256": request_sha(body),
                "response": error,
            })
            data["checks"][key] = row

    mutate_json(root / "acceptance.json", mutate)


def mutate_video_request(root: Path, key: str) -> None:
    def mutate(data: dict) -> None:
        row = data["checks"][key]
        row["request"]["max_tokens"] = 9
        row["request_sha256"] = request_sha(row["request"])

    mutate_json(root / "acceptance.json", mutate)


def replace_video_fixture_everywhere(root: Path) -> None:
    replacement = (root / "video-tiny.gif").read_bytes() + b"forged"
    (root / "video-tiny.gif").write_bytes(replacement)

    def mutate(data: dict) -> None:
        data["video_fixture"]["sha256"] = hashlib.sha256(replacement).hexdigest()
        data["video_fixture"]["size"] = len(replacement)
        url = "data:image/gif;base64," + base64.b64encode(replacement).decode()
        for key in ("direct_video_rejected", "proxy_video_rejected"):
            row = data["checks"][key]
            row["request"]["messages"][0]["content"][1]["video_url"]["url"] = url
            row["request_sha256"] = request_sha(row["request"])

    mutate_json(root / "acceptance.json", mutate)


def replace_ocr_fixture_everywhere(root: Path) -> None:
    replacement = (root / "vision-ocr.png").read_bytes() + b"forged"
    (root / "vision-ocr.png").write_bytes(replacement)

    def mutate(data: dict) -> None:
        data["ocr_fixture"]["sha256"] = hashlib.sha256(replacement).hexdigest()
        data["ocr_fixture"]["size"] = len(replacement)
        url = "data:image/png;base64," + base64.b64encode(replacement).decode()
        for key in ("direct_ocr", "proxy_ocr"):
            row = data["checks"][key]
            row["request"]["messages"][0]["content"][1]["image_url"]["url"] = url
            row["request_sha256"] = request_sha(row["request"])

    mutate_json(root / "acceptance.json", mutate)


def substitute_direct_ocr_request(root: Path) -> None:
    def mutate(data: dict) -> None:
        row = data["checks"]["direct_ocr"]
        row["request"]["messages"][0]["content"][0]["text"] = "Read unrelated text."
        row["request_sha256"] = request_sha(row["request"])

    mutate_json(root / "acceptance.json", mutate)


def mutate_inspect(data: dict, host: str, mutator) -> None:
    command = data["hosts"][host]["docker_inspect"]
    inspect = json.loads(command["stdout"])
    mutator(inspect[0])
    command["stdout"] = json.dumps(inspect) + "\n"


def mutate_sparkrun_status(data: dict) -> None:
    status = json.loads(data["status"]["stdout"])
    status["groups"][data["cluster_id"]]["containers"][1]["status"] = "Exited (0)"
    data["status"]["stdout"] = json.dumps(status) + "\n"


def shift_telemetry_outside_window(root: Path) -> None:
    path = root / "load-telemetry.log"
    text, count = re.subn(
        r"(sample=(?:before|during|after) time=)([^\n]+)",
        lambda match: match.group(1)
        + (datetime.fromisoformat(match.group(2)) + timedelta(hours=1)).isoformat(),
        path.read_text(),
    )
    if count != 3:
        raise AssertionError(f"expected three telemetry timestamps, found {count}")
    path.write_text(text)


def install_expected_top_command(root: Path, host: str, rank: int) -> None:
    expected = runpy.run_path(str(VERIFY))["expected_runtime_command"](rank)

    def mutate(data: dict) -> None:
        top = data["hosts"][host]["docker_top"]
        lines = top["stdout"].splitlines()
        matches = 0
        for index, line in enumerate(lines):
            match = re.fullmatch(
                r"(\s*[1-9]\d*\s+[0-9]+\s+[1-9]\d*\s+[1-9]\d*\s+)/usr/bin/python3 /usr/local/bin/vllm serve .+",
                line,
            )
            if match:
                lines[index] = match.group(1) + expected
                matches += 1
        if matches != 1:
            raise AssertionError(f"expected one serving command for {host}, found {matches}")
        top["stdout"] = "\n".join(lines) + "\n"

    mutate_json(root / "runtime.json", mutate)


def install_expected_top_inventory(root: Path, host: str, rank: int) -> None:
    verify = runpy.run_path(str(root / "verify.py"))
    contract = verify["reviewed_recipe_command"](RECIPE)

    def mutate(data: dict) -> None:
        row = data["hosts"][host]
        top = row["docker_top"]
        parsed = []
        for line in top["stdout"].splitlines()[1:]:
            match = re.fullmatch(
                r"\s*([1-9]\d*)\s+[0-9]+\s+[1-9]\d*\s+[1-9]\d*\s+(.+)",
                line,
            )
            if match is None:
                raise AssertionError(f"unexpected legacy docker top row: {line!r}")
            parsed.append([int(match.group(1)), match.group(2)])
        replacements = {
            "wrapper": verify["expected_wrapper_command"](rank, contract),
            "runtime": verify["expected_runtime_command"](rank, contract),
        }
        counts = {key: 0 for key in replacements}
        for item in parsed:
            command = item[1]
            kind = (
                "wrapper" if "serve_wrapper.sh " in command
                else "runtime" if "/usr/local/bin/vllm serve " in command
                else None
            )
            if kind is not None:
                item[1] = replacements[kind]
                counts[kind] += 1
        if counts != {"wrapper": 1, "runtime": 1}:
            raise AssertionError(f"unexpected docker top serving cardinalities: {counts}")
        serving_match = re.fullmatch(
            r"pid=([1-9]\d*)\ncommand=.+\n?", row["serving_process"]["stdout"]
        )
        if serving_match is None:
            raise AssertionError("serving PID receipt missing")
        serving_pid = int(serving_match.group(1))
        wrapper_pid = next(pid for pid, command in parsed if command == replacements["wrapper"])
        serving_host_pid = next(
            pid for pid, command in parsed if command == replacements["runtime"]
        )
        parent_host_pid = max(pid for pid, _ in parsed) + 100000
        container_id = json.loads(row["docker_inspect"]["stdout"])[0]["Id"]
        by_command = {command: pid for pid, command in parsed}
        launcher_pid = by_command[verify["CONTAINER_LAUNCHER_COMMAND"]]
        shell_pid = by_command[verify["CONTAINER_SHELL_COMMAND"]]
        watchdog_pid = by_command[verify["WATCHDOG_COMMAND"]]
        engine_pid = by_command.get("VLLM::EngineCore")
        output = ["PID PPID PGID SID COMMAND"]
        resource_pattern = re.compile(
            r"/usr/bin/python3 -c from multiprocessing[.]resource_tracker import main;main\([1-9]\d*\)"
        )
        for pid, command in parsed:
            if command == replacements["runtime"]:
                ppid, pgid, sid = wrapper_pid, serving_host_pid, serving_host_pid
            elif command == replacements["wrapper"]:
                ppid, pgid, sid = parent_host_pid, wrapper_pid, wrapper_pid
            elif command == verify["CONTAINER_LAUNCHER_COMMAND"]:
                ppid, pgid, sid = parent_host_pid, launcher_pid, launcher_pid
            elif command == verify["CONTAINER_SHELL_COMMAND"]:
                ppid, pgid, sid = launcher_pid, launcher_pid, launcher_pid
            elif command == "sleep infinity":
                ppid, pgid, sid = shell_pid, launcher_pid, launcher_pid
            elif command == verify["WATCHDOG_COMMAND"]:
                ppid, pgid, sid = parent_host_pid, watchdog_pid, watchdog_pid
            elif resource_pattern.fullmatch(command):
                ppid, pgid, sid = serving_host_pid, serving_host_pid, serving_host_pid
            elif command == "VLLM::EngineCore":
                ppid, pgid, sid = serving_host_pid, serving_host_pid, serving_host_pid
            elif command == "VLLM::Worker_TP0":
                if engine_pid is None:
                    raise AssertionError("rank-0 EngineCore missing")
                ppid, pgid, sid = engine_pid, serving_host_pid, serving_host_pid
            elif command == "VLLM::Worker_TP1":
                ppid, pgid, sid = serving_host_pid, serving_host_pid, serving_host_pid
            elif command == "sleep 5":
                ppid, pgid, sid = watchdog_pid, watchdog_pid, watchdog_pid
            else:
                raise AssertionError(f"unexpected docker top command: {command!r}")
            output.append(f"{pid} {ppid} {pgid} {sid} {command}")
        top["stdout"] = "\n".join(output) + "\n"
        top_command = f"docker top {row['container']} -eo pid,ppid,pgid,sid,args"
        top["argv"] = verify["expected_remote_argv"](host, top_command)
        namespace_command = verify["pid_namespace_command"](serving_host_pid)
        row["pid_namespace"] = {
            "argv": verify["expected_remote_argv"](host, namespace_command),
            "returncode": 0,
            "stdout": (
                f"host_pid={serving_host_pid}\ncontainer_pid={serving_pid}\n"
            ),
            "stderr": "",
        }
        lineage_command = verify["wrapper_lineage_command"](wrapper_pid)
        row["wrapper_lineage"] = {
            "argv": verify["expected_remote_argv"](host, lineage_command),
            "returncode": 0,
            "stdout": json.dumps({
                "wrapper_host_pid": wrapper_pid,
                "parent_host_pid": parent_host_pid,
                "parent_executable": "/usr/bin/containerd-shim-runc-v2",
                "parent_namespace": "moby",
                "parent_container_id": container_id,
                "parent_address": "/run/containerd/containerd.sock",
            }, sort_keys=True) + "\n",
            "stderr": "",
        }
        marker_command = verify["serve_marker_command"](row["container"])
        row["serve_markers"] = {
            "argv": verify["expected_remote_argv"](host, marker_command),
            "returncode": 0,
            "stdout": json.dumps({
                "required_marker_counts": {
                    marker: 1 for marker in verify["REQUIRED_SERVE_MARKERS"]
                },
                "e3_markers": [{
                    "grouped_calls": 7,
                    "fat_expert_runs": 11,
                    "configured_tier": "grouped",
                    "effective_tier": "grouped",
                }],
                "fatal_matches": [],
            }, sort_keys=True) + "\n",
            "stderr": "",
        }
        listener_command = verify["direct_listener_command"](row["container"], host, rank)
        listeners = (
            [{
                "bind": "0.0.0.0",
                "inode": "67890",
                "pid": serving_pid,
                "command": replacements["runtime"],
            }]
            if rank == 0 else []
        )
        row["direct_listener"] = {
            "argv": verify["expected_remote_argv"](host, listener_command),
            "returncode": 0,
            "stdout": json.dumps({
                "namespace": "container" if rank == 0 else "host",
                "host": host,
                "container": row["container"],
                "rank": rank,
                "endpoint": {"port": 8000},
                "listeners": listeners,
            }, sort_keys=True) + "\n",
            "stderr": "",
        }

    mutate_json(root / "runtime.json", mutate)


def mutate_wrapper_lineage_field(
    root: Path, host: str, field: str, value: object
) -> None:
    def mutate(data: dict) -> None:
        receipt = data["hosts"][host]["wrapper_lineage"]
        payload = json.loads(receipt["stdout"])
        payload[field] = value
        receipt["stdout"] = json.dumps(payload, sort_keys=True) + "\n"

    mutate_json(root / "runtime.json", mutate)


def mutate_serve_markers(root: Path, host: str, kind: str) -> None:
    def mutate(data: dict) -> None:
        receipt = data["hosts"][host]["serve_markers"]
        payload = json.loads(receipt["stdout"])
        if kind == "missing":
            marker = next(iter(payload["required_marker_counts"]))
            payload["required_marker_counts"][marker] = 0
        elif kind == "fatal":
            payload["fatal_matches"] = ["CUDA error: an illegal memory access"]
        elif kind == "tier":
            payload["e3_markers"][0]["effective_tier"] = "fallback"
        else:
            raise AssertionError(f"unknown serve marker mutation {kind}")
        receipt["stdout"] = json.dumps(payload, sort_keys=True) + "\n"

    mutate_json(root / "runtime.json", mutate)


def mutate_top_process_relation(
    root: Path, host: str, command_fragment: str, column: str, value: int
) -> None:
    if column not in {"ppid", "pgid", "sid"}:
        raise AssertionError(f"unsupported process relation column: {column}")

    def mutate(data: dict) -> None:
        receipt = data["hosts"][host]["docker_top"]
        lines = receipt["stdout"].splitlines()
        changed = 0
        output = [lines[0]]
        for line in lines[1:]:
            match = re.fullmatch(
                r"\s*([1-9]\d*)\s+([0-9]+)\s+([1-9]\d*)\s+([1-9]\d*)\s+(.+)",
                line,
            )
            if match is None:
                raise AssertionError(f"unexpected docker top row: {line!r}")
            fields = [int(match.group(i)) for i in range(1, 5)]
            command = match.group(5)
            command_matches = (
                re.fullmatch(command_fragment, command) is not None
                if command_fragment.startswith("^")
                else command_fragment in command
            )
            if command_matches:
                fields[{"ppid": 1, "pgid": 2, "sid": 3}[column]] = value
                changed += 1
            output.append(
                f"{fields[0]} {fields[1]} {fields[2]} {fields[3]} {command}"
            )
        if changed != 1:
            raise AssertionError(
                f"expected one process relation mutation for {command_fragment!r}, got {changed}"
            )
        receipt["stdout"] = "\n".join(output) + "\n"

    mutate_json(root / "runtime.json", mutate)


def mutate_named_top_command(
    root: Path, host: str, command_marker: str, old: str, new: str
) -> None:
    def mutate(data: dict) -> None:
        top = data["hosts"][host]["docker_top"]
        lines = top["stdout"].splitlines()
        indexes = [index for index, line in enumerate(lines) if command_marker in line]
        if len(indexes) != 1 or old not in lines[indexes[0]]:
            raise AssertionError("target docker top command or mutation text missing")
        lines[indexes[0]] = lines[indexes[0]].replace(old, new, 1)
        top["stdout"] = "\n".join(lines) + "\n"

    mutate_json(root / "runtime.json", mutate)


def mutate_named_top_pid(root: Path, host: str, command_marker: str, new_pid: int) -> None:
    def mutate(data: dict) -> None:
        top = data["hosts"][host]["docker_top"]
        lines = top["stdout"].splitlines()
        indexes = [index for index, line in enumerate(lines) if command_marker in line]
        if len(indexes) != 1:
            raise AssertionError("target docker top PID row missing")
        fields = lines[indexes[0]].split(maxsplit=4)
        fields[0] = str(new_pid)
        lines[indexes[0]] = " ".join(fields)
        top["stdout"] = "\n".join(lines) + "\n"

    mutate_json(root / "runtime.json", mutate)


def install_finalized_metadata_contract(root: Path) -> None:
    run_id = "0123456789abcdef"
    verifier_path = root / "verify.py"
    text = verifier_path.read_text()
    match = re.findall(
        r'^EXPECTED_ACCEPTANCE_RUN_ID = "(?:PENDING_RECAPTURE|[0-9a-f]{16})"$',
        text,
        re.MULTILINE,
    )
    if len(match) != 1:
        raise AssertionError("single run-ID constant missing")
    verifier_path.write_text(
        text.replace(match[0], f'EXPECTED_ACCEPTANCE_RUN_ID = "{run_id}"')
    )
    verify = runpy.run_path(str(verifier_path))

    def acceptance_mutator(data: dict) -> None:
        data.update({
            "schema": 1,
            "process_role": "exact_final",
            "passed": True,
            "run_id": run_id,
            "invocation_argv": verify["EXPECTED_ACCEPTANCE_ARGV"],
        })

    mutate_json(root / "acceptance.json", acceptance_mutator)
    launch_epoch = int((root / "launch-epoch.txt").read_text())

    def runtime_mutator(data: dict) -> None:
        data.update({
            "schema": 1,
            "process_role": "exact_final",
            "producer": "capture_runtime.py",
            "acceptance_run_id": run_id,
            "capture_argv": verify["expected_capture_argv"](launch_epoch, run_id),
        })

    mutate_json(root / "runtime.json", runtime_mutator)
    refresh_artifact_binding(root, "acceptance.json")
    refresh_artifact_binding(root, "runtime.json")


def install_bound_runtime_record(root: Path, host: str, rank: int) -> None:
    install_expected_top_command(root, host, rank)
    expected = runpy.run_path(str(VERIFY))["expected_runtime_command"](rank)

    def mutate(data: dict) -> None:
        row = data["hosts"][host]
        container = f"{data['cluster_id']}_node_{rank}"
        pid = dict(re.findall(r"^(pid|path|sha256|package_version)=(.+)$", row["nccl_loaded"]["stdout"], re.MULTILINE))["pid"]
        row["ssh_host"] = host
        row["rank"] = rank
        row["container"] = container
        row["serving_process"] = {
            "argv": ["ssh", "-o", "BatchMode=yes", host, "bash", "-lc", f"docker exec {container} inspect serving process"],
            "returncode": 0,
            "stdout": f"pid={pid}\ncommand={expected}\n",
            "stderr": "",
        }
        row["nccl_loaded"]["argv"] = [
            "ssh", "-o", "BatchMode=yes", host, "bash", "-lc",
            f"docker exec {container} read /proc/{pid}/maps",
        ]

    mutate_json(root / "runtime.json", mutate)


def install_secure_status_command(root: Path) -> None:
    defaults = runpy.run_path(str(VERIFY))["IMMUTABLE_DEFAULTS"]

    def mutate(data: dict) -> None:
        status = json.loads(data["status"]["stdout"])
        state = status["groups"][data["cluster_id"]]["meta"]["recipe_state"]
        state["_raw"]["defaults"] = defaults
        state["_raw"]["command"] = "vllm serve --host 0.0.0.0 --port 8000 --allowed-media-domains media.invalid"
        data["status"]["stdout"] = json.dumps(status) + "\n"

    mutate_json(root / "runtime.json", mutate)


def install_telemetry_capture_hash(root: Path) -> None:
    mutate_json(
        root / "acceptance.json",
        lambda data: data.__setitem__(
            "telemetry_capture_script_sha256",
            hashlib.sha256((root / "capture_load_telemetry.py").read_bytes()).hexdigest(),
        ),
    )


def install_stop_receipts(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))

    def receipt(request: dict, choice: dict) -> dict:
        return {
            "passed": True,
            "http": 200,
            "started_at": 10.0,
            "completed_at": 20.0,
            "request": request,
            "request_sha256": request_sha(request),
            "response": {
                "model": verify["MODEL"],
                "object": "chat.completion",
                "choices": [choice],
            },
        }

    def mutate(data: dict) -> None:
        reasoning = verify["reasoning_open_stop_request"]()
        disabled = verify["thinking_disabled_stop_request"]()
        data["checks"]["direct_reasoning_open_stop"] = receipt(
            reasoning,
            {
                "message": {
                    "reasoning": "... 144 ...",
                    "content": "GLM53_REASONING_STOP_OK",
                },
                "finish_reason": "stop",
                "stop_reason": 154827,
            },
        )
        data["checks"]["direct_thinking_disabled_stop"] = receipt(
            disabled,
            {
                "message": {"reasoning": None, "content": "BEFORE "},
                "finish_reason": "stop",
                "stop_reason": "Question:",
            },
        )

    mutate_json(root / "acceptance.json", mutate)


def artifact_binding_negative_controls() -> None:
    verify = runpy.run_path(str(VERIFY))
    checker = verify["artifact_binding_failures"]
    with tempfile.TemporaryDirectory(prefix="glm53-artifact-binding-") as tmp:
        copied = Path(tmp) / "evidence"
        shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns("__pycache__"))
        names = tuple(verify["FINAL_ARTIFACT_SHA256"])
        bindings = {
            name: hashlib.sha256((copied / name).read_bytes()).hexdigest()
            for name in names
        }
        assert checker(copied, bindings) == []
        for name in bindings:
            original = (copied / name).read_bytes()
            (copied / name).write_bytes(original + b"forged\n")
            write_manifest(copied)
            assert f"final artifact exact hash: {name}" in checker(copied, bindings)
            (copied / name).write_bytes(original)
            print(f"rejected: self-consistent substitution of {name}")


def install_load_overlap(root: Path) -> None:
    during = runpy.run_path(str(VERIFY))["parse_telemetry"](
        root / "load-telemetry.log"
    )["during"]["timestamp"]
    mutate_json(
        root / "acceptance.json",
        lambda data: data["checks"]["direct_telemetry_load"].update(
            {"started_at": during - 1, "completed_at": during + 1}
        ),
    )


def move_load_receipt_outside_during(data: dict) -> None:
    acceptance_start = data["started_at"]
    row = data["checks"]["direct_telemetry_load"]
    row["started_at"] = acceptance_start
    row["completed_at"] = acceptance_start + 0.01


def install_telemetry_intervals(root: Path) -> None:
    path = root / "load-telemetry.log"
    if " acquisition_started_at=" in path.read_text():
        return
    output: list[str] = []
    current_time: datetime | None = None
    current_host: str | None = None
    for line in path.read_text().splitlines():
        sample = re.fullmatch(r"sample=\S+ time=(\S+)", line)
        if sample:
            if current_host is not None and current_time is not None:
                output.append(
                    f"host={current_host} acquisition_completed_at="
                    f"{(current_time + timedelta(milliseconds=100)).isoformat()}"
                )
            current_host = None
            current_time = datetime.fromisoformat(sample.group(1))
            output.append(line)
            continue
        if line.startswith("host="):
            if current_host is not None and current_time is not None:
                output.append(
                    f"host={current_host} acquisition_completed_at="
                    f"{(current_time + timedelta(milliseconds=100)).isoformat()}"
                )
            current_host = line.split("=", 1)[1]
            assert current_time is not None
            output.append(
                f"host={current_host} acquisition_started_at="
                f"{(current_time - timedelta(milliseconds=100)).isoformat()}"
            )
            continue
        output.append(line)
    if current_host is not None and current_time is not None:
        output.append(
            f"host={current_host} acquisition_completed_at="
            f"{(current_time + timedelta(milliseconds=100)).isoformat()}"
        )
    path.write_text("\n".join(output) + "\n")
    install_load_overlap(root)


def shift_during_host_interval(root: Path, host: str) -> None:
    path = root / "load-telemetry.log"
    text = path.read_text()
    before, during_and_after = text.split("sample=during", 1)
    during, after = during_and_after.split("sample=after", 1)
    acceptance = json.loads((root / "acceptance.json").read_text())
    completed = acceptance["checks"]["direct_telemetry_load"]["completed_at"]
    start = datetime.fromtimestamp(completed + 10).astimezone().isoformat()
    end = datetime.fromtimestamp(completed + 11).astimezone().isoformat()
    during, count1 = re.subn(
        rf"host={re.escape(host)} acquisition_started_at=\S+",
        f"host={host} acquisition_started_at={start}",
        during,
        count=1,
    )
    during, count2 = re.subn(
        rf"host={re.escape(host)} acquisition_completed_at=\S+",
        f"host={host} acquisition_completed_at={end}",
        during,
        count=1,
    )
    assert count1 == count2 == 1
    path.write_text(before + "sample=during" + during + "sample=after" + after)


def substitute_remote_command(root: Path, host: str, key: str, command: str) -> None:
    mutate_json(
        root / "runtime.json",
        lambda data: data["hosts"][host][key]["argv"].__setitem__(-1, command),
    )


def install_runtime_env_command(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))
    host = "192.168.178.47"

    def mutate(data: dict) -> None:
        container = data["hosts"][host]["container"]
        command = (
            f"docker exec {shlex.quote(container)} bash -lc "
            + shlex.quote(verify["runtime_env_command"]())
        )
        data["hosts"][host]["runtime_env"]["argv"] = verify[
            "expected_remote_argv"
        ](host, command)

    mutate_json(root / "runtime.json", mutate)


def run_exact_runtime_command_controls() -> None:
    host = "192.168.178.47"
    keys = (
        "container_discovery",
        "earlyoom",
        "docker_inspect",
        "docker_top",
        "docker_logs",
        "serve_log",
        "runtime_env",
        "roce_records",
        "runtime_overlay",
        "runtime_mod_manifest",
        "serving_process",
        "nccl_loaded",
        "rdma",
        "memory",
        "kernel_since_launch",
        "kernel_after_acceptance",
    )
    for index, key in enumerate(keys):
        replacement = "'true'" if index % 2 == 0 else "'printf unrelated'"
        prepare = (
            install_runtime_overlay_contract
            if key == "runtime_overlay"
            else install_runtime_mod_manifest_receipts
            if key == "runtime_mod_manifest"
            else install_roce_records
            if key == "roce_records"
            else install_runtime_env_command
            if key == "runtime_env"
            else (lambda root: install_expected_top_inventory(root, host, 0))
            if key == "docker_top"
            else None
        )
        rejected(
            f"{key} command substituted with refreshed manifest",
            lambda root, receipt=key, value=replacement: substitute_remote_command(
                root, host, receipt, value
            ),
            f"{host} {key} exact command",
            prepare=prepare,
        )


def install_image_inspect_receipt(root: Path, host: str = "192.168.178.47") -> None:
    verify = runpy.run_path(str(VERIFY))
    image = verify["IMAGE"]
    command = f"docker image inspect {shlex.quote(image)}"

    def mutate(data: dict) -> None:
        data["hosts"][host]["image_inspect"] = {
            "argv": [
                "ssh", "-o", "BatchMode=yes", host, "bash", "-lc", shlex.quote(command)
            ],
            "returncode": 0,
            "stdout": json.dumps([{
                "Id": "sha256:9581c4c7425786be27a7904a78c01bb27ae138e33840e605ecc706e76c761900",
                "RepoDigests": [image],
            }]) + "\n",
            "stderr": "",
        }

    mutate_json(root / "runtime.json", mutate)


def mutate_image_inspect(root: Path, mutator) -> None:
    def mutate(data: dict) -> None:
        receipt = data["hosts"]["192.168.178.47"]["image_inspect"]
        payload = json.loads(receipt["stdout"])
        mutator(payload[0])
        receipt["stdout"] = json.dumps(payload) + "\n"

    mutate_json(root / "runtime.json", mutate)


def add_root_mount(inspect: dict) -> None:
    inspect["HostConfig"]["Binds"].append("/:/host:rw")
    inspect["Mounts"].append({
        "Type": "bind",
        "Source": "/",
        "Destination": "/host",
        "Mode": "rw",
        "RW": True,
        "Propagation": "rprivate",
    })


def install_listener_receipts(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))
    host = "192.168.178.47"
    runtime = json.loads((root / "runtime.json").read_text())
    row = runtime["hosts"][host]
    serving = re.fullmatch(
        r"pid=(\d+)\ncommand=(.+)\n?", row["serving_process"]["stdout"]
    )
    assert serving is not None
    serving_pid, serving_command = serving.groups()
    proxy_supervisor_pid = 760552
    proxy_parent_pid = 760553
    proxy_pid = 760554
    supervisor = {
        "role": "autodiscover_supervisor",
        "pid": proxy_supervisor_pid,
        "ppid": 2000,
        "pgid": proxy_supervisor_pid,
        "sid": proxy_supervisor_pid,
        "started_at": 99.0,
        "argv": verify["proxy_supervisor_argv"](),
    }
    parent = {
        "role": "uv_parent",
        "pid": proxy_parent_pid,
        "ppid": proxy_supervisor_pid,
        "pgid": proxy_parent_pid,
        "sid": proxy_parent_pid,
        "started_at": 100.0,
        "argv": verify["proxy_parent_argv"]("dynamic"),
    }
    child = {
        "role": "listener_child",
        "pid": proxy_pid,
        "ppid": proxy_parent_pid,
        "pgid": proxy_parent_pid,
        "sid": proxy_parent_pid,
        "started_at": 101.0,
        "argv": verify["proxy_child_argv"]("dynamic"),
    }
    runtime["proxy_status"] = {
        "argv": ["sparkrun", "proxy", "status"],
        "returncode": 0,
        "stdout": verify["canonical_proxy_status"](
            proxy_parent_pid, proxy_supervisor_pid
        ),
        "stderr": "",
    }
    runtime["proxy_processes"] = [supervisor, parent, child]
    runtime["proxy_listener"] = {
        "probe_argv": ["python3", "-c", verify["LISTENER_PROBE"], "4000"],
        "bind": "0.0.0.0",
        "port": 4000,
        "inode": "12345",
        **child,
    }
    runtime["proxy_upstream_sockets"] = [{
        "owner_pid": proxy_pid,
        "state": "ESTABLISHED",
        "remote_address": "192.168.178.47",
        "remote_port": 8000,
        "inode": "54321",
    }]
    command = verify["direct_listener_command"](row["container"], host, 0)
    row["direct_listener"] = {
        "argv": verify["expected_remote_argv"](host, command),
        "returncode": 0,
        "stdout": json.dumps({
            "namespace": "container",
            "host": host,
            "container": row["container"],
            "rank": 0,
            "endpoint": {"port": 8000},
            "listeners": [{
                "bind": "0.0.0.0",
                "inode": "67890",
                "pid": int(serving_pid),
                "command": serving_command,
            }],
        }, sort_keys=True) + "\n",
        "stderr": "",
    }
    (root / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")


def mutate_listener(root: Path, key: str, field: str, value) -> None:
    mutate_json(root / "runtime.json", lambda data: data[key].__setitem__(field, value))


def mutate_direct_listener(root: Path, field: str, value) -> None:
    def mutate(data: dict) -> None:
        receipt = data["hosts"]["192.168.178.47"]["direct_listener"]
        payload = json.loads(receipt["stdout"])
        if field == "port":
            payload["endpoint"]["port"] = value
        elif field in {"pid", "bind", "inode", "command"}:
            payload["listeners"][0][field] = value
        else:
            payload[field] = value
        receipt["stdout"] = json.dumps(payload) + "\n"

    mutate_json(root / "runtime.json", mutate)


def mutate_worker_listener(root: Path, mode: str) -> None:
    def mutate(data: dict) -> None:
        row = data["hosts"]["192.168.178.46"]
        if mode == "omitted":
            row.pop("direct_listener", None)
            return
        if mode == "null":
            row["direct_listener"] = None
            return
        receipt = row["direct_listener"]
        payload = json.loads(receipt["stdout"])
        if mode == "nonempty":
            payload["listeners"] = [{"pid": 7}]
        elif mode == "relabelled":
            payload["rank"] = 0
        else:
            raise AssertionError(f"unknown worker listener mutation: {mode}")
        receipt["stdout"] = json.dumps(payload, sort_keys=True) + "\n"

    mutate_json(root / "runtime.json", mutate)


def install_e3_execution_markers(root: Path) -> None:
    def mutate(data: dict) -> None:
        for row in data["hosts"].values():
            log = row["serve_log"]
            if "[glm53-e3-executed]" not in log["stdout"]:
                log["stdout"] += (
                    "[glm53-e3-executed] grouped_calls=1 fat_expert_runs=2 "
                    "configured_tier=grouped effective_tier=grouped\n"
                )

    mutate_json(root / "runtime.json", mutate)


def mutate_e3_marker(root: Path, host: str, old: str, new: str) -> None:
    def mutate(data: dict) -> None:
        log = data["hosts"][host]["serve_log"]
        assert old in log["stdout"]
        position = log["stdout"].rfind(old)
        log["stdout"] = (
            log["stdout"][:position] + new + log["stdout"][position + len(old):]
        )

    mutate_json(root / "runtime.json", mutate)


def zero_e3_counter(root: Path, host: str) -> None:
    def mutate(data: dict) -> None:
        log = data["hosts"][host]["serve_log"]
        replaced, count = re.subn(
            r"(\[glm53-e3-executed\][^\n]*fat_expert_runs=)\d+",
            r"\g<1>0",
            log["stdout"],
            count=1,
        )
        assert count == 1
        log["stdout"] = replaced

    mutate_json(root / "runtime.json", mutate)


def install_runtime_overlay_contract(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))

    def mutate(data: dict) -> None:
        for host, row in data["hosts"].items():
            container = row["container"]
            receipt = row["runtime_overlay"]
            receipt["argv"] = verify["expected_runtime_overlay_argv"](host, container)
            receipt["returncode"] = 0
            receipt["stderr"] = ""
            if "[OK] complete GLM-5.3 runtime patched state verified" not in receipt["stdout"]:
                receipt["stdout"] += "[OK] complete GLM-5.3 runtime patched state verified\n"

    mutate_json(root / "runtime.json", mutate)


def install_runtime_mod_manifest_receipts(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))
    failures, entries = verify["mod_manifest_failures"](MOD)
    assert failures == []
    expected = verify["EXPECTED_MOD_MANIFEST_SHA256"]
    stdout = "\n".join(
        [f"manifest_sha256={expected}"] + [f"{entry}: OK" for entry in entries]
    ) + "\n"

    def mutate(data: dict) -> None:
        for host, row in data["hosts"].items():
            container = row["container"]
            command = verify["runtime_mod_manifest_command"](container, expected)
            row["runtime_mod_manifest"] = {
                "argv": verify["expected_remote_argv"](host, command),
                "returncode": 0,
                "stdout": stdout,
                "stderr": "",
            }
        data["expected_mod_manifest_sha256"] = expected

    mutate_json(root / "runtime.json", mutate)


def install_roce_records(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))

    def mutate(data: dict) -> None:
        for host, row in data["hosts"].items():
            command = verify["roce_records_command"](row["container"])
            row["roce_records"] = {
                "argv": verify["expected_remote_argv"](host, command),
                "returncode": 0,
                "stdout": json.dumps(verify["EXPECTED_ROCE"][host], sort_keys=True) + "\n",
                "stderr": "",
            }

    mutate_json(root / "runtime.json", mutate)


def replace_telemetry_hca(root: Path, old: str, new: str) -> None:
    path = root / "load-telemetry.log"
    path.write_text(path.read_text().replace(old, new))


def install_acceptance_routes(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))

    def bind(key: str, row: dict) -> None:
        route = "proxy" if key.startswith("proxy") else "direct"
        base = "http://127.0.0.1:4000" if route == "proxy" else "http://127.0.0.1:8000"
        path = "/v1/models" if key.endswith("models") else "/v1/chat/completions"
        row.update({"requested_url": base + path, "effective_url": base + path, "path": path})

    def mutate(data: dict) -> None:
        data["invocation_argv"] = verify["EXPECTED_ACCEPTANCE_ARGV"]
        for key, row in data["checks"].items():
            if key == "direct_concurrency_c4":
                for index, item in enumerate(row["results"]):
                    bind(f"direct_concurrency_c4[{index}]", item)
            else:
                bind(key, row)

    mutate_json(root / "acceptance.json", mutate)


def install_proxy_remote_media_receipt(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))

    def mutate(data: dict) -> None:
        direct = data["checks"]["direct_remote_media_rejected"]
        proxy = json.loads(json.dumps(direct))
        proxy["requested_url"] = "http://127.0.0.1:4000/v1/chat/completions"
        proxy["effective_url"] = "http://127.0.0.1:4000/v1/chat/completions"
        proxy["path"] = "/v1/chat/completions"
        proxy["response"] = verify["PROXY_REMOTE_MEDIA_ERROR"]
        data["checks"]["proxy_remote_media_rejected"] = proxy

    mutate_json(root / "acceptance.json", mutate)
    install_acceptance_routes(root)


def install_current_acceptance_check_set(root: Path) -> None:
    install_stop_receipts(root)
    install_proxy_remote_media_receipt(root)
    install_ocr_receipts(root)
    install_video_receipts(root)


def install_launch_receipt(root: Path) -> None:
    verify = runpy.run_path(str(VERIFY))
    epoch = int((root / "launch-epoch.txt").read_text().strip())
    receipt = {
        "schema": 1,
        "cluster_id": verify["CLUSTER"],
        "recipe_sha256": hashlib.sha256(verify["RECIPE"].read_bytes()).hexdigest(),
        "recipe_command": verify["reviewed_recipe_command"](verify["RECIPE"])["raw_command"],
        "mod_manifest_sha256": verify["EXPECTED_MOD_MANIFEST_SHA256"],
        "launch_epoch": epoch,
        "recorded_at": float(epoch),
        "argv": verify["EXPECTED_LAUNCH_ARGV"],
    }
    receipt_path = root / "launch-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    mutate_json(
        root / "runtime.json",
        lambda data: data.__setitem__(
            "launch_receipt_sha256", hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        ),
    )


def shift_launch_epoch_self_consistently(root: Path) -> None:
    acceptance = json.loads((root / "acceptance.json").read_text())
    shifted = int(acceptance["started_at"]) - 1
    old = int((root / "launch-epoch.txt").read_text().strip())
    (root / "launch-epoch.txt").write_text(f"{shifted}\n")
    receipt_path = root / "launch-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update({"launch_epoch": shifted, "recorded_at": float(shifted)})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    def mutate(data: dict) -> None:
        data["launch_epoch"] = shifted
        data["launch_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        for row in data["hosts"].values():
            command = row["kernel_since_launch"]
            command["argv"][-1] = command["argv"][-1].replace(
                f"'@{old}'", f"'@{shifted}'"
            )

    mutate_json(root / "runtime.json", mutate)


def install_command_chain(root: Path, recipe_path: Path) -> None:
    recipe_doc = yaml.safe_load(recipe_path.read_text())
    command = recipe_doc["command"]
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()

    mutate_json(
        root / "acceptance.json",
        lambda data: data.__setitem__("recipe_sha256", recipe_sha),
    )

    receipt_path = root / "launch-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update({"recipe_sha256": recipe_sha, "recipe_command": command})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    def runtime_mutator(data: dict) -> None:
        status = json.loads(data["status"]["stdout"])
        status["groups"][data["cluster_id"]]["meta"]["recipe_state"]["_raw"][
            "command"
        ] = command
        data["status"]["stdout"] = json.dumps(status) + "\n"
        data["launch_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    mutate_json(root / "runtime.json", runtime_mutator)
    for rel in ("acceptance.json", "launch-receipt.json", "runtime.json"):
        refresh_artifact_binding(root, rel)
    write_manifest(root)


def command_honest_refresh_control(name: str, mutate_command, expected_failure: str) -> None:
    with tempfile.TemporaryDirectory(prefix="glm53-command-refresh-") as tmp:
        tmp_path = Path(tmp)
        copied = tmp_path / "evidence"
        recipe_path = tmp_path / RECIPE.name
        shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(RECIPE, recipe_path)
        install_command_chain(copied, recipe_path)
        baseline = verifier_failures(
            copied, verifier=copied / "verify.py", recipe=recipe_path
        )
        recipe_doc = yaml.safe_load(recipe_path.read_text())
        recipe_doc["command"] = mutate_command(recipe_doc["command"])
        recipe_path.write_text(yaml.safe_dump(recipe_doc, sort_keys=False))
        install_command_chain(copied, recipe_path)
        failures = verifier_failures(
            copied, verifier=copied / "verify.py", recipe=recipe_path
        )
        if failures.count(expected_failure) <= baseline.count(expected_failure):
            raise AssertionError(
                f"honest command refresh was not rejected: {name}; "
                f"wanted {expected_failure!r}; baseline={baseline!r}; mutated={failures!r}"
            )
    print(f"rejected: {name}")


def write_mod_manifest(mod: Path) -> None:
    paths = sorted(
        path for path in mod.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts
    )
    (mod / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(mod)}\n"
        for path in paths
    ))


def executable_mod_substitution_control() -> None:
    expected_failure = "reviewed local mod manifest digest"
    baseline = verifier_failures(ROOT)
    with tempfile.TemporaryDirectory(prefix="glm53-mod-substitution-") as tmp:
        copied_mod = Path(tmp) / MOD.name
        shutil.copytree(MOD, copied_mod, ignore=shutil.ignore_patterns("__pycache__"))
        run_path = copied_mod / "run.sh"
        run_path.write_bytes(run_path.read_bytes() + b"\n# forged executable substitution\n")
        write_mod_manifest(copied_mod)
        failures = verifier_failures(ROOT, mod=copied_mod)
    if failures.count(expected_failure) <= baseline.count(expected_failure):
        raise AssertionError(
            "self-consistent executable-mod substitution was not rejected by external digest"
        )
    print("rejected: executable mod substituted with regenerated manifest")


def sanitizer_negative_controls() -> None:
    capture = runpy.run_path(str(ROOT / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    redacted = capture["REDACTED"]

    mapping = sanitize({
        "%61pi_key": "mapping-secret",
        "%2561uthorization": "header-secret",
        "headers": {"%2541uthorization": "nested-header-secret"},
        "x_api_key_backup": "wrapped-name-secret",
        "backupapikeycopy": "compact-infix-secret",
        "backup%2561pikeycopy": "encoded-compact-infix-secret",
        "backupsecretcopy": "compact-secret-name",
        "backuptokencopy": "compact-token-name",
        "service_url": "lowercase-connection-name",
        "%70ath": "/v1/models",
    })
    if (
        mapping["%70ath"] != "/v1/models"
        or mapping["headers"]["%2541uthorization"] != redacted
        or any(
            value != redacted
            for key, value in mapping.items()
            if key not in {"headers", "%70ath"}
        )
    ):
        raise AssertionError("encoded sensitive mapping/header key was not redacted safely")
    print("rejected: encoded sensitive mapping and nested header keys")

    standard_environment = sanitize({
        name: "environment-secret"
        for name in (
            "MYSQL_PWD", "DATABASE_URL", "DB_URL", "POSTGRES_URL",
            "POSTGRESQL_URL", "REDIS_URL", "MONGODB_URI", "MONGO_URI",
            "PGURI", "JDBC_URL", "AZURE_STORAGE_ACCOUNT_KEY",
            "DOCKER_AUTH_CONFIG", "SQLALCHEMY_DATABASE_URI",
            "CELERY_BROKER_URL", "AMQP_URL", "RABBITMQ_URL", "KAFKA_URL",
            "ELASTICSEARCH_URL", "OPENSEARCH_URL", "MSSQL_URL", "MYSQL_URL",
            "EXAMPLE_SERVICE_URL", "EXAMPLE_SERVICE_URI", "EXAMPLE_SERVICE_DSN",
        )
    })
    if "secret" in json.dumps(standard_environment):
        raise AssertionError("standard credential environment value was not redacted")
    print("rejected: standard credential environment mappings")

    composed_cases = (
        "PGPASSWORD=raw-secret\n"
        "https://example.invalid/v1%3Fapi_key%3Dencoded-secret\n"
        + urllib.parse.quote(
            json.dumps(json.dumps({"X-Api-Key": "serialized-secret"})),
            safe="",
        ),
        "prefix Authorization: Bearer placeholder suffix "
        + urllib.parse.quote(
            urllib.parse.quote(json.dumps({"api_key": "inline-secret"}), safe=""),
            safe="",
        ),
    )
    if any("secret" in sanitize(payload) for payload in composed_cases):
        raise AssertionError("mixed raw and encoded credential material was not redacted")
    print("rejected: mixed raw and encoded credential fragments")

    header_sequences = sanitize({
        "headers": [
            ["Authorization", "pair-secret"],
            {"name": "Authorization", "value": "object-secret"},
            {"name": "Authorization", "data": "data-secret"},
            "Authorization", "flat-secret",
            "prefix Authorization: Bearer embedded-secret suffix",
        ]
    })
    if "secret" in json.dumps(header_sequences):
        raise AssertionError("nested header sequence value was not redacted")
    print("rejected: nested header sequence representations")

    direct_header_mappings = sanitize({
        "headers": {"name": "Authorization", "value": "direct-secret"},
        "http_headers": {"name": "Authorization", "value": "http-secret"},
        "header": ["Authorization", "singular-secret"],
    })
    if "secret" in json.dumps(direct_header_mappings):
        raise AssertionError("direct or singular header representation was not redacted")
    print("rejected: direct and singular header representations")

    marker = "multitoken-header-secret"
    multitoken_headers = (
        f"X-Api-Key: *** {marker}",
        f"prefix X-Auth-Token: *** {marker} trailing-text",
        f"Cookie: synthetic-name=synthetic-value; private={marker}",
        f'prefix Authorization: Digest ***"synthetic", nonce="{marker}"',
        f'prefix Proxy-Authorization: Digest realm="synthetic", response="{marker}"',
    )
    for payload in multitoken_headers:
        sanitized = sanitize(payload)
        if marker in sanitized or redacted not in sanitized or sanitize(sanitized) != sanitized:
            raise AssertionError("complete multi-token header value was not redacted")
    print("rejected: complete multi-token credential header values")

    marker = "rfc-tchar-header-secret"
    rfc_tchar_headers = []
    for separator in "!#$%&'*+-.^_`|~":
        field = separator.join(("X", "Api", "Key"))
        scalar = f"{field}: synthetic-prefix {marker} trailing-text"
        redacted_line = f"{field}: {redacted}"
        rfc_tchar_headers.extend((
            (scalar, redacted_line),
            (f"embedded-prefix {scalar}", f"embedded-prefix {redacted_line}"),
            (
                json.dumps({"line": scalar}),
                json.dumps({"line": redacted_line}, indent=2),
            ),
            (
                {"headers": [scalar, {"name": field, "value": marker}]},
                {
                    "headers": [
                        redacted_line,
                        {"name": field, "value": redacted},
                    ]
                },
            ),
        ))
    for payload, expected in rfc_tchar_headers:
        sanitized = sanitize(payload)
        if sanitized != expected or marker in json.dumps(sanitized) or sanitize(sanitized) != sanitized:
            raise AssertionError("RFC tchar credential header was not exactly redacted")
    print("rejected: RFC tchar scalar and nested credential headers")

    composition_cases = (
        {"%68eaders": [["Authorization", marker]]},
        {"%2568eaders": [["Authorization", marker]]},
        json.dumps({"%68eaders": [["Authorization", marker]]}),
        {"headers": [{"%6Eame": "Authorization", "value": marker}]},
        {
            "headers": [
                {"name": "Content-Type", "key": "Authorization", "value": marker}
            ]
        },
        {"headers": [{"headerName": "Authorization", "headerValue": marker}]},
        {"headers": [["Authorization", redacted, marker]]},
        {
            "headers": [
                {"name": "Authorization", "value": redacted, "raw": marker}
            ]
        },
        {"headerList": [{"headerName": "Authorization", "headerValue": marker}]},
        {"header_list": [["Authorization", marker]]},
        {"header%4Cist": [{"headerName": "Authorization", "headerValue": marker}]},
        {"argvList": ["tool", "--api-key", marker]},
        {"argv_list": ["tool", "--api-key", marker]},
        {"argv%4Cist": ["tool", "--api-key", marker]},
        json.dumps({"headerList": [["Authorization", marker]]}),
        json.dumps({"argvList": ["tool", "--api-key", marker]}),
        {"headerName": "Authorization", "headerValue": marker},
        {"field_name": "Authorization", "field_value": marker},
        {"httpHeaderName": "Authorization", "httpHeaderValue": marker},
        {"name": "Authorization", "value": marker},
        {"headers": {"items": [["Authorization", marker]]}},
        {"headers": {"items": [["Authorization", marker]], "raw": marker}},
        {"argv": {"items": ["--api-key", marker]}},
        {"argv": {"items": ["--api-key", marker], "raw": marker}},
        {"headers": {"Authorization": marker, "raw": marker}},
        ["Authorization", marker],
        json.dumps({"headers": {"items": [["Authorization", marker]]}}),
        json.dumps({"argv": {"items": ["--api-key", marker]}}),
    )
    for payload in composition_cases:
        sanitized = sanitize(payload)
        if marker in json.dumps(sanitized) or sanitize(sanitized) != sanitized:
            raise AssertionError("structured header composition was not redacted")
    print("rejected: encoded, aliased, ambiguous, and extended structures")

    runtime = json.loads((ROOT / "runtime.json").read_text())
    proxy_argv = next(
        row["argv"]
        for row in runtime["proxy_processes"]
        if any(item.endswith("/bin/litellm") for item in row["argv"])
    )
    canonical_proxy = " ".join(proxy_argv)
    if not capture["safe_command_value"](canonical_proxy):
        raise AssertionError("canonical proxy command was not preserved")
    for index, item in enumerate(proxy_argv[:2]):
        archive_id = item.split("/archive-v0/", 1)[1].split("/", 1)[0]
        mutated_argv = list(proxy_argv)
        mutated_argv[index] = item.replace(
            archive_id, "synthetic-proxy-command-canary", 1
        )
        mutated = " ".join(mutated_argv)
        if sanitize(mutated, "command") != redacted:
            raise AssertionError("noncanonical proxy command was preserved")
    print("rejected: noncanonical proxy archive commands")

    encoded_argv = sanitize({"probe_argv": ["tool", "--api%2Dkey", "argv-secret"]})
    if "secret" in json.dumps(encoded_argv):
        raise AssertionError("encoded credential option value was not redacted")
    print("rejected: encoded argv credential option")

    deep_option = "--api%2Dkey"
    for _ in range(capture["JSON_SANITIZE_MAX_DEPTH"] + 1):
        deep_option = deep_option.replace("%", "%25")
    if capture["percent_decode_fixed_point"](deep_option)[1]:
        raise AssertionError("over-depth argv fixture unexpectedly converged")
    nonconvergent_argv = (
        {"argv": ["--api%ZZkey", marker]},
        {"argvList": [deep_option, marker]},
        {"argv_list": [deep_option, {"opaque": marker}]},
        json.dumps({"argv": ["--api%ZZkey", marker]}),
        json.dumps({"argvList": [deep_option, marker]}),
    )
    if any(marker in json.dumps(sanitize(item)) for item in nonconvergent_argv):
        raise AssertionError("nonconvergent argv record was not wholly redacted")
    print("rejected: malformed and over-depth percent-encoded argv records")

    attached_headers = (
        f"-HAuthorization:Bearer-{marker}",
        f"-H Authorization:Bearer-{marker}",
        f"-H=Authorization:Bearer-{marker}",
        urllib.parse.quote_plus(f"-HAuthorization:Bearer-{marker}"),
    )
    attached_header_argv = (
        {field: [argument]}
        for field in ("argv", "argvList", "argv_list")
        for argument in attached_headers
    )
    if any(marker in json.dumps(sanitize(item)) for item in attached_header_argv):
        raise AssertionError("attached curl header argument was not redacted")
    print("rejected: attached raw and encoded curl header arguments")

    nested_fields = ("argv", "argvList", "argv_list", "argv%4Cist", "argv%5Flist")
    nested_options = ("--api-key", "--api%ZZkey", deep_option)
    nested_wrappers = (
        lambda option: [["tool", option, marker]],
        lambda option: {"items": ["tool", option, marker]},
        lambda option: {"items": {"items": ["tool", option, marker]}},
        lambda option: [{"items": ["tool", option, marker]}],
        lambda option: ({"items": ["tool", option, marker]},),
    )
    nested_argv_cases = []
    for nested_field in nested_fields:
        for nested_option in nested_options:
            for nested_wrapper in nested_wrappers:
                payload = {nested_field: nested_wrapper(nested_option)}
                nested_argv_cases.extend((payload, json.dumps(payload)))
    if len(nested_argv_cases) != 150:
        raise AssertionError("nested argv matrix is incomplete")
    for payload in nested_argv_cases:
        sanitized = sanitize(payload)
        if marker in json.dumps(sanitized) or sanitize(sanitized) != sanitized:
            raise AssertionError("nested argv context was lost")
    print("rejected: nested argv structural context matrix")

    inner_wrappers = (
        lambda item: [item],
        lambda item: [[item]],
        lambda item: {"items": [item]},
        lambda item: {"items": {"items": [item]}},
        lambda item: [{"items": [item]}],
        lambda item: ({"items": [item]},),
    )
    inner_argv_cases = []
    for nested_field in nested_fields:
        for nested_option in nested_options:
            inner = json.dumps(["tool", nested_option, marker])
            serialized = (
                inner,
                urllib.parse.quote(inner, safe=""),
                urllib.parse.quote(urllib.parse.quote(inner, safe=""), safe=""),
            )
            for inner_wrapper in inner_wrappers:
                for item in serialized:
                    payload = {nested_field: inner_wrapper(item)}
                    inner_argv_cases.extend((payload, json.dumps(payload)))
    if len(inner_argv_cases) != 540:
        raise AssertionError("inner-serialized argv matrix is incomplete")
    for payload in inner_argv_cases:
        sanitized = sanitize(payload)
        if marker in json.dumps(sanitized) or sanitize(sanitized) != sanitized:
            raise AssertionError("inner-serialized argv context was lost")
    print("rejected: inner-serialized argv structural context matrix")

    split_wrappers = (
        lambda item: ["tool", item, marker],
        lambda item: ("tool", item, marker),
        lambda item: {"items": ["tool", item, marker]},
        lambda item: {"items": {"items": ["tool", item, marker]}},
        lambda item: [{"items": ["tool", item, marker]}],
        lambda item: ({"items": ["tool", item, marker]},),
    )
    split_argv_cases = []
    for nested_field in nested_fields:
        for nested_option in nested_options:
            inner = json.dumps(nested_option)
            serialized = (
                inner,
                urllib.parse.quote(inner, safe=""),
                urllib.parse.quote(urllib.parse.quote(inner, safe=""), safe=""),
            )
            for split_wrapper in split_wrappers:
                for item in serialized:
                    payload = {nested_field: split_wrapper(item)}
                    split_argv_cases.extend((payload, json.dumps(payload)))
    if len(split_argv_cases) != 540:
        raise AssertionError("split-token argv matrix is incomplete")
    for payload in split_argv_cases:
        sanitized = sanitize(payload)
        if marker in json.dumps(sanitized) or sanitize(sanitized) != sanitized:
            raise AssertionError("split-token argv context was lost")
    print("rejected: individually serialized argv option matrix")

    singleton_argv_cases = []
    singleton = json.dumps(["--api-key"])
    singleton_forms = (
        singleton,
        urllib.parse.quote(singleton, safe=""),
        urllib.parse.quote(urllib.parse.quote(singleton, safe=""), safe=""),
    )
    for nested_field in nested_fields:
        for item in singleton_forms:
            payload = {nested_field: ["tool", item, marker]}
            singleton_argv_cases.extend((payload, json.dumps(payload)))
    if len(singleton_argv_cases) != 30:
        raise AssertionError("singleton argv fragment matrix is incomplete")
    for payload in singleton_argv_cases:
        sanitized = sanitize(payload)
        if marker in json.dumps(sanitized) or sanitize(sanitized) != sanitized:
            raise AssertionError("singleton argv fragment leaked or was unstable")
    print("rejected: serialized singleton argv option fragments")

    nested_fragment_cases = []
    fragments = (
        [["--api-key"]],
        {"items": ["--api-key"]},
        {"items": {"items": ["--api-key"]}},
        [{"items": ["--api-key"]}],
    )
    for nested_field in nested_fields:
        for fragment in fragments:
            inner = json.dumps(fragment)
            forms = (
                inner,
                urllib.parse.quote(inner, safe=""),
                urllib.parse.quote(urllib.parse.quote(inner, safe=""), safe=""),
            )
            for item in forms:
                payload = {nested_field: ["tool", item, marker]}
                nested_fragment_cases.extend((payload, json.dumps(payload)))
    if len(nested_fragment_cases) != 120:
        raise AssertionError("nested serialized argv fragment matrix is incomplete")
    for payload in nested_fragment_cases:
        sanitized = sanitize(payload)
        if marker in json.dumps(sanitized) or sanitize(sanitized) != sanitized:
            raise AssertionError("nested serialized argv fragment leaked or was unstable")
    print("rejected: nested serialized argv option fragments")

    structural_fragment_cases = []
    for nested_field in nested_fields:
        for fragment in fragments:
            payload = {nested_field: ["tool", fragment, marker]}
            structural_fragment_cases.extend((payload, json.dumps(payload)))
    if len(structural_fragment_cases) != 40:
        raise AssertionError("nested structural argv fragment matrix is incomplete")
    for payload in structural_fragment_cases:
        sanitized = sanitize(payload)
        if marker in json.dumps(sanitized) or sanitize(sanitized) != sanitized:
            raise AssertionError("nested structural argv fragment leaked or was unstable")
    print("rejected: nested structural argv option fragments")

    sibling_mapping_cases = []
    for sibling in ("raw", "value", "raw_value", "data"):
        payload = {"api_key": marker, sibling: marker}
        sibling_mapping_cases.extend((payload, json.dumps(payload)))
    if len(sibling_mapping_cases) != 8:
        raise AssertionError("sensitive mapping sibling matrix is incomplete")
    for payload in sibling_mapping_cases:
        sanitized = sanitize(payload)
        if marker in json.dumps(sanitized) or sanitize(sanitized) != sanitized:
            raise AssertionError("sensitive mapping sibling payload leaked")
    print("rejected: sensitive mapping sibling payloads")

    secret_gate = runpy.run_path(str(REPO / "scripts/verify_detect_secrets.py"))
    gate_row = {
        "filename": "fixture.txt",
        "type": "Secret Keyword",
        "hashed_secret": "synthetic-digest",
    }
    duplicate_document = {
        "results": {"fixture.txt": [gate_row, dict(gate_row)]},
    }
    duplicate_counts = secret_gate["identities"](duplicate_document)
    identity = ("fixture.txt", "Secret Keyword", "synthetic-digest")
    if duplicate_counts[identity] != 2:
        raise AssertionError("detect-secrets finding cardinality was collapsed")
    if secret_gate["scan_process_errors"](0, "", "synthetic warning") != [
        "detect-secrets wrote stderr"
    ]:
        raise AssertionError("detect-secrets stderr did not fail closed")
    print("rejected: duplicate detect-secrets finding and scanner stderr")

    extended_options = (
        "--database-url", "--db-url", "--mysql-pwd", "--proxy-authorization",
        "--azure-storage-account-key", "--docker-auth-config", "--user",
        "--proxy-user", "--oauth2-bearer", "-u",
        "--example-service-url", "--example-service-uri", "--example-service-dsn",
    )
    extended_argv = [
        sanitize({"probe_argv": ["tool", option, "argv-secret"]})
        for option in extended_options
    ] + [
        sanitize({
            "probe_argv": [
                "tool", urllib.parse.quote(option, safe=""), "encoded-argv-secret",
            ]
        })
        for option in extended_options
    ]
    extended_argv.append(sanitize({"probe_argv": ["curl", "-uattached-secret"]}))
    if any("secret" in json.dumps(item) for item in extended_argv):
        raise AssertionError("extended argv credential option was not redacted")
    print("rejected: extended raw and encoded argv credential options")

    contextual_names = (
        {"oauth2_bearer": "contextual-secret"},
        {"proxy_user": "contextual-secret"},
        "https://example.invalid/?proxy%5Fuser=contextual-secret",
        urllib.parse.quote(json.dumps({"oauth2_bearer": "contextual-secret"}), safe=""),
    )
    if any("secret" in json.dumps(sanitize(item)) for item in contextual_names):
        raise AssertionError("contextual CLI credential name was not redacted")
    print("rejected: contextual CLI credential names in mappings and serialized values")

    malformed_percent_names = (
        {"api%ZZ_key": "malformed-secret"},
        {"secr%ZZet": "malformed-secret"},
        "https://example.invalid/?secr%25ZZet=malformed-secret",
    )
    if any("secret" in json.dumps(sanitize(item)) for item in malformed_percent_names):
        raise AssertionError("malformed percent escape did not fail closed")
    print("rejected: malformed percent escapes in sensitive names")

    structured_argv = (
        {"probe_argv": ["tool", "--api-key", {"payload": "structured-secret"}]},
        {"probe_argv": ["tool", "--proxy-user", ["payload", "structured-secret"]]},
    )
    if any("secret" in json.dumps(sanitize(item)) for item in structured_argv):
        raise AssertionError("structured separate argv value was not redacted")
    print("rejected: structured separate argv credential values")

    nested = "structure-secret"
    for _ in range(1500):
        nested = [nested]
    bounded = sanitize(nested)
    cursor = bounded
    for _ in range(capture["STRUCTURE_SANITIZE_MAX_DEPTH"]):
        if not isinstance(cursor, list) or len(cursor) != 1:
            raise AssertionError("bounded structural redaction changed shape early")
        cursor = cursor[0]
    if cursor != redacted:
        raise AssertionError("deep structural recursion did not fail closed")
    print("rejected: direct structure beyond sanitizer depth limit")

    deeply_serialized = "[" * 1500 + '"deep-json-secret"' + "]" * 1500
    if sanitize(deeply_serialized) != redacted:
        raise AssertionError("deep serialized structure did not fail closed")
    print("rejected: serialized structure beyond parser recursion limit")

    scalar_cases = (
        ("encoded URI userinfo delimiter", "https://user%3Auserinfo-secret%40example.invalid/v1"),
        ("encoded URI query delimiter", "https://example.invalid/v1%3Fapi_key%3Dquery-secret"),
        ("encoded URI assignment delimiter", "https://example.invalid/v1?api_key%253Dencoded-query-secret"),
        ("export-prefixed environment assignment", "export PGPASSWORD=export-secret"),
        ("whitespace-separated export assignment", "export PGPASSWORD whitespace-secret"),
        ("env-prefixed environment assignment", "env PGPASSWORD=env-secret command"),
        ("declare-prefixed environment assignment", "declare -x PGPASSWORD=declare-secret"),
        ("readonly-prefixed environment assignment", "readonly PGPASSWORD=readonly-secret"),
        ("setenv environment assignment", "setenv API_KEY setenv-secret"),
        ("printf environment assignment", "printf -v API_KEY %s printf-secret"),
        ("curl user credential option", "curl -u curl-secret https://example.invalid/"),
        ("curl proxy-user credential option", "curl --proxy-user proxy-user-secret https://example.invalid/"),
        ("curl attached user credential option", "curl -uattached-secret https://example.invalid/"),
        ("embedded bearer header", "prefix Authorization: Bearer bearer-secret suffix"),
        ("embedded basic header", "prefix Proxy-Authorization: Basic basic-secret suffix"),
        ("percent-encoded serialized JSON", urllib.parse.quote(json.dumps({"X-Api-Key": "json-secret"}), safe="")),
    )
    for name, payload in scalar_cases:
        sanitized = sanitize(payload)
        if "secret" in sanitized or redacted not in sanitized:
            raise AssertionError(f"sanitizer negative control failed: {name}")
        print(f"rejected: {name}")

    too_deep = "https://example.invalid/v1?api_key=deep-secret"
    for _ in range(capture["JSON_SANITIZE_MAX_DEPTH"] + 2):
        too_deep = urllib.parse.quote(too_deep, safe="")
    if sanitize(too_deep) != redacted:
        raise AssertionError("non-convergent percent encoding was not rejected")
    print("rejected: percent encoding beyond the sanitizer depth limit")

    safe_encoded_url = "https://example.invalid/a%20safe%20path?mode=readonly"
    if sanitize(safe_encoded_url) != safe_encoded_url:
        raise AssertionError("safe encoded path/URL was not preserved")


def main() -> int:
    sanitizer_negative_controls()
    rejected_with_refreshed_binding(
        "README runtime completion shifted with refreshed binding",
        lambda root: (root / "README.md").write_text(
            re.sub(
                r"^- Runtime capture completed: `[^`]+`$",
                "- Runtime capture completed: `2026-09-10T21:01:18.579189+02:00`",
                (root / "README.md").read_text(),
                count=1,
                flags=re.MULTILINE,
            )
        ),
        "README runtime capture completion",
        rel="README.md",
    )
    command_honest_refresh_control(
        "recipe video limit changed 0 to 1 with all bindings refreshed",
        lambda command: command.replace('"video":0', '"video":1'),
        "192.168.178.47 serving PID/command binding",
    )
    command_honest_refresh_control(
        "recipe executable flag added with all bindings refreshed",
        lambda command: command.replace("--skip-mm-profiling", "--skip-mm-profiling --enforce-eager"),
        "192.168.178.47 serving PID/command binding",
    )
    command_honest_refresh_control(
        "recipe executable token added with all bindings refreshed",
        lambda command: command.replace("--skip-mm-profiling", "--skip-mm-profiling unexpected-token"),
        "192.168.178.47 serving PID/command binding",
    )
    command_honest_refresh_control(
        "recipe shell statement added with all bindings refreshed",
        lambda command: command.rstrip() + "; echo injected\n",
        "reviewed recipe command contract: recipe command contains more than one executable",
    )
    executable_mod_substitution_control()
    head = "192.168.178.47"
    worker = "192.168.178.46"
    for name, host, rank, old, new in (
        ("wrapper port mutated with refreshed binding", head, 0, "--port 8000", "--port 8001"),
        ("wrapper flag removed with refreshed binding", head, 0, " --gpu-memory-utilization 0.85", ""),
        ("worker rank mutated with refreshed binding", worker, 1, "--node-rank 1", "--node-rank 0"),
    ):
        rejected_with_refreshed_binding(
            name,
            lambda root, target=host, before=old, after=new: mutate_json(
                root / "runtime.json",
                lambda data: data["hosts"][target]["docker_top"].__setitem__(
                    "stdout",
                    data["hosts"][target]["docker_top"]["stdout"].replace(
                        before, after, 1
                    ),
                ),
            ),
            f"{host} docker top closed process inventory",
            rel="runtime.json",
            prepare=lambda root, target=host, node_rank=rank: install_expected_top_inventory(
                root, target, node_rank
            ),
        )
    for name, host, rank, old, new in (
        ("vLLM port mutated with refreshed binding", head, 0, "--port 8000", "--port 8001"),
        ("vLLM flag removed with refreshed binding", head, 0, " --gpu-memory-utilization 0.85", ""),
        ("vLLM worker rank mutated with refreshed binding", worker, 1, "--node-rank 1", "--node-rank 0"),
    ):
        rejected_with_refreshed_binding(
            name,
            lambda root, target=host, before=old, after=new: mutate_named_top_command(
                root, target, "/usr/local/bin/vllm serve ", before, after
            ),
            f"{host} docker top closed process inventory",
            rel="runtime.json",
            prepare=lambda root, target=host, node_rank=rank: install_expected_top_inventory(
                root, target, node_rank
            ),
        )
    for name, host, rank, marker in (
        ("head vLLM PID substituted with refreshed binding", head, 0, "/usr/local/bin/vllm serve "),
        ("worker vLLM PID substituted with refreshed binding", worker, 1, "/usr/local/bin/vllm serve "),
        ("head wrapper PID substituted with refreshed binding", head, 0, "serve_wrapper.sh "),
        ("worker wrapper PID substituted with refreshed binding", worker, 1, "serve_wrapper.sh "),
    ):
        rejected_with_refreshed_binding(
            name,
            lambda root, target=host, command_marker=marker: mutate_named_top_pid(
                root, target, command_marker, 999999
            ),
            f"{host} docker top process identity join",
            rel="runtime.json",
            prepare=lambda root, target=host, node_rank=rank: install_expected_top_inventory(
                root, target, node_rank
            ),
        )
    for host, rank in ((head, 0), (worker, 1)):
        rejected_with_refreshed_binding(
            f"rank {rank} PID namespace mapping substituted with refreshed binding",
            lambda root, target=host: mutate_json(
                root / "runtime.json",
                lambda data: data["hosts"][target]["pid_namespace"].__setitem__(
                    "stdout", "host_pid=999999\ncontainer_pid=999999\n"
                ),
            ),
            f"{host} serving PID namespace join",
            rel="runtime.json",
            prepare=lambda root, target=host, node_rank=rank: install_expected_top_inventory(
                root, target, node_rank
            ),
        )
        for name, marker, column in (
            ("launcher shim", "c2xlZXAgaW5maW5pdHk=", "ppid"),
            ("launcher process group", "c2xlZXAgaW5maW5pdHk=", "pgid"),
            ("container shell parent", r"^bash --noprofile --norc$", "ppid"),
            ("keepalive parent", "sleep infinity", "ppid"),
            ("watchdog shim", "SERVE_PID=$(cat /tmp/sparkrun_serve.pid)", "ppid"),
            ("watchdog process group", "SERVE_PID=$(cat /tmp/sparkrun_serve.pid)", "pgid"),
            ("resource tracker parent", "multiprocessing.resource_tracker", "ppid"),
            (
                "rank worker parent",
                f"VLLM::Worker_TP{rank}",
                "ppid",
            ),
            (
                "rank worker process group",
                f"VLLM::Worker_TP{rank}",
                "pgid",
            ),
            ("watchdog sleep parent", r"^sleep 5$", "ppid"),
        ) + (
            (("engine parent", "VLLM::EngineCore", "ppid"),)
            if rank == 0
            else ()
        ):
            rejected_with_refreshed_binding(
                f"rank {rank} {name} substituted with refreshed binding",
                lambda root, target=host, fragment=marker, field=column: mutate_top_process_relation(
                    root, target, fragment, field, 999999
                ),
                f"{host} docker top process identity join",
                rel="runtime.json",
                prepare=lambda root, target=host, node_rank=rank: install_expected_top_inventory(
                    root, target, node_rank
                ),
            )
        for field, value in (
            ("wrapper_host_pid", 999999),
            ("parent_host_pid", 0),
            ("parent_executable", "/tmp/shim"),
            ("parent_namespace", "other"),
            ("parent_container_id", "f" * 64),
            ("parent_address", "/tmp/containerd.sock"),
        ):
            rejected_with_refreshed_binding(
                f"rank {rank} wrapper lineage {field} substituted with refreshed binding",
                lambda root, target=host, key=field, replacement=value: mutate_wrapper_lineage_field(
                    root, target, key, replacement
                ),
                f"{host} wrapper shim and PID namespace join",
                rel="runtime.json",
                prepare=lambda root, target=host, node_rank=rank: install_expected_top_inventory(
                    root, target, node_rank
                ),
            )
        for kind in ("missing", "fatal", "tier"):
            rejected_with_refreshed_binding(
                f"rank {rank} structured serve marker {kind} substituted with refreshed binding",
                lambda root, target=host, mutation=kind: mutate_serve_markers(
                    root, target, mutation
                ),
                f"{host} structured E3 runtime execution marker and fatal scan",
                rel="runtime.json",
                prepare=lambda root, target=host, node_rank=rank: install_expected_top_inventory(
                    root, target, node_rank
                ),
            )
    for name, field, value, expected in (
        ("acceptance run ID substituted with refreshed binding", "run_id", "ffffffffffffffff", "acceptance exact run ID"),
        ("acceptance schema substituted with refreshed binding", "schema", 2, "acceptance schema/process/verdict"),
        ("acceptance process role substituted with refreshed binding", "process_role", "draft", "acceptance schema/process/verdict"),
        ("acceptance passed verdict substituted with refreshed binding", "passed", False, "acceptance schema/process/verdict"),
    ):
        rejected_with_refreshed_binding(
            name,
            lambda root, key=field, replacement=value: mutate_json(
                root / "acceptance.json",
                lambda data: data.__setitem__(key, replacement),
            ),
            expected,
            rel="acceptance.json",
            prepare=install_finalized_metadata_contract,
        )
    for name, field, value, expected in (
        ("runtime schema substituted with refreshed binding", "schema", 2, "runtime schema/process/producer"),
        ("runtime process role substituted with refreshed binding", "process_role", "draft", "runtime schema/process/producer"),
        ("runtime producer substituted with refreshed binding", "producer", "other.py", "runtime schema/process/producer"),
        ("runtime capture argv substituted with refreshed binding", "capture_argv", ["capture_runtime.py"], "runtime capture argv"),
    ):
        rejected_with_refreshed_binding(
            name,
            lambda root, key=field, replacement=value: mutate_json(
                root / "runtime.json",
                lambda data: data.__setitem__(key, replacement),
            ),
            expected,
            rel="runtime.json",
            prepare=install_finalized_metadata_contract,
        )
    rejected_with_refreshed_binding(
        "response fingerprint substituted with refreshed binding",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_exact"]["response"].__setitem__(
                "system_fingerprint", "forged-runtime"
            ),
        ),
        "direct_exact system fingerprint",
        rel="acceptance.json",
    )
    rejected_with_refreshed_binding(
        "response created timestamp zeroed with refreshed binding",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_exact"]["response"].__setitem__(
                "created", 0
            ),
        ),
        "direct_exact response created interval",
        rel="acceptance.json",
    )
    rejected_with_refreshed_binding(
        "contradictory successful receipt verdict false with refreshed binding",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_exact"].__setitem__("passed", False),
        ),
        "direct_exact verdict",
        rel="acceptance.json",
    )
    for mode in ("omitted", "null", "nonempty", "relabelled"):
        rejected_with_refreshed_binding(
            f"worker negative listener receipt {mode} with refreshed binding",
            lambda root, mutation=mode: mutate_worker_listener(root, mutation),
            "worker negative listener receipt",
            rel="runtime.json",
            prepare=lambda root: install_expected_top_inventory(root, worker, 1),
        )
    for name, command in (
        ("unknown worker process added with refreshed binding", "/usr/bin/nc -l -p 9999"),
        ("extra worker listener process added with refreshed binding", "python3 -m http.server 9999"),
    ):
        rejected_with_refreshed_binding(
            name,
            lambda root, added=command: mutate_json(
                root / "runtime.json",
                lambda data: data["hosts"][worker]["docker_top"].__setitem__(
                    "stdout",
                    data["hosts"][worker]["docker_top"]["stdout"]
                    + f"999999              {added}\n",
                ),
            ),
            f"{worker} docker top closed process inventory",
            rel="runtime.json",
            prepare=lambda root: install_expected_top_inventory(root, worker, 1),
        )
    run_exact_runtime_command_controls()
    for host in ("192.168.178.47", "192.168.178.46"):
        rejected(
            f"per-rank runtime mod digest substituted on {host}",
            lambda root, target=host: mutate_json(
                root / "runtime.json",
                lambda data: data["hosts"][target]["runtime_mod_manifest"].__setitem__(
                    "stdout",
                    data["hosts"][target]["runtime_mod_manifest"]["stdout"].replace(
                        data["expected_mod_manifest_sha256"], "0" * 64, 1
                    ),
                ),
            ),
            f"{host} complete runtime mod manifest verification",
            prepare=install_runtime_mod_manifest_receipts,
        )
    rejected(
        "launch receipt mod digest substituted with refreshed manifest",
        lambda root: mutate_json(
            root / "launch-receipt.json",
            lambda data: data.__setitem__("mod_manifest_sha256", "0" * 64),
        ),
        "launch mod manifest binding",
        prepare=install_launch_receipt,
    )
    rejected(
        "runtime overlay verifier success marker removed with refreshed manifest",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["hosts"]["192.168.178.47"]["runtime_overlay"].__setitem__(
                "stdout",
                data["hosts"]["192.168.178.47"]["runtime_overlay"]["stdout"].replace(
                    "[OK] complete GLM-5.3 runtime patched state verified", "[removed]"
                ),
            ),
        ),
        "192.168.178.47 runtime patch live gate",
        prepare=install_runtime_overlay_contract,
    )
    rejected(
        "runtime overlay FATAL stderr added with refreshed manifest",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["hosts"]["192.168.178.47"]["runtime_overlay"].__setitem__(
                "stderr", "FATAL: forced verifier failure\n"
            ),
        ),
        "192.168.178.47 runtime overlay FATAL stderr",
        prepare=install_runtime_overlay_contract,
    )
    rejected(
        "runtime NCCL HCA selection substituted with refreshed manifest",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["hosts"]["192.168.178.47"]["runtime_env"].__setitem__(
                "stdout",
                data["hosts"]["192.168.178.47"]["runtime_env"]["stdout"].replace(
                    "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1",
                    "NCCL_IB_HCA=fabricated0,fabricated1",
                ),
            ),
        ),
        "192.168.178.47 runtime NCCL_IB_HCA",
        prepare=install_roce_records,
    )
    rejected(
        "runtime NCCL GID index substituted with refreshed manifest",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["hosts"]["192.168.178.47"]["runtime_env"].__setitem__(
                "stdout",
                data["hosts"]["192.168.178.47"]["runtime_env"]["stdout"].replace(
                    "NCCL_IB_GID_INDEX=3", "NCCL_IB_GID_INDEX=4"
                ),
            ),
        ),
        "192.168.178.47 runtime NCCL_IB_GID_INDEX",
        prepare=install_roce_records,
    )
    rejected(
        "runtime NCCL socket interfaces substituted with refreshed manifest",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["hosts"]["192.168.178.47"]["runtime_env"].__setitem__(
                "stdout",
                data["hosts"]["192.168.178.47"]["runtime_env"]["stdout"].replace(
                    "NCCL_SOCKET_IFNAME=enP7s7,enp1s0f1np1,enP2p1s0f1np1",
                    "NCCL_SOCKET_IFNAME=enP7s7,enp1s0f1np1",
                ),
            ),
        ),
        "192.168.178.47 runtime NCCL_SOCKET_IFNAME",
        prepare=install_roce_records,
    )
    rejected(
        "runtime RoCE netdev record substituted with refreshed manifest",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["hosts"]["192.168.178.47"]["roce_records"].__setitem__(
                "stdout",
                data["hosts"]["192.168.178.47"]["roce_records"]["stdout"].replace(
                    "enp1s0f1np1", "fabricated0"
                ),
            ),
        ),
        "192.168.178.47 runtime RoCE records",
        prepare=install_roce_records,
    )
    rejected(
        "telemetry HCA names substituted with refreshed manifest",
        lambda root: replace_telemetry_hca(root, "rocep1s0f1", "fabricated0"),
        "telemetry grammar",
        prepare=install_roce_records,
    )
    rejected(
        "late launch epoch shifted self-consistently with refreshed manifest",
        shift_launch_epoch_self_consistently,
        "launch epoch precedes Docker Created",
        prepare=install_launch_receipt,
    )
    rejected(
        "acceptance invocation argv substituted with refreshed manifest",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["invocation_argv"].append("--forged"),
        ),
        "acceptance invocation argv",
        prepare=install_acceptance_routes,
    )
    rejected(
        "direct requested URL rebound to proxy port with refreshed manifest",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_exact"].__setitem__(
                "requested_url", "http://127.0.0.1:4000/v1/chat/completions"
            ),
        ),
        "direct_exact requested/effective URL/path",
        prepare=install_acceptance_routes,
    )
    rejected(
        "direct receipt rebound to proxy port with refreshed manifest",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_exact"].__setitem__(
                "effective_url", "http://127.0.0.1:4000/v1/chat/completions"
            ),
        ),
        "direct_exact requested/effective URL/path",
        prepare=install_acceptance_routes,
    )
    rejected(
        "proxy receipt rebound to direct port with refreshed manifest",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["proxy_exact"].__setitem__(
                "effective_url", "http://127.0.0.1:8000/v1/chat/completions"
            ),
        ),
        "proxy_exact requested/effective URL/path",
        prepare=install_acceptance_routes,
    )
    for host in ("192.168.178.47", "192.168.178.46"):
        rejected(
            f"during telemetry host interval moved outside canonical request: {host}",
            lambda root, target=host: shift_during_host_interval(root, target),
            f"during host acquisition overlap {host}",
            prepare=install_telemetry_intervals,
        )
    rejected(
        "proxy registry model provenance substituted",
        lambda root: mutate_json(
            root / "proxy-models.json",
            lambda data: data[0].__setitem__("model_name", "forged-model"),
        ),
        "proxy registry receipt",
    )
    rejected(
        "proxy status bind substituted",
        lambda root: (root / "proxy-start.log").write_text(
            (root / "proxy-start.log").read_text().replace("Host:    0.0.0.0", "Host:    127.0.0.1")
        ),
        "proxy status receipt",
    )
    rejected(
        "proxy status PID substituted",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["proxy_status"].__setitem__(
                "stdout", data["proxy_status"]["stdout"].replace("PID:     760553", "PID:     999999")
            ),
        ),
        "proxy process identity",
        prepare=install_listener_receipts,
    )
    rejected(
        "proxy status autodiscovery supervisor PID substituted",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["proxy_status"].__setitem__(
                "stdout",
                data["proxy_status"]["stdout"].replace(
                    "Auto-discover: running (PID 760552)",
                    "Auto-discover: running (PID 999998)",
                ),
            ),
        ),
        "proxy process identity",
        prepare=install_listener_receipts,
    )
    rejected(
        "proxy listener identity substituted",
        lambda root: mutate_listener(root, "proxy_listener", "pid", 999999),
        "proxy listener identity",
        prepare=install_listener_receipts,
    )
    rejected(
        "stale extra LiteLLM descendant",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["proxy_processes"].append(
                dict(data["proxy_processes"][2], pid=999999, started_at=99.0)
            ),
        ),
        "proxy process identity",
        prepare=install_listener_receipts,
    )
    rejected(
        "stale extra autodiscovery supervisor",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["proxy_processes"].append(
                dict(data["proxy_processes"][0], pid=999998, started_at=98.0)
            ),
        ),
        "proxy process identity",
        prepare=install_listener_receipts,
    )
    rejected(
        "proxy upstream socket removed",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data.__setitem__("proxy_upstream_sockets", []),
        ),
        "proxy established upstream sockets",
        prepare=install_listener_receipts,
    )
    rejected(
        "proxy listener bind mutated",
        lambda root: mutate_listener(root, "proxy_listener", "bind", "127.0.0.1"),
        "proxy listener identity",
        prepare=install_listener_receipts,
    )
    rejected(
        "direct listener owner PID differs from serving PID",
        lambda root: mutate_direct_listener(root, "pid", 999999),
        "head direct listener serving PID",
        prepare=install_listener_receipts,
    )
    rejected(
        "direct listener endpoint mutated",
        lambda root: mutate_direct_listener(root, "port", 18000),
        "head direct listener endpoint",
        prepare=install_listener_receipts,
    )
    rejected(
        "direct listener capture command substituted",
        lambda root: substitute_remote_command(
            root, "192.168.178.47", "direct_listener", "'true'"
        ),
        "head direct listener command",
        prepare=install_listener_receipts,
    )
    rejected(
        "container top-level image config digest substituted",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: mutate_inspect(
                data,
                "192.168.178.47",
                lambda inspect: inspect.__setitem__("Image", "sha256:" + "0" * 64),
            ),
        ),
        "192.168.178.47 image config digest",
    )
    rejected(
        "extra root bind and mount added",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: mutate_inspect(data, "192.168.178.47", add_root_mount),
        ),
        "192.168.178.47 exact binds",
    )
    rejected(
        "docker image inspect config Id substituted",
        lambda root: mutate_image_inspect(
            root, lambda image: image.__setitem__("Id", "sha256:" + "0" * 64)
        ),
        "192.168.178.47 image inspect config digest",
        prepare=install_image_inspect_receipt,
    )
    rejected(
        "docker image inspect RepoDigest substituted",
        lambda root: mutate_image_inspect(
            root, lambda image: image.__setitem__("RepoDigests", ["forged.invalid/image@sha256:" + "0" * 64])
        ),
        "192.168.178.47 image inspect RepoDigests",
        prepare=install_image_inspect_receipt,
    )
    rejected(
        "docker image inspect command substituted",
        lambda root: substitute_remote_command(
            root, "192.168.178.47", "image_inspect", "'true'"
        ),
        "192.168.178.47 image_inspect exact command",
        prepare=install_image_inspect_receipt,
    )
    rejected(
        "fabricated HCA names retain plausible rates and states",
        lambda root: mutate_json(
            root / "runtime.json",
            lambda data: data["hosts"]["192.168.178.47"]["rdma"].__setitem__(
                "stdout",
                data["hosts"]["192.168.178.47"]["rdma"]["stdout"]
                .replace("rocep1s0f1", "fabricated0")
                .replace("roceP2p1s0f1", "fabricated1"),
            ),
        ),
        "192.168.178.47 exact HCA map",
    )
    rejected(
        "runtime launch epoch replaced with zero",
        lambda root: mutate_json(
            root / "runtime.json", lambda data: data.__setitem__("launch_epoch", 0)
        ),
        "runtime launch epoch binding",
    )
    rejected(
        "forged direct semantic output with refreshed manifest",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data["checks"]["direct_exact"]["response"]["choices"][0]["message"].__setitem__("content", "FORGED")),
        "direct exact completion",
    )
    rejected(
        "non-overlapping C4 timestamps with refreshed manifest",
        lambda root: mutate_json(root / "acceptance.json", lambda data: [row.__setitem__("started_at", row["completed_at"] + index + 1) for index, row in enumerate(data["checks"]["direct_concurrency_c4"]["results"])]),
        "C4 requests did not overlap",
    )
    rejected(
        "forged long-context token count with refreshed manifest",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data["checks"]["direct_long_context"]["response"]["usage"].__setitem__("prompt_tokens", 1)),
        "long-context prompt token count",
    )
    rejected(
        "forged telemetry-load token count with refreshed manifest",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data["checks"]["direct_telemetry_load"]["response"]["usage"].__setitem__("completion_tokens", 1)),
        "telemetry load completion tokens",
    )
    rejected(
        "telemetry host acquisitions outside direct load receipt",
        lambda root: mutate_json(root / "acceptance.json", move_load_receipt_outside_during),
        "during host acquisition overlap 192.168.178.47",
        prepare=install_telemetry_intervals,
    )
    rejected(
        "substituted worker image with refreshed manifest",
        lambda root: mutate_json(root / "runtime.json", mutate_worker_image),
        "192.168.178.46 image pin",
    )
    rejected(
        "post-acceptance kernel Xid with refreshed manifest",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["hosts"]["192.168.178.47"]["kernel_after_acceptance"].__setitem__("stdout", "NVRM: Xid 79\n")),
        "192.168.178.47 post-acceptance kernel safety",
    )
    rejected(
        "forged HCA traffic delta with refreshed manifest",
        forge_telemetry,
        "192.168.178.47 rocep1s0f1 traffic delta",
        prepare=install_telemetry_intervals,
    )
    rejected(
        "deleted proxy receipt with refreshed manifest",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data["checks"].pop("proxy_exact")),
        "acceptance check set",
        prepare=install_current_acceptance_check_set,
    )
    rejected(
        "resolved --root controls checksum reads",
        lambda root: (root / "static-validation.log").write_text("forged\n"),
        "manifest static-validation.log",
        refresh_manifest=False,
    )
    rejected(
        "required artifact deleted with its manifest row",
        lambda root: (root / "launch.log").unlink(),
        "manifest exact file set",
    )
    rejected(
        "canonical request substituted with self-consistent hash",
        lambda root: mutate_json(root / "acceptance.json", substitute_direct_request),
        "direct_exact canonical request",
    )
    rejected(
        "canonical request hash replaced",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data["checks"]["direct_exact"].__setitem__("request_sha256", "0" * 64)),
        "direct_exact request SHA",
    )
    rejected(
        "fixture and both inline requests replaced self-consistently",
        replace_fixture_everywhere,
        "canonical vision fixture bytes",
    )
    rejected(
        "OCR fixture and both inline requests replaced self-consistently",
        replace_ocr_fixture_everywhere,
        "canonical OCR fixture bytes",
        prepare=install_ocr_receipts,
    )
    rejected(
        "direct OCR canonical request substituted self-consistently",
        substitute_direct_ocr_request,
        "direct_ocr canonical request",
        prepare=install_ocr_receipts,
    )
    for route in ("direct", "proxy"):
        key = f"{route}_video_rejected"
        rejected(
            f"{route} video rejection changed to success",
            lambda root, key=key: mutate_json(
                root / "acceptance.json",
                lambda data: (
                    data["checks"][key].__setitem__("http", 200),
                    data["checks"][key].__setitem__("response", {"id": "success"}),
                ),
            ),
            f"{key} HTTP status",
            prepare=install_video_receipts,
        )
        rejected(
            f"{route} video request substituted self-consistently",
            lambda root, key=key: mutate_json(
                root / "acceptance.json",
                lambda data: (
                    data["checks"][key]["request"]["messages"][0]["content"][0].__setitem__(
                        "text", "mutated-video-request"
                    ),
                    data["checks"][key].__setitem__(
                        "request_sha256",
                        request_sha(data["checks"][key]["request"]),
                    ),
                ),
            ),
            f"{key} canonical request",
            prepare=install_video_receipts,
        )
    rejected(
        "video fixture and both inline rejection requests replaced self-consistently",
        replace_video_fixture_everywhere,
        "canonical video fixture bytes",
        prepare=install_video_receipts,
    )
    for key in ("direct_video_rejected", "proxy_video_rejected"):
        rejected_with_refreshed_binding(
            f"{key} rejection changed to success with refreshed binding",
            lambda root, target=key: mutate_json(
                root / "acceptance.json",
                lambda data: data["checks"][target].update({
                    "http": 200,
                    "response": {"model": "GLM-5.3-Flash-EXL3"},
                }),
            ),
            f"{key} HTTP status",
            rel="acceptance.json",
            prepare=install_video_receipts,
        )
        rejected_with_refreshed_binding(
            f"{key} request substituted with refreshed binding",
            lambda root, target=key: mutate_video_request(root, target),
            f"{key} canonical request",
            rel="acceptance.json",
            prepare=install_video_receipts,
        )
    rejected(
        "successful receipt changed to HTTP error",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data["checks"]["direct_exact"].__setitem__("http", 500)),
        "direct_exact HTTP status",
    )
    rejected(
        "response model substituted",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data["checks"]["direct_exact"]["response"].__setitem__("model", "forged-model")),
        "direct_exact response model",
    )
    rejected(
        "tool arguments substituted",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data["checks"]["direct_tool_call"]["response"]["choices"][0]["message"]["tool_calls"][0]["function"].__setitem__("arguments", '{"city":"London"}')),
        "exact tool arguments",
    )
    rejected(
        "reasoning-open response stops at client string",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_reasoning_open_stop"]["response"]["choices"][0].__setitem__("stop_reason", "144"),
        ),
        "reasoning-open exact stop reason",
        prepare=install_stop_receipts,
    )
    rejected(
        "reasoning-open final answer removed",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_reasoning_open_stop"]["response"]["choices"][0]["message"].__setitem__("content", ""),
        ),
        "reasoning-open exact final answer",
        prepare=install_stop_receipts,
    )
    rejected(
        "reasoning-open final answer prefixed",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_reasoning_open_stop"]["response"][
                "choices"
            ][0]["message"].__setitem__(
                "content", "prefix GLM53_REASONING_STOP_OK"
            ),
        ),
        "reasoning-open exact final answer",
        prepare=install_stop_receipts,
    )
    rejected(
        "reasoning-open finish reason changed to length",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_reasoning_open_stop"]["response"][
                "choices"
            ][0].__setitem__("finish_reason", "length"),
        ),
        "reasoning-open finish reason",
        prepare=install_stop_receipts,
    )
    rejected_with_refreshed_binding(
        "reasoning-open stop reason forged with refreshed binding and manifest",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_reasoning_open_stop"]["response"][
                "choices"
            ][0].__setitem__("stop_reason", 999999),
        ),
        "reasoning-open exact stop reason",
        rel="acceptance.json",
        prepare=install_stop_receipts,
    )
    rejected_with_refreshed_binding(
        "thinking-disabled finish reason changed to length with refreshed binding and manifest",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_thinking_disabled_stop"]["response"][
                "choices"
            ][0].__setitem__("finish_reason", "length"),
        ),
        "thinking-disabled finish reason",
        rel="acceptance.json",
        prepare=install_stop_receipts,
    )
    rejected(
        "thinking-disabled request ignores client stop",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_thinking_disabled_stop"]["response"]["choices"][0].update({"stop_reason": 154827, "message": {"reasoning": None, "content": "BEFORE Question: AFTER"}}),
        ),
        "thinking-disabled stop truncated output",
        prepare=install_stop_receipts,
    )
    rejected(
        "thinking-disabled stop reason substituted",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_thinking_disabled_stop"]["response"]["choices"][0].__setitem__("stop_reason", 154827),
        ),
        "thinking-disabled client stop reason",
        prepare=install_stop_receipts,
    )
    rejected(
        "direct route substituted",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data.__setitem__("direct_url", "http://attacker.invalid:8000")),
        "acceptance direct route",
    )
    rejected(
        "remote media rejection changed to success",
        lambda root: mutate_json(root / "acceptance.json", lambda data: data["checks"]["direct_remote_media_rejected"].__setitem__("http", 200)),
        "direct_remote_media_rejected HTTP status",
    )
    rejected(
        "proxy remote media rejection changed to success",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["proxy_remote_media_rejected"].__setitem__(
                "http", 200
            ),
        ),
        "proxy_remote_media_rejected HTTP status",
        prepare=install_proxy_remote_media_receipt,
    )
    rejected(
        "proxy remote media allowlist error substituted",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["proxy_remote_media_rejected"]["response"][
                "error"
            ].__setitem__("message", "unrelated bad request"),
        ),
        "proxy_remote_media_rejected exact allowlist error",
        prepare=install_proxy_remote_media_receipt,
    )
    rejected(
        "remote media rejection request substituted self-consistently",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: (
                data["checks"]["direct_remote_media_rejected"]["request"]["messages"][0]["content"][1]["image_url"].__setitem__("url", "http://media.invalid/image.png"),
                data["checks"]["direct_remote_media_rejected"].__setitem__("request_sha256", request_sha(data["checks"]["direct_remote_media_rejected"]["request"])),
            ),
        ),
        "direct_remote_media_rejected canonical request",
    )
    for field, value in (
        ("message", "unrelated bad request"),
        ("type", "ValidationError"),
        ("param", "image_url"),
        ("code", 401),
    ):
        rejected(
            f"remote media allowlist error {field} substituted",
            lambda root, key=field, replacement=value: mutate_json(
                root / "acceptance.json",
                lambda data: data["checks"]["direct_remote_media_rejected"]["response"]["error"].__setitem__(key, replacement),
            ),
            "direct_remote_media_rejected exact allowlist error",
        )
    rejected(
        "stopped container accepted",
        lambda root: mutate_json(root / "runtime.json", lambda data: mutate_inspect(data, "192.168.178.46", lambda inspect: inspect["State"].update({"Running": False, "Status": "exited"}))),
        "192.168.178.46 container running",
    )
    rejected(
        "exited SparkRun container accepted",
        lambda root: mutate_json(root / "runtime.json", mutate_sparkrun_status),
        "SparkRun containers running",
    )
    for name, mutator, expected in (
        ("container user", lambda inspect: inspect["Config"].__setitem__("User", "nobody"), "container user"),
        ("privileged mode", lambda inspect: inspect["HostConfig"].__setitem__("Privileged", True), "privileged mode"),
        ("network mode", lambda inspect: inspect["HostConfig"].__setitem__("NetworkMode", "bridge"), "network mode"),
        ("IPC mode", lambda inspect: inspect["HostConfig"].__setitem__("IpcMode", "private"), "IPC mode"),
        ("security options", lambda inspect: inspect["HostConfig"].__setitem__("SecurityOpt", []), "security options"),
        ("capabilities", lambda inspect: inspect["HostConfig"].__setitem__("CapAdd", []), "capabilities"),
        ("RDMA devices", lambda inspect: inspect["HostConfig"].__setitem__("Devices", []), "RDMA device"),
    ):
        rejected(
            f"mutated {name}",
            lambda root, mutate=mutator: mutate_json(root / "runtime.json", lambda data: mutate_inspect(data, "192.168.178.47", mutate)),
            f"192.168.178.47 {expected}",
        )
    rejected(
        "worker SSH record relabeled as head",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["hosts"]["192.168.178.46"]["container_discovery"]["argv"].__setitem__(3, "192.168.178.47")),
        "192.168.178.46 container_discovery exact command",
    )
    for name, mutate in (
        ("host", lambda row: row["argv"].__setitem__(3, "192.168.178.47")),
        ("container", lambda row: row["argv"].__setitem__(-1, row["argv"][-1].replace("_node_1", "_node_0"))),
        ("command", lambda row: row["argv"].__setitem__(-1, row["argv"][-1].replace("verify_runtime_patch_state.py", "true"))),
        ("returncode", lambda row: row.__setitem__("returncode", 9)),
    ):
        rejected(
            f"runtime overlay {name} substituted",
            lambda root, mutator=mutate: mutate_json(
                root / "runtime.json",
                lambda data: mutator(data["hosts"]["192.168.178.46"]["runtime_overlay"]),
            ),
            "192.168.178.46 runtime_overlay exact command",
            prepare=install_runtime_overlay_contract,
        )
    for name, key, mutate in (
        ("host", "postready_rc", lambda row: row["argv"].__setitem__(3, "192.168.178.46")),
        ("container", "postready_ok", lambda row: row["argv"].__setitem__(-1, row["argv"][-1].replace("_node_0", "_node_1"))),
        ("command", "postready_log", lambda row: row["argv"].__setitem__(-1, row["argv"][-1].replace("glm53-postready.log", "other.log"))),
        ("returncode", "postready_rc", lambda row: row.__setitem__("returncode", 7)),
    ):
        rejected(
            f"postready {name} substituted",
            lambda root, receipt=key, mutator=mutate: mutate_json(
                root / "runtime.json",
                lambda data: mutator(data["hosts"]["192.168.178.47"][receipt]),
            ),
            f"post-ready {key} exact command",
        )
    rejected(
        "worker runtime rank relabeled",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["hosts"]["192.168.178.46"]["docker_top"].__setitem__("stdout", data["hosts"]["192.168.178.46"]["docker_top"]["stdout"].replace("--node-rank 1", "--node-rank 0"))),
        "192.168.178.46 docker top closed process inventory",
        prepare=lambda root: install_expected_top_inventory(root, "192.168.178.46", 1),
    )
    rejected(
        "worker host/rank metadata relabeled",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["hosts"]["192.168.178.46"].__setitem__("rank", 0)),
        "192.168.178.46 host/rank metadata",
        prepare=lambda root: install_bound_runtime_record(root, "192.168.178.46", 1),
    )
    rejected(
        "worker container metadata relabeled",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["hosts"]["192.168.178.46"].__setitem__("container", "forged-container")),
        "192.168.178.46 expected container",
        prepare=lambda root: install_bound_runtime_record(root, "192.168.178.46", 1),
    )
    rejected(
        "serving PID relabeled independently of NCCL maps",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["hosts"]["192.168.178.46"]["serving_process"].__setitem__("stdout", re.sub(r"pid=\d+", "pid=999", data["hosts"]["192.168.178.46"]["serving_process"]["stdout"], count=1))),
        "192.168.178.46 NCCL serving PID",
        prepare=lambda root: install_bound_runtime_record(root, "192.168.178.46", 1),
    )
    rejected(
        "secure SparkRun command loses media allowlist",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["status"].__setitem__("stdout", data["status"]["stdout"].replace(" --allowed-media-domains media.invalid", ""))),
        "reviewed recipe/SparkRun/launch command binding",
        prepare=lambda root: install_command_chain(root, RECIPE),
    )
    rejected(
        "runtime process loses media allowlist",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["hosts"]["192.168.178.46"]["docker_top"].__setitem__("stdout", data["hosts"]["192.168.178.46"]["docker_top"]["stdout"].replace(" --allowed-media-domains media.invalid", ""))),
        "192.168.178.46 docker top closed process inventory",
        prepare=lambda root: install_expected_top_inventory(root, "192.168.178.46", 1),
    )
    rejected(
        "NCCL path substituted",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["hosts"]["192.168.178.47"]["nccl_loaded"].__setitem__("stdout", data["hosts"]["192.168.178.47"]["nccl_loaded"]["stdout"].replace("/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2", "/tmp/libnccl.so.2"))),
        "192.168.178.47 canonical NCCL path",
    )
    rejected(
        "NCCL digest substituted",
        lambda root: mutate_json(root / "runtime.json", lambda data: data["hosts"]["192.168.178.47"]["nccl_loaded"].__setitem__("stdout", re.sub(r"sha256=[0-9a-f]{64}", "sha256=" + "0" * 64, data["hosts"]["192.168.178.47"]["nccl_loaded"]["stdout"]))),
        "192.168.178.47 canonical NCCL digest",
    )
    rejected(
        "telemetry capture script hash substituted",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data.__setitem__("telemetry_capture_script_sha256", "0" * 64),
        ),
        "telemetry capture script SHA",
        prepare=install_telemetry_capture_hash,
    )
    rejected(
        "telemetry timestamps moved outside acceptance",
        shift_telemetry_outside_window,
        "telemetry acceptance window",
    )
    rejected(
        "telemetry unknown line",
        lambda root: (root / "load-telemetry.log").write_text(
            (root / "load-telemetry.log").read_text() + "unknown=record\n"
        ),
        "telemetry grammar",
    )
    rejected(
        "telemetry duplicate record",
        lambda root: (root / "load-telemetry.log").write_text(
            (root / "load-telemetry.log").read_text().replace(
                'vllm:num_requests_waiting{',
                'vllm:num_requests_running{engine="0",model_name="GLM-5.3-Flash-EXL3"} 0.0\n'
                'vllm:num_requests_waiting{',
                1,
            )
        ),
        "telemetry grammar",
    )
    rejected(
        "telemetry incomplete GPU sample set",
        lambda root: (root / "load-telemetry.log").write_text(
            re.sub(
                r"^P\d+, \d+ %, \d+ MHz, [0-9.]+ W\n",
                "",
                (root / "load-telemetry.log").read_text(),
                count=1,
                flags=re.MULTILINE,
            )
        ),
        "telemetry grammar",
    )
    rejected(
        "telemetry out-of-order host",
        lambda root: (root / "load-telemetry.log").write_text(
            (root / "load-telemetry.log").read_text().replace(
                "host=192.168.178.47 acquisition_started_at=",
                "host=192.168.178.46 acquisition_started_at=",
                1,
            )
        ),
        "telemetry grammar",
    )
    rejected(
        "request receipt starts before launch",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_exact"].update(
                {
                    "started_at": int((root / "launch-epoch.txt").read_text()) - 2,
                    "completed_at": int((root / "launch-epoch.txt").read_text()) - 1,
                }
            ),
        ),
        "direct_exact request receipt interval",
        prepare=install_proxy_remote_media_receipt,
    )
    rejected(
        "request receipt interval reversed",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_exact"].update(
                {
                    "started_at": data["checks"]["direct_exact"]["completed_at"],
                    "completed_at": data["checks"]["direct_exact"]["started_at"],
                }
            ),
        ),
        "direct_exact request receipt interval",
        prepare=install_proxy_remote_media_receipt,
    )
    rejected(
        "request receipt timestamp missing",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_exact"].pop("completed_at"),
        ),
        "direct_exact request receipt interval",
        prepare=install_proxy_remote_media_receipt,
    )
    rejected(
        "request receipt completes outside acceptance window",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_exact"].__setitem__(
                "completed_at", data["completed_at"] + 1
            ),
        ),
        "direct_exact request receipt interval",
        prepare=install_proxy_remote_media_receipt,
    )
    rejected(
        "request receipt cardinality reduced",
        lambda root: mutate_json(
            root / "acceptance.json",
            lambda data: data["checks"]["direct_concurrency_c4"]["results"].pop(),
        ),
        "C4 cardinality",
        prepare=install_proxy_remote_media_receipt,
    )
    artifact_binding_negative_controls()
    print("negative controls: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
