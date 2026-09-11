#!/usr/bin/env python3
"""Capture exact-final runtime evidence for GLM-5.3 EXL3 850K."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HOSTS = ("192.168.178.47", "192.168.178.46")
IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:eecb36e14dc34c92d46827fde7b09f7e0bf27e27c426ece126376c02dea6cd2f"
IMAGE_DIGEST = IMAGE.rsplit("@", 1)[1]
E3_EXECUTION_MARKER = "[glm53-e3-executed]"
SAFE_RECIPE_COMMAND_SHA256 = "37145c6f9afd2359f7dee7240141738a27797aaa6e0d54a31bb7ad6bcfbfa082"
SAFE_PROXY_COMMAND_SHA256 = "4451f42dc2bbd6f3f56ce061ebbad83192aa2e9390dc09b6d3d5b0d6f355510e"
EXPECTED_MOD_MANIFEST_SHA256 = "6ea2a5ae515b76604e52918c3feb8e75cf563556df235ff73e1ae96cdda8f87c"
REDACTED = "[REDACTED]"
JSON_SANITIZE_MAX_DEPTH = 6
JSON_SANITIZE_MAX_SIZE = 1_000_000
STRUCTURE_SANITIZE_MAX_DEPTH = 64
HTTP_TCHAR = r"!#$%&'*+\-.^_`|~0-9A-Za-z"
HTTP_HEADER_FIELD = re.compile(
    rf"(?i)(?<![{HTTP_TCHAR}])([{HTTP_TCHAR}]+)(\s*:\s*)"
)
SAFE_ENV_NAMES = frozenset({
    "ABLIT",
    "EXL3_FAT_GROUPED",
    "GLM53_ADAPTIVE_K",
    "GLM53_DENSE_FP8",
    "GLM53_INDEXER_WORKSPACE",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
})
SENSITIVE_NAMES = frozenset({
    "authorization", "proxy_authorization", "api_key", "apikey", "access_key",
    "access_key_id", "accesskeyid", "access_token", "auth_token", "token",
    "session_token", "sessiontoken", "refresh_token", "password", "passwd",
    "secret", "client_secret", "clientsecret", "credential", "credentials",
    "connection_string", "cookie", "passphrase", "private_key", "privatekey",
    "github_pat", "pat", "mysql_pwd", "database_url", "db_url",
    "postgres_url", "postgresql_url", "redis_url", "mongodb_uri", "mongo_uri",
    "pguri", "jdbc_url", "account_key", "auth_config",
    "sqlalchemy_database_uri", "celery_broker_url", "amqp_url",
    "rabbitmq_url", "kafka_url", "elasticsearch_url", "opensearch_url",
    "mssql_url", "mysql_url", "oauth2_bearer", "proxy_user",
})
SENSITIVE_CLI_NAMES = frozenset({"user", "proxy_user", "oauth2_bearer", "u"})
SENSITIVE_CONNECTION_SUFFIXES = frozenset({"url", "uri", "dsn"})
SENSITIVE_ENV_SUFFIXES = frozenset({
    "key", "secret", "token", "password", "passwd", "pwd", "credential",
    "credentials", "url", "uri", "dsn", "auth", "auth_config",
})
SAFE_NONSENSITIVE_NAMES = frozenset({
    "path", "requested_url", "direct_url", "proxy_url", "effective_url", "url",
    "source_registry_url", "container_ai_vllm_build_url",
    "container_org_opencontainers_image_url", "ai_vllm_build_url",
    "org_opencontainers_image_url",
    "max_num_batched_tokens", "num_speculative_tokens", "max_tokens",
    "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens",
})
COMPACT_INFIX_SENSITIVE = frozenset({
    "authorization", "proxyauthorization", "apikey", "accesskey", "accesskeyid",
    "accesstoken", "authtoken", "sessiontoken", "refreshtoken", "password",
    "passwd", "clientsecret", "credential", "credentials", "connectionstring",
    "privatekey", "accountkey", "authconfig", "githubpat", "mysqlpwd",
    "databaseurl", "dburl", "secret", "token",
})
SAFE_ASSIGNMENTS = {
    "BatchMode": re.compile(r"yes"),
    "ABLIT": re.compile(r"0"),
    "EXL3_FAT_GROUPED": re.compile(r"1"),
    "GLM53_ADAPTIVE_K": re.compile(r"off"),
    "GLM53_DENSE_FP8": re.compile(r"off"),
    "GLM53_INDEXER_WORKSPACE": re.compile(r"rightsize"),
    "NCCL_IB_GID_INDEX": re.compile(r"3"),
    "NCCL_IB_HCA": re.compile(r"rocep1s0f1,roceP2p1s0f1"),
    "NCCL_SOCKET_IFNAME": re.compile(r"enP7s7,enp1s0f1np1,enP2p1s0f1np1"),
    "path": re.compile(re.escape("/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2")),
    "sha256": re.compile(r"[0-9a-f]{64}"),
    "manifest_sha256": re.compile(r"[0-9a-f]{64}"),
    "package_version": re.compile(r"2[.]30[.]7"),
    "pid": re.compile(r"[1-9][0-9]*"),
    "oom_kill": re.compile(r"[0-9]+"),
    "label": re.compile(r"disable"),
    "memlock": re.compile(r"-1:-1"),
    "stack": re.compile(r"67108864:67108864"),
    "nofile": re.compile(r"65535:65535"),
    "symbols": re.compile(
        r"exl3_moe,exl3_fat_gemm,exl3_fat_gemm_scatter,"
        r"exl3_fat_moe_gateup,exl3_fat_moe_down,exl3_fat_moe_gather"
    ),
}


def expected_runtime_command(rank: int) -> str:
    speculative = '{"method":"dflash","model":"/cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/dc77ff1c99eeb2df044ee3d4f0094eb033fee410","num_speculative_tokens":7,"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":2}'
    command = (
        "/usr/bin/python3 /usr/local/bin/vllm serve "
        "/cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/024db9f7e9871e8efdf21538ba55af7442be3cd5 "
        "--served-model-name GLM-5.3-Flash-EXL3 --host 0.0.0.0 --port 8000 "
        "--distributed-executor-backend mp --tensor-parallel-size 2 --pipeline-parallel-size 1 "
        "--quantization exl3 --kv-cache-dtype fp8 --gpu-memory-utilization 0.85 "
        "--max-model-len 850000 --max-num-seqs 4 --max-num-batched-tokens 7168 "
        f"--speculative-config {speculative} "
        "--cudagraph-capture-sizes 1 2 4 8 16 24 32 --enable-prefix-caching "
        "--no-enable-flashinfer-autotune --tool-call-parser glm47 --enable-auto-tool-choice "
        "--reasoning-parser glm45 --chat-template /opt/glm53/chat_template.jinja "
        "--allowed-media-domains media.invalid --limit-mm-per-prompt {\"image\":4,\"video\":0} "
        f"--skip-mm-profiling --nnodes 2 --node-rank {rank} "
        "--master-addr 192.168.178.47 --master-port 25000"
    )
    if rank == 1:
        command += " --headless"
    return command


def expected_wrapper_command(rank: int) -> str:
    return (
        "bash /workspace/mods/glm-5.3-flash-exl3-upstream-850k/serve_wrapper.sh "
        + expected_runtime_command(rank).removeprefix(
            "/usr/bin/python3 /usr/local/bin/vllm serve "
        )
    )


CONTAINER_LAUNCHER_COMMAND = (
    "bash -c printf %s c2xlZXAgaW5maW5pdHk= | base64 -d -- | "
    "bash --noprofile --norc"
)
CONTAINER_SHELL_COMMAND = "bash --noprofile --norc"
WATCHDOG_COMMAND = (
    "bash -c SERVE_PID=$(cat /tmp/sparkrun_serve.pid); while kill -0 $SERVE_PID "
    "2>/dev/null; do sleep 5; done; kill 1"
)


def valid_docker_top_inventory(text: str, rank: int) -> bool:
    lines = text.splitlines()
    if not lines or re.fullmatch(r"PID\s+PPID\s+PGID\s+SID\s+COMMAND", lines[0]) is None:
        return False
    commands: list[str] = []
    pids: set[int] = set()
    for line in lines[1:]:
        match = re.fullmatch(
            r"\s*([1-9]\d*)\s+[0-9]+\s+[1-9]\d*\s+[1-9]\d*\s+(.+)", line
        )
        if match is None or int(match.group(1)) in pids:
            return False
        pids.add(int(match.group(1)))
        commands.append(match.group(2))
    exact = {
        CONTAINER_LAUNCHER_COMMAND: 1,
        CONTAINER_SHELL_COMMAND: 1,
        "sleep infinity": 1,
        expected_wrapper_command(rank): 1,
        expected_runtime_command(rank): 1,
        WATCHDOG_COMMAND: 1,
        f"VLLM::Worker_TP{rank}": 1,
    }
    if rank == 0:
        exact["VLLM::EngineCore"] = 1
    resource_tracker = re.compile(
        r"/usr/bin/python3 -c from multiprocessing[.]resource_tracker import main;"
        r"main\([1-9]\d*\)"
    )
    known = [command for command in commands if command in exact]
    resources = sum(resource_tracker.fullmatch(command) is not None for command in commands)
    sleeps = commands.count("sleep 5")
    return (
        all(known.count(command) == count for command, count in exact.items())
        and resources == 1
        and sleeps in (0, 1)
        and len(commands) == sum(exact.values()) + resources + sleeps
    )


def docker_top_runtime_host_pid(text: str, rank: int) -> int:
    expected = expected_runtime_command(rank)
    matches: list[int] = []
    for line in text.splitlines()[1:]:
        match = re.fullmatch(
            r"\s*([1-9]\d*)\s+[0-9]+\s+[1-9]\d*\s+[1-9]\d*\s+(.+)", line
        )
        if match is not None and match.group(2) == expected:
            matches.append(int(match.group(1)))
    if len(matches) != 1:
        raise RuntimeError("docker top does not identify exactly one serving host PID")
    return matches[0]


def docker_top_wrapper_host_pid(text: str, rank: int) -> int:
    expected = expected_wrapper_command(rank)
    matches: list[int] = []
    for line in text.splitlines()[1:]:
        match = re.fullmatch(
            r"\s*([1-9]\d*)\s+[0-9]+\s+[1-9]\d*\s+[1-9]\d*\s+(.+)", line
        )
        if match is not None and match.group(2) == expected:
            matches.append(int(match.group(1)))
    if len(matches) != 1:
        raise RuntimeError("docker top does not identify exactly one wrapper host PID")
    return matches[0]


def docker_top_wrapper_parent_host_pid(text: str, rank: int) -> int:
    expected = expected_wrapper_command(rank)
    matches: list[int] = []
    for line in text.splitlines()[1:]:
        match = re.fullmatch(
            r"\s*[1-9]\d*\s+([1-9]\d*)\s+[1-9]\d*\s+[1-9]\d*\s+(.+)",
            line,
        )
        if match is not None and match.group(2) == expected:
            matches.append(int(match.group(1)))
    if len(matches) != 1:
        raise RuntimeError("docker top does not identify exactly one wrapper parent host PID")
    return matches[0]


def percent_decode_fixed_point(
    value: str,
    max_depth: int = JSON_SANITIZE_MAX_DEPTH,
) -> tuple[str, bool]:
    decoded = value
    for _ in range(max_depth):
        if re.search(r"%(?![0-9A-Fa-f]{2})", decoded):
            return decoded, False
        next_value = urllib.parse.unquote_plus(decoded)
        if next_value == decoded:
            return decoded, True
        decoded = next_value
    if re.search(r"%(?![0-9A-Fa-f]{2})", decoded):
        return decoded, False
    return decoded, urllib.parse.unquote_plus(decoded) == decoded


def percent_decode_outside_header_names(value: str) -> tuple[str, bool]:
    sentinel = "\x00"
    if sentinel in value:
        return value, False
    masked = list(value)
    for found in HTTP_HEADER_FIELD.finditer(value):
        for index in range(found.start(1), found.end(1)):
            if masked[index] == "%":
                masked[index] = sentinel
    decoded, converged = percent_decode_fixed_point("".join(masked))
    return decoded.replace(sentinel, "%"), converged


def normalized_name(name: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def sensitive_environment_name(name: str) -> bool:
    decoded, converged = percent_decode_fixed_point(name)
    if not converged:
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", decoded) is None:
        return False
    normalized = normalized_name(decoded)
    return any(
        normalized == suffix or normalized.endswith("_" + suffix)
        for suffix in SENSITIVE_ENV_SUFFIXES
    )


def sensitive_name(name: str) -> bool:
    decoded, converged = percent_decode_fixed_point(name)
    if not converged:
        return True
    normalized = normalized_name(decoded)
    if normalized in SAFE_NONSENSITIVE_NAMES:
        return False
    compact = normalized.replace("_", "")
    wrapped = f"_{normalized}_"
    normalized_sensitive = {item.replace("_", "") for item in SENSITIVE_NAMES}
    return (
        sensitive_environment_name(decoded)
        or any(
            normalized == suffix or normalized.endswith("_" + suffix)
            for suffix in SENSITIVE_CONNECTION_SUFFIXES
        )
        or normalized in SENSITIVE_NAMES
        or compact in normalized_sensitive
        or any(f"_{item}_" in wrapped for item in SENSITIVE_NAMES)
        or any(
            normalized.startswith(item + "_")
            or normalized.endswith("_" + item)
            for item in SENSITIVE_NAMES
        )
        or any(item in compact for item in COMPACT_INFIX_SENSITIVE)
    )


def sensitive_cli_option(option: str) -> bool:
    decoded, converged = percent_decode_fixed_point(option)
    if not converged:
        return True
    name = decoded.split("=", 1)[0]
    if not name.startswith("-"):
        return False
    normalized = normalized_name(name.lstrip("-"))
    return (
        normalized in SENSITIVE_CLI_NAMES
        or sensitive_name(normalized)
        or any(
            normalized == suffix or normalized.endswith("_" + suffix)
            for suffix in SENSITIVE_CONNECTION_SUFFIXES
        )
    )


def header_container_field(field: str | None) -> bool:
    if field is None:
        return False
    decoded, converged = percent_decode_fixed_point(field)
    if not converged:
        return True
    normalized = normalized_name(decoded)
    aliases = {
        "header", "headers", "http_header", "http_headers",
        "header_list", "headers_list", "headerlist", "headerslist",
        "header_map", "headers_map", "headermap", "headersmap",
        "header_values", "headers_values", "headervalues", "headersvalues",
    }
    return normalized in aliases or any(
        normalized.endswith("_" + alias) for alias in aliases
    )


def argv_container_field(field: str | None) -> bool:
    if field is None:
        return False
    decoded, converged = percent_decode_fixed_point(field)
    if not converged:
        return True
    normalized = normalized_name(decoded)
    aliases = {
        "argv", "argv_list", "argvlist", "argv_values", "argvvalues",
    }
    return normalized in aliases or any(
        normalized.endswith("_" + alias) for alias in aliases
    )


def structural_header_role(field: str) -> tuple[str, bool]:
    decoded, converged = percent_decode_fixed_point(field)
    if not converged:
        return "", False
    return normalized_name(decoded), True


def safe_command_value(value: str) -> bool:
    digest = hashlib.sha256(value.encode()).hexdigest()
    return digest == SAFE_PROXY_COMMAND_SHA256 or value in {
        expected_runtime_command(0),
        expected_runtime_command(1),
    } or digest == SAFE_RECIPE_COMMAND_SHA256


def safe_assignment_value(name: str, value: str) -> bool:
    if name == "command":
        return safe_command_value(value)
    validator = SAFE_ASSIGNMENTS.get(name)
    return validator is not None and validator.fullmatch(value) is not None


def safe_runtime_stdout(value: str) -> bool:
    """Accept only the two canonical live process receipt formats."""
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "mods/glm-5.3-flash-exl3-upstream-850k/SHA256SUMS"
    )
    try:
        manifest_entries = [
            line.split("  ", 1)[1]
            for line in manifest_path.read_text().splitlines()
            if line
        ]
    except (OSError, IndexError):
        manifest_entries = []
    expected_manifest_stdout = "\n".join((
        f"manifest_sha256={EXPECTED_MOD_MANIFEST_SHA256}",
        *(f"{entry}: OK" for entry in manifest_entries),
        "",
    ))
    if manifest_entries and value == expected_manifest_stdout:
        return True

    serving = re.fullmatch(
        r"pid=([1-9][0-9]*)\ncommand=(.*)\n?", value
    )
    if serving:
        return serving.group(2) in {
            expected_runtime_command(0), expected_runtime_command(1)
        }

    if re.fullmatch(r"host_pid=[1-9][0-9]*\ncontainer_pid=[1-9][0-9]*\n?", value):
        return True

    return any(valid_docker_top_inventory(value, rank) for rank in (0, 1))


def safe_remote_shell_arg(value: str) -> bool:
    try:
        parsed = shlex.split(value)
    except ValueError:
        return False
    if len(parsed) != 1:
        return False
    command = parsed[0]
    rdma_command = (
        "rdma link; for d in /sys/class/infiniband/*; do h=$(basename \"$d\"); "
        "printf '%s rate=' \"$h\"; cat \"$d/ports/1/rate\"; "
        "printf '%s state=' \"$h\"; cat \"$d/ports/1/state\"; done"
    )
    if command == rdma_command:
        return True
    containers = re.findall(r"sparkrun_[0-9a-f]+_[0-9a-f]+_node_[01]", command)
    if len(containers) == 1 and any(
        command == direct_listener_command(containers[0], host, rank)
        for rank, host in enumerate(HOSTS)
    ):
        return True
    namespace_match = re.fullmatch(
        r"python3 -c .+ ([1-9][0-9]*)", command, re.DOTALL
    )
    if namespace_match:
        host_pid = int(namespace_match.group(1))
        if command in {
            pid_namespace_command(host_pid),
            wrapper_lineage_command(host_pid),
        }:
            return True
    match = re.match(
        r"docker exec (sparkrun_[0-9a-f]+_[0-9a-f]+_node_[01]) ", command
    )
    if not match:
        return False
    container = match.group(1)
    serving_process = (
        f"docker exec {container} bash -lc "
        + shlex.quote(
            "pid=$(pgrep -fo '/usr/local/bin/vllm serve'); test -n \"$pid\"; "
            "cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline); "
            "printf 'pid=%s\\ncommand=%s\\n' \"$pid\" \"${cmd% }\""
        )
    )
    nccl_pid = re.search(r"bash -lc .*pid=([1-9][0-9]*);", command)
    nccl_loaded = None
    if nccl_pid:
        pid = nccl_pid.group(1)
        nccl_loaded = (
            f"docker exec {container} bash -lc "
            + shlex.quote(
                f"pid={pid}; test -r /proc/{pid}/maps; "
                f"lib=$(awk '/libnccl[.]so[.]2/{{print $6; exit}}' /proc/{pid}/maps); "
                "test -n \"$lib\"; printf 'pid=%s\\npath=%s\\nsha256=' \"$pid\" "
                "\"$(readlink -f \"$lib\")\"; sha256sum \"$lib\" | cut -d' ' -f1; "
                "python3 -c \"import importlib.metadata as m; "
                "print('package_version='+m.version('nvidia-nccl-cu13'))\""
            )
        )
    return command in {
        runtime_overlay_command(container),
        runtime_mod_manifest_command(container, EXPECTED_MOD_MANIFEST_SHA256),
        serve_marker_command(container),
        serving_process,
        nccl_loaded,
        f"docker exec {container} python3 -c {shlex.quote(LISTENER_PROBE)} 8000",
        f"docker exec {container} bash -lc {shlex.quote(runtime_env_command())}",
    } or any(
        command == direct_listener_command(container, host, rank)
        for rank, host in enumerate(HOSTS)
    )


def redact_text_once(text: str) -> str:
    output: list[str] = []
    assignment = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*=\s*)(.*)$")
    export_assignment = re.compile(
        r"^(\s*export\s+)([A-Za-z_][A-Za-z0-9_-]*)(\s*=\s*|\s+)(.*)$"
    )
    header = re.compile(r"(?i)^(\s*(?:proxy-)?authorization\s*:\s*).*$")
    option = re.compile(
        r"(?i)(--(?:api-?key|access-?key(?:-id)?|auth-?token|session-?token|token|password|passwd|secret|client-secret|credential|connection-string)(?:=|\s+))\S+"
    )
    cli_header = re.compile(r"(?i)(?:(--(?:header|http-header)|-H)(?:=|\s+))([^\n]+)")
    inline = re.compile(
        r"(?i)(\b(?:authorization|api[_-]?key|access[_-]?key(?:[_-]?id)?|(?:api|auth|access|refresh|hf|session)[_-]?token|password|passwd|secret|client[_-]?secret|credential|connection[_-]?string|cookie|passphrase|private[_-]?key)[\\\"']{0,8}\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|\S+)"
    )
    embedded_authorization = re.compile(
        r"(?i)(\b(?:proxy-)?authorization\s*:\s*)"
        r"(?:(?:bearer|basic|digest|negotiate|ntlm|oauth)\s+)?"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
    )
    shell_assignment = re.compile(
        r"(?<![A-Za-z0-9_-])([A-Za-z_][A-Za-z0-9_-]*)(\s*=\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s;]+)"
    )
    setenv_assignment = re.compile(
        r"(?i)(?<!\S)(setenv\s+)([A-Za-z_][A-Za-z0-9_-]*)(\s+)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s;]+)"
    )
    printf_assignment = re.compile(
        r"(?i)(?<!\S)(printf\s+-v\s+)([A-Za-z_][A-Za-z0-9_-]*)"
        r"(\s+(?:%[^\s]+\s+)?)(?:\"[^\"]*\"|'[^']*'|[^\s;]+)"
    )
    shell_option = re.compile(
        r"(?<!\S)(-{1,2}[A-Za-z0-9_%.-]+)(=|\s+)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s;]+)"
    )
    attached_user_option = re.compile(r"(?<!\S)-u(?![=\s])[^\s;]+")
    uri_credentials = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^\s/@]+@")
    uri_query_parameter = re.compile(r"([?&])([^&#\s\"']*)")
    def redact_complete_header_value(value: str) -> str:
        for found in HTTP_HEADER_FIELD.finditer(value):
            if sensitive_name(found.group(1)):
                return value[: found.start()] + found.group(0) + REDACTED
        return value

    def redact_uri_query(match: re.Match[str]) -> str:
        prefix, raw_segment = match.group(1, 2)
        if "=" in raw_segment:
            raw_name, _ = raw_segment.split("=", 1)
            if sensitive_name(raw_name):
                return f"{prefix}{raw_name}={REDACTED}"
        decoded_segment, converged = percent_decode_fixed_point(raw_segment)
        if not converged:
            return f"{prefix}{REDACTED}"
        decoded_parts = re.split(r"[&;]", decoded_segment)
        for decoded_part in decoded_parts:
            decoded_name = decoded_part.split("=", 1)[0]
            if sensitive_name(decoded_name):
                if len(decoded_parts) == 1 and "=" in decoded_part:
                    return f"{prefix}{decoded_name}={REDACTED}"
                return f"{prefix}{REDACTED}"
        return match.group(0)
    for line in text.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        export_match = export_assignment.fullmatch(body)
        match = assignment.fullmatch(body)
        if export_match and sensitive_name(export_match.group(2)):
            body = "".join(export_match.group(1, 2, 3)) + REDACTED
        elif match:
            name, value = match.group(2), match.group(4)
            if safe_assignment_value(name, value):
                body = body
            else:
                body = "".join(match.group(1, 2, 3)) + REDACTED
        elif header.fullmatch(body):
            body = header.sub(r"\1" + REDACTED, body)
        else:
            body = redact_complete_header_value(body)
            body = setenv_assignment.sub(
                lambda found: (
                    "".join(found.group(1, 2, 3)) + REDACTED
                    if sensitive_name(found.group(2))
                    else found.group(0)
                ),
                body,
            )
            body = printf_assignment.sub(
                lambda found: (
                    "".join(found.group(1, 2, 3)) + REDACTED
                    if sensitive_name(found.group(2))
                    else found.group(0)
                ),
                body,
            )
            body = shell_assignment.sub(
                lambda found: (
                    found.group(1) + found.group(2) + REDACTED
                    if sensitive_name(found.group(1))
                    else found.group(0)
                ),
                body,
            )
            body = cli_header.sub(lambda match: match.group(1) + " " + REDACTED, body)
            body = attached_user_option.sub("-u" + REDACTED, body)
            body = shell_option.sub(
                lambda found: (
                    found.group(1) + found.group(2) + REDACTED
                    if sensitive_cli_option(found.group(1))
                    else found.group(0)
                ),
                body,
            )
            body = option.sub(r"\1" + REDACTED, body)
            body = embedded_authorization.sub(r"\1" + REDACTED, body)
            body = inline.sub(r"\1" + REDACTED, body)
            body = uri_credentials.sub(r"\1" + REDACTED + "@", body)
            body = uri_query_parameter.sub(redact_uri_query, body)
        output.append(body + ending)
    return "".join(output)


def redact_text(
    text: str,
    field: str | None = None,
    *,
    _json_depth: int = 0,
    _json_max_depth: int = JSON_SANITIZE_MAX_DEPTH,
    _json_max_size: int = JSON_SANITIZE_MAX_SIZE,
    _structure_depth: int = 0,
    _structure_max_depth: int = STRUCTURE_SANITIZE_MAX_DEPTH,
) -> str:
    if len(text.encode()) > _json_max_size:
        return REDACTED
    sanitized = text
    for _ in range(JSON_SANITIZE_MAX_DEPTH):
        next_value = redact_text_once(sanitized)
        if next_value == sanitized:
            break
        sanitized = next_value
    else:
        return REDACTED
    fragments = [sanitized, *sanitized.splitlines()]
    for fragment in fragments:
        decoded, converged = percent_decode_outside_header_names(fragment)
        if not converged:
            return REDACTED
        for candidate in (fragment, decoded):
            if redact_text_once(candidate) != candidate:
                return REDACTED
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            except RecursionError:
                return REDACTED
            if isinstance(parsed, (dict, list, str)):
                if _json_depth >= _json_max_depth:
                    return REDACTED
                sanitized_json = sanitize_evidence(
                    parsed,
                    field,
                    _json_depth=_json_depth + 1,
                    _json_max_depth=_json_max_depth,
                    _json_max_size=_json_max_size,
                    _structure_depth=_structure_depth + 1,
                    _structure_max_depth=_structure_max_depth,
                )
                if sanitized_json != parsed:
                    return REDACTED
    return sanitized


def decode_serialized_argv_fragment(value) -> tuple[str | None, bool]:
    def unwrap(candidate, json_depth: int, structure_depth: int):
        if isinstance(candidate, str):
            decoded, decoded_converged = percent_decode_fixed_point(candidate)
            if not decoded_converged:
                return decoded, False
            try:
                parsed = json.loads(decoded)
            except json.JSONDecodeError:
                return decoded, True
            except RecursionError:
                return decoded, False
            if json_depth >= JSON_SANITIZE_MAX_DEPTH:
                return decoded, False
            return unwrap(parsed, json_depth + 1, structure_depth)
        if isinstance(candidate, (list, tuple)) and len(candidate) == 1:
            if structure_depth >= STRUCTURE_SANITIZE_MAX_DEPTH:
                return None, False
            return unwrap(candidate[0], json_depth, structure_depth + 1)
        if isinstance(candidate, dict) and len(candidate) == 1:
            key, item = next(iter(candidate.items()))
            role, role_converged = structural_header_role(str(key))
            if not role_converged:
                return None, False
            if role == "items":
                if structure_depth >= STRUCTURE_SANITIZE_MAX_DEPTH:
                    return None, False
                return unwrap(item, json_depth, structure_depth + 1)
        return None, True

    scalar, converged = unwrap(value, 0, 0)
    return (scalar if isinstance(scalar, str) else None), converged


def sanitize_argv(
    argv: list,
    field: str | None = "argv",
    *,
    _json_depth: int = 0,
    _json_max_depth: int = JSON_SANITIZE_MAX_DEPTH,
    _json_max_size: int = JSON_SANITIZE_MAX_SIZE,
    _structure_depth: int = 0,
    _structure_max_depth: int = STRUCTURE_SANITIZE_MAX_DEPTH,
) -> list:
    result: list = []
    redact_next = False
    header_options = {"-H", "--header", "--http-header"}
    for item in argv:
        if isinstance(item, str) and safe_remote_shell_arg(item):
            continue
        if not decode_serialized_argv_fragment(item)[1]:
            return [REDACTED] * len(argv)
    for item in argv:
        if redact_next:
            result.append(REDACTED)
            redact_next = False
            continue
        if isinstance(item, str) and safe_remote_shell_arg(item):
            result.append(item)
            continue
        decoded_item, converged = decode_serialized_argv_fragment(item)
        if not converged:
            result.append(REDACTED)
            continue
        if decoded_item is None and not isinstance(item, str):
            result.append(sanitize_evidence(
                item,
                field,
                _json_depth=_json_depth,
                _json_max_depth=_json_max_depth,
                _json_max_size=_json_max_size,
                _structure_depth=_structure_depth + 1,
                _structure_max_depth=_structure_max_depth,
            ))
            continue
        if decoded_item is None:
            decoded_item = item
        if not isinstance(decoded_item, str):
            result.append(REDACTED)
            continue
        if re.fullmatch(r"-u[^=\s].*", decoded_item, re.DOTALL):
            result.append(REDACTED)
            continue
        short_header = re.fullmatch(r"(?i)-H(?:=|\s)?(.*)", decoded_item, re.DOTALL)
        if short_header is not None:
            if short_header.group(1):
                result.append(REDACTED)
            else:
                result.append(item)
                redact_next = True
            continue
        if decoded_item in header_options or (
            "=" not in decoded_item and sensitive_cli_option(decoded_item)
        ):
            result.append(item)
            redact_next = True
            continue
        if re.match(r"(?i)^--(?:header|http-header)=", decoded_item):
            result.append(REDACTED)
            continue
        if "=" in decoded_item and sensitive_cli_option(decoded_item):
            result.append(REDACTED)
            continue
        if isinstance(item, str) and item in {
            globals().get("LISTENER_PROBE"),
            globals().get("DIRECT_LISTENER_PROBE"),
        }:
            result.append(item)
            continue
        if isinstance(item, str) and safe_remote_shell_arg(item):
            result.append(item)
            continue
        if (
            isinstance(item, str)
            and "/usr/local/bin/vllm serve " in item
            and not safe_command_value(item)
        ):
            result.append(REDACTED)
            continue
        if isinstance(item, str):
            result.append(redact_text(
                item,
                field,
                _json_depth=_json_depth,
                _json_max_depth=_json_max_depth,
                _json_max_size=_json_max_size,
                _structure_depth=_structure_depth + 1,
                _structure_max_depth=_structure_max_depth,
            ))
        else:
            result.append(sanitize_evidence(
                item,
                field,
                _json_depth=_json_depth,
                _json_max_depth=_json_max_depth,
                _json_max_size=_json_max_size,
                _structure_depth=_structure_depth + 1,
                _structure_max_depth=_structure_max_depth,
            ))
    if redact_next:
        raise ValueError("credential-bearing option is missing its value")
    return result


def sanitize_evidence(
    value,
    field: str | None = None,
    *,
    _json_depth: int = 0,
    _json_max_depth: int = JSON_SANITIZE_MAX_DEPTH,
    _json_max_size: int = JSON_SANITIZE_MAX_SIZE,
    _structure_depth: int = 0,
    _structure_max_depth: int = STRUCTURE_SANITIZE_MAX_DEPTH,
):
    if isinstance(value, dict):
        if _structure_depth >= _structure_max_depth:
            return REDACTED
        header_context = header_container_field(field)
        argv_context = argv_container_field(field)
        discriminator_roles = {
            "name", "key", "header", "header_name", "headername",
            "field_name", "fieldname", "http_header_name", "httpheadername",
        }
        discriminators = []
        for key, candidate in value.items():
            role, converged = structural_header_role(str(key))
            if not converged:
                return REDACTED
            if role in discriminator_roles:
                discriminators.append((key, candidate))
        if len(discriminators) > 1 and (
            header_context
            or any(
                isinstance(candidate, str) and sensitive_name(candidate)
                for _, candidate in discriminators
            )
        ):
            return REDACTED
        if len(discriminators) == 1:
            discriminator, header_name = discriminators[0]
            if not isinstance(header_name, str):
                if header_context:
                    return REDACTED
            else:
                if re.fullmatch(rf"[{HTTP_TCHAR}]+", header_name) is not None:
                    decoded_header_name, converged = header_name, True
                else:
                    decoded_header_name, converged = percent_decode_fixed_point(header_name)
                if (
                    not converged
                    or re.fullmatch(rf"[{HTTP_TCHAR}]+", decoded_header_name) is None
                ):
                    if header_context:
                        return REDACTED
                elif sensitive_name(decoded_header_name):
                    return {
                        key: item if key == discriminator else REDACTED
                        for key, item in value.items()
                    }
        sensitive_mapping = any(sensitive_name(str(key)) for key in value)
        sibling_payload_roles = {"raw", "value", "raw_value", "rawvalue", "data"}
        sanitized_mapping = {
            key: (
                REDACTED
                if sensitive_name(str(key)) or (
                    sensitive_mapping
                    and (
                        not structural_header_role(str(key))[1]
                        or structural_header_role(str(key))[0] in sibling_payload_roles
                    )
                )
                else sanitize_evidence(
                    item,
                    field if header_context or argv_context else str(key),
                    _json_depth=_json_depth,
                    _json_max_depth=_json_max_depth,
                    _json_max_size=_json_max_size,
                    _structure_depth=_structure_depth + 1,
                    _structure_max_depth=_structure_max_depth,
                )
            )
            for key, item in value.items()
        }
        if (header_context or argv_context) and any(
            sanitized_mapping[key] != item for key, item in value.items()
        ):
            return {
                key: sanitized_mapping[key]
                if sanitized_mapping[key] != item
                else REDACTED
                for key, item in value.items()
            }
        return sanitized_mapping
    if isinstance(value, (list, tuple)):
        if _structure_depth >= _structure_max_depth:
            return REDACTED
        if (
            len(value) >= 2
            and isinstance(value[0], str)
            and ":" not in value[0]
            and sensitive_name(value[0])
        ):
            return [value[0], *([REDACTED] * (len(value) - 1))]
        if argv_container_field(field):
            return sanitize_argv(
                list(value),
                field,
                _json_depth=_json_depth,
                _json_max_depth=_json_max_depth,
                _json_max_size=_json_max_size,
                _structure_depth=_structure_depth,
                _structure_max_depth=_structure_max_depth,
            )
        if header_container_field(field):
            headers: list = []
            index = 0
            while index < len(value):
                item = value[index]
                if (
                    isinstance(item, str)
                    and ":" not in item
                    and sensitive_name(item)
                    and index + 1 < len(value)
                ):
                    headers.extend((item, REDACTED))
                    index += 2
                    continue
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[0], str)
                    and sensitive_name(item[0])
                ):
                    headers.append([item[0], *([REDACTED] * (len(item) - 1))])
                    index += 1
                    continue
                if isinstance(item, dict):
                    headers.append(sanitize_evidence(
                        item,
                        field,
                        _json_depth=_json_depth,
                        _json_max_depth=_json_max_depth,
                        _json_max_size=_json_max_size,
                        _structure_depth=_structure_depth + 1,
                        _structure_max_depth=_structure_max_depth,
                    ))
                    index += 1
                    continue
                headers.append(sanitize_evidence(
                    item,
                    field,
                    _json_depth=_json_depth,
                    _json_max_depth=_json_max_depth,
                    _json_max_size=_json_max_size,
                    _structure_depth=_structure_depth + 1,
                    _structure_max_depth=_structure_max_depth,
                ))
                index += 1
            return headers
        return [
            sanitize_evidence(
                item,
                field,
                _json_depth=_json_depth,
                _json_max_depth=_json_max_depth,
                _json_max_size=_json_max_size,
                _structure_depth=_structure_depth + 1,
                _structure_max_depth=_structure_max_depth,
            )
            for item in value
        ]
    if isinstance(value, str):
        if len(value.encode()) > _json_max_size:
            return REDACTED
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        except RecursionError:
            return REDACTED
        if isinstance(parsed, (dict, list, str)):
            if _json_depth >= _json_max_depth:
                return REDACTED
            ending = "\n" if value.endswith("\n") else ""
            sanitized = sanitize_evidence(
                parsed,
                field,
                _json_depth=_json_depth + 1,
                _json_max_depth=_json_max_depth,
                _json_max_size=_json_max_size,
                _structure_depth=_structure_depth + 1,
                _structure_max_depth=_structure_max_depth,
            )
            if sanitized == REDACTED:
                return REDACTED
            return json.dumps(sanitized, indent=2) + ending
        if field == "stdout" and safe_runtime_stdout(value):
            return value
        if field == "stdout" and "/usr/local/bin/vllm serve " in value:
            return REDACTED
        if (
            (field == "command" or "/usr/local/bin/vllm serve " in value)
            and not safe_command_value(value)
        ):
            return REDACTED
        return redact_text(
            value,
            field,
            _json_depth=_json_depth,
            _json_max_depth=_json_max_depth,
            _json_max_size=_json_max_size,
            _structure_depth=_structure_depth,
            _structure_max_depth=_structure_max_depth,
        )
    return value


def run(argv: list[str], timeout: int = 300) -> dict:
    p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def remote(host: str, command: str, timeout: int = 300) -> dict:
    return run(["ssh", "-o", "BatchMode=yes", host, "bash", "-lc", shlex.quote(command)], timeout)


def runtime_overlay_command(container: str) -> str:
    return (
        f"docker exec {shlex.quote(container)} bash -euo pipefail -c "
        + shlex.quote(
            "python3 /workspace/mods/glm-5.3-flash-exl3-upstream-850k/"
            "verify_runtime_patch_state.py && "
            "python3 /workspace/mods/glm-5.3-flash-exl3-upstream-850k/"
            "test_suppress_stops_multitoken.py --production && "
            "python3 - <<'PY'\n"
            "import exllamav3_ext as e\n"
            "required=('exl3_moe','exl3_fat_gemm','exl3_fat_gemm_scatter',"
            "'exl3_fat_moe_gateup','exl3_fat_moe_down','exl3_fat_moe_gather')\n"
            "print('symbols=' + ','.join(x for x in required if hasattr(e,x)))\n"
            "assert all(hasattr(e,x) for x in required)\n"
            "PY"
        )
    )


def runtime_env_command() -> str:
    names = tuple(sorted(SAFE_ENV_NAMES))
    script = (
        "import os\n"
        f"names={names!r}\n"
        "for name in names: print(f'{name}={os.environ[name]}')"
    )
    return f"python3 -c {shlex.quote(script)}"


def runtime_mod_manifest_command(container: str, expected_digest: str) -> str:
    inner = (
        "cd /workspace/mods/glm-5.3-flash-exl3-upstream-850k; "
        "actual=$(sha256sum SHA256SUMS | cut -d' ' -f1); "
        f"test \"$actual\" = {shlex.quote(expected_digest)}; "
        "printf 'manifest_sha256=%s\\n' \"$actual\"; "
        "sha256sum -c SHA256SUMS"
    )
    return f"docker exec {shlex.quote(container)} bash -euo pipefail -c {shlex.quote(inner)}"


LISTENER_PROBE = r'''import json,os,sys
port=int(sys.argv[1]); sockets=[]
for table,family in (("/proc/net/tcp","ipv4"),("/proc/net/tcp6","ipv6")):
    try: lines=open(table).read().splitlines()[1:]
    except OSError: continue
    for line in lines:
        fields=line.split(); local,state,inode=fields[1],fields[3],fields[9]
        address,hexport=local.split(":")
        if int(hexport,16)==port and state=="0A":
            bind="0.0.0.0" if family=="ipv4" and int(address,16)==0 else ("::" if int(address,16)==0 else address)
            sockets.append((inode,bind))
assert len(sockets)==1,sockets
inode,bind=sockets[0]; owners=[]
for name in os.listdir("/proc"):
    if not name.isdigit(): continue
    try:
        if any(os.readlink("/proc/"+name+"/fd/"+fd)=="socket:["+inode+"]" for fd in os.listdir("/proc/"+name+"/fd")):
            command=open("/proc/"+name+"/cmdline","rb").read().replace(b"\0",b" ").decode().strip()
            owners.append((int(name),command))
    except (OSError,PermissionError): pass
assert len(owners)==1,owners
pid,command=owners[0]
print(json.dumps({"bind":bind,"port":port,"inode":inode,"pid":pid,"command":command},sort_keys=True))'''


def proxy_parent_argv() -> list[str]:
    return [
        "/home/max/.local/bin/uv",
        "tool",
        "uvx",
        "--from",
        "litellm[proxy]==1.82.6",
        "litellm",
        "--config",
        "/home/max/.cache/sparkrun/proxy/litellm_config.yaml",
        "--host",
        "0.0.0.0",
        "--port",
        "4000",
    ]


def proxy_supervisor_argv() -> list[str]:
    return [
        "/home/max/.local/share/uv/tools/sparkrun/bin/python",
        "-m",
        "sparkrun.proxy.autodiscover",
        "/home/max/.cache/sparkrun/proxy/autodiscover.yaml",
    ]


def proxy_child_argv(archive: str) -> list[str]:
    base = f"/home/max/.cache/uv/archive-v0/{archive}/bin"
    return [
        f"{base}/python",
        f"{base}/litellm",
        "--config",
        "/home/max/.cache/sparkrun/proxy/litellm_config.yaml",
        "--host",
        "0.0.0.0",
        "--port",
        "4000",
    ]


def classify_proxy_argv(argv: list[str]) -> str | None:
    if argv == proxy_supervisor_argv():
        return "autodiscover_supervisor"
    if argv == proxy_parent_argv():
        return "uv_parent"
    if len(argv) == 8:
        match = re.fullmatch(
            r"/home/max/[.]cache/uv/archive-v0/([^/]+)/bin/python", argv[0]
        )
        if match and argv == proxy_child_argv(match.group(1)):
            return "listener_child"
    return None


def process_fact(pid: int, role: str | None = None) -> dict:
    proc = Path("/proc") / str(pid)
    argv = [part.decode() for part in proc.joinpath("cmdline").read_bytes().split(b"\0") if part]
    actual_role = classify_proxy_argv(argv)
    if actual_role is None or (role is not None and actual_role != role):
        raise RuntimeError(f"unexpected proxy process identity for PID {pid}")
    stat = proc.joinpath("stat").read_text()
    fields = stat[stat.rfind(") ") + 2 :].split()
    boot_time = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text().splitlines()
        if line.startswith("btime ")
    )
    return {
        "role": actual_role,
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "started_at": boot_time + int(fields[19]) / int(os.sysconf("SC_CLK_TCK")),
        "argv": argv,
    }


def listener_owner(port: int) -> dict:
    result = run(["python3", "-c", LISTENER_PROBE, str(port)])
    if result["returncode"] != 0:
        raise RuntimeError(f"listener probe failed for {port}")
    payload = json.loads(result["stdout"])
    payload.pop("command", None)
    fact = process_fact(int(payload["pid"]), "listener_child")
    return {"probe_argv": result["argv"], **payload, **fact}


def proxy_processes(status_text: str) -> list[dict]:
    matches = re.findall(r"^\s*PID:\s*([1-9]\d*)\s*$", status_text, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError("proxy status did not contain exactly one PID")
    status_pid = int(matches[0])
    rows: list[dict] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [
                part.decode()
                for part in proc.joinpath("cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (OSError, UnicodeDecodeError):
            continue
        if not (
            any("litellm" in argument.lower() for argument in argv)
            or argv == proxy_supervisor_argv()
        ):
            continue
        role = classify_proxy_argv(argv)
        if role is None:
            raise RuntimeError(f"unrecognized LiteLLM process at PID {proc.name}")
        rows.append(process_fact(int(proc.name), role))
    expected_roles = {
        "autodiscover_supervisor",
        "uv_parent",
        "listener_child",
    }
    if len(rows) != 3 or {row["role"] for row in rows} != expected_roles:
        raise RuntimeError(
            "expected exactly one autodiscovery supervisor, LiteLLM uv parent, and listener child"
        )
    supervisor = next(row for row in rows if row["role"] == "autodiscover_supervisor")
    parent = next(row for row in rows if row["role"] == "uv_parent")
    if parent["pid"] != status_pid:
        raise RuntimeError("SparkRun proxy status PID does not identify the uv parent")
    if parent["ppid"] != supervisor["pid"]:
        raise RuntimeError("LiteLLM uv parent is not owned by the autodiscovery supervisor")
    return sorted(rows, key=lambda row: row["pid"])


def proxy_upstream_sockets(owner_pid: int) -> list[dict]:
    owned: set[str] = set()
    for fd in (Path("/proc") / str(owner_pid) / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        match = re.fullmatch(r"socket:\[(\d+)\]", target)
        if match:
            owned.add(match.group(1))
    rows: list[dict] = []
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            remote, state, inode = fields[2], fields[3], fields[9]
            remote_hex, port_hex = remote.rsplit(":", 1)
            remote_port = int(port_hex, 16)
            if state != "01" or remote_port != 8000 or inode not in owned:
                continue
            if table.name == "tcp":
                address = str(ipaddress.IPv4Address(bytes.fromhex(remote_hex)[::-1]))
            else:
                address = str(ipaddress.IPv6Address(bytes.fromhex(remote_hex)))
            rows.append({
                "owner_pid": owner_pid,
                "state": "ESTABLISHED",
                "remote_address": address,
                "remote_port": remote_port,
                "inode": inode,
            })
    if not rows:
        raise RuntimeError("proxy listener has no established upstream port-8000 socket")
    return sorted(rows, key=lambda row: int(row["inode"]))


DIRECT_LISTENER_PROBE = r'''import json,os,sys
namespace,host,container,rank,port=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]); sockets=[]
for table,family in (("/proc/net/tcp","ipv4"),("/proc/net/tcp6","ipv6")):
    try: lines=open(table).read().splitlines()[1:]
    except OSError: continue
    for line in lines:
        fields=line.split(); local,state,inode=fields[1],fields[3],fields[9]
        address,hexport=local.split(":")
        if int(hexport,16)==port and state=="0A":
            bind="0.0.0.0" if family=="ipv4" and int(address,16)==0 else ("::" if int(address,16)==0 else address)
            sockets.append((inode,bind))
listeners=[]
for inode,bind in sockets:
    owners=[]
    for name in os.listdir("/proc"):
        if not name.isdigit(): continue
        try:
            if any(os.readlink("/proc/"+name+"/fd/"+fd)=="socket:["+inode+"]" for fd in os.listdir("/proc/"+name+"/fd")):
                command=open("/proc/"+name+"/cmdline","rb").read().replace(b"\0",b" ").decode().strip()
                owners.append((int(name),command))
        except (OSError,PermissionError): pass
    assert len(owners)==1,owners
    pid,command=owners[0]
    listeners.append({"bind":bind,"inode":inode,"pid":pid,"command":command})
print(json.dumps({"namespace":namespace,"host":host,"container":container,"rank":rank,"endpoint":{"port":port},"listeners":listeners},sort_keys=True))'''


REQUIRED_SERVE_MARKERS = (
    "configured_tier=grouped effective_tier=grouped",
    "Using fp8_ds_mla KV cache format",
    "[glm53-indexer-workspace] rightsize",
    "Graph capturing finished",
    "DFlash2 drafter KV",
)
FATAL_SERVE_MARKERS = (
    "Traceback (most recent call last):",
    "NVRM: Xid",
    "CUDA error: an illegal memory access",
)
SERVE_MARKER_PROBE = r'''import json,pathlib,re
text=pathlib.Path("/tmp/sparkrun_serve.log").read_text()
required=("configured_tier=grouped effective_tier=grouped","Using fp8_ds_mla KV cache format","[glm53-indexer-workspace] rightsize","Graph capturing finished","DFlash2 drafter KV")
fatal=("Traceback (most recent call last):","NVRM: Xid","CUDA error: an illegal memory access")
counts={marker:text.count(marker) for marker in required}
matches=re.findall(r"\[glm53-e3-executed\] grouped_calls=([1-9][0-9]*) fat_expert_runs=([1-9][0-9]*) configured_tier=(\S+) effective_tier=(\S+)",text)
e3=[{"grouped_calls":int(a),"fat_expert_runs":int(b),"configured_tier":c,"effective_tier":d} for a,b,c,d in matches]
found=[marker for marker in fatal if marker in text]
assert all(value>0 for value in counts.values()) and len(e3)==1 and e3[0]["configured_tier"]=="grouped" and e3[0]["effective_tier"]=="grouped" and not found
print(json.dumps({"required_marker_counts":counts,"e3_markers":e3,"fatal_matches":found},sort_keys=True))'''


def serve_marker_command(container: str) -> str:
    return (
        f"docker exec {shlex.quote(container)} /usr/bin/python3 -S -c "
        f"{shlex.quote(SERVE_MARKER_PROBE)}"
    )


def direct_listener_command(container: str, host: str, rank: int) -> str:
    args = f"{shlex.quote(host)} {shlex.quote(container)} {rank} 8000"
    if rank == 0:
        return (
            f"docker exec {shlex.quote(container)} /usr/bin/python3 -S -c "
            f"{shlex.quote(DIRECT_LISTENER_PROBE)} container {args}"
        )
    if rank == 1:
        return (
            f"/usr/bin/python3 -S -c {shlex.quote(DIRECT_LISTENER_PROBE)} "
            f"host {args}"
        )
    raise ValueError("rank must be 0 or 1")


PID_NAMESPACE_PROBE = r'''import pathlib,re,sys
host_pid=int(sys.argv[1]); text=pathlib.Path(f"/proc/{host_pid}/status").read_text()
match=re.search(r"^NSpid:\s+([0-9\s]+)$",text,re.MULTILINE); assert match
pids=[int(value) for value in match.group(1).split()]; assert len(pids)==2 and pids[0]==host_pid
print(f"host_pid={host_pid}\ncontainer_pid={pids[1]}")'''


def pid_namespace_command(host_pid: int) -> str:
    if host_pid <= 0:
        raise ValueError("host PID must be positive")
    return f"python3 -c {shlex.quote(PID_NAMESPACE_PROBE)} {host_pid}"


WRAPPER_LINEAGE_PROBE = r'''import json,pathlib,re,sys
wrapper_host_pid=int(sys.argv[1]); status=pathlib.Path(f"/proc/{wrapper_host_pid}/status").read_text()
parent_match=re.search(r"^PPid:\s+([1-9][0-9]*)$",status,re.MULTILINE); assert parent_match
nspid_match=re.search(r"^NSpid:\s+([0-9\s]+)$",status,re.MULTILINE); assert nspid_match
nspids=[int(value) for value in nspid_match.group(1).split()]; assert len(nspids)==2 and nspids[0]==wrapper_host_pid
parent_host_pid=int(parent_match.group(1)); argv=pathlib.Path(f"/proc/{parent_host_pid}/cmdline").read_bytes().split(b"\0"); argv=[item.decode() for item in argv if item]
assert len(argv)==7 and argv[0]=="/usr/bin/containerd-shim-runc-v2" and argv[1:4]==["-namespace","moby","-id"] and re.fullmatch(r"[0-9a-f]{64}",argv[4]) and argv[5:]==["-address","/run/containerd/containerd.sock"]
payload={"wrapper_host_pid":wrapper_host_pid,"parent_host_pid":parent_host_pid,"parent_executable":argv[0],"parent_namespace":argv[2],"parent_container_id":argv[4],"parent_address":argv[6]}
print(json.dumps(payload,sort_keys=True))'''


def wrapper_lineage_command(wrapper_host_pid: int) -> str:
    if wrapper_host_pid <= 0:
        raise ValueError("wrapper host PID must be positive")
    return f"python3 -c {shlex.quote(WRAPPER_LINEAGE_PROBE)} {wrapper_host_pid}"


def roce_records_command(container: str) -> str:
    inner = (
        "IFS=, read -r -a hcas <<< \"${NCCL_IB_HCA:?NCCL_IB_HCA is required}\"; "
        "python3 /workspace/mods/glm-5.3-flash-exl3-upstream-850k/verify_roce_gid.py "
        "--json --gid-index \"${NCCL_IB_GID_INDEX:?NCCL_IB_GID_INDEX is required}\" "
        "\"${hcas[@]}\""
    )
    return f"docker exec {shlex.quote(container)} bash -euo pipefail -c {shlex.quote(inner)}"


def get(url: str) -> dict:
    started = time.time()
    with urllib.request.urlopen(url, timeout=60) as response:
        effective_url = response.geturl()
        return {
            "requested_url": url,
            "effective_url": effective_url,
            "path": urllib.parse.urlparse(effective_url).path,
            "http": response.status,
            "started_at": started,
            "completed_at": time.time(),
            "body": response.read().decode(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--launch-epoch", required=True, type=int)
    parser.add_argument("--launch-receipt", required=True)
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--acceptance-run-id", required=True)
    parser.add_argument("--expected-mod-manifest-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.expected_mod_manifest_sha256 != EXPECTED_MOD_MANIFEST_SHA256:
        raise RuntimeError("external mod-manifest digest does not match the reviewed capture script")

    acceptance = json.loads(Path(args.acceptance).read_text())
    if (
        re.fullmatch(r"[0-9a-f]{16}", args.acceptance_run_id) is None
        or acceptance.get("run_id") != args.acceptance_run_id
    ):
        raise RuntimeError("acceptance run ID does not match --acceptance-run-id")
    launch_receipt_path = Path(args.launch_receipt)
    launch_receipt = json.loads(launch_receipt_path.read_text())
    if launch_receipt.get("launch_epoch") != args.launch_epoch:
        raise RuntimeError("launch receipt epoch does not match --launch-epoch")
    if launch_receipt.get("mod_manifest_sha256") != args.expected_mod_manifest_sha256:
        raise RuntimeError("launch receipt does not bind the externally expected mod manifest")
    proxy_status = run(["sparkrun", "proxy", "status"])
    if proxy_status["returncode"] != 0:
        raise RuntimeError("SparkRun proxy status failed")
    proxy_models = get("http://127.0.0.1:4000/v1/models")
    proxy_rows = proxy_processes(proxy_status["stdout"])
    proxy_listener = listener_owner(4000)
    upstream_sockets = proxy_upstream_sockets(proxy_listener["pid"])
    record: dict = {
        "schema": 1,
        "process_role": "exact_final",
        "producer": "capture_runtime.py",
        "capture_argv": sys.argv,
        "acceptance_run_id": args.acceptance_run_id,
        "cluster_id": args.cluster_id,
        "launch_epoch": args.launch_epoch,
        "launch_receipt_sha256": hashlib.sha256(launch_receipt_path.read_bytes()).hexdigest(),
        "acceptance_completed_at": acceptance["completed_at"],
        "captured_at": time.time(),
        "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_image_digest": IMAGE_DIGEST,
        "expected_mod_manifest_sha256": args.expected_mod_manifest_sha256,
        "status": run(["sparkrun", "status", "--cluster", "vacation-pair2", "--json"]),
        "proxy_status": proxy_status,
        "proxy_processes": proxy_rows,
        "proxy_listener": proxy_listener,
        "proxy_upstream_sockets": upstream_sockets,
        "direct_models": get("http://127.0.0.1:8000/v1/models"),
        "proxy_models": proxy_models,
        "hosts": {},
    }

    for rank, host in enumerate(HOSTS):
        name_result = remote(
            host,
            f"docker ps --filter name={shlex.quote(args.cluster_id)} --format '{{{{.Names}}}}'",
        )
        names = [line for line in name_result["stdout"].splitlines() if line.strip()]
        expected_container = f"{args.cluster_id}_node_{rank}"
        if name_result["returncode"] != 0 or names != [expected_container]:
            raise RuntimeError(f"expected one serving container on {host}, got {names!r}")
        container = names[0]
        qcontainer = shlex.quote(container)
        docker_inspect = remote(host, f"docker inspect {qcontainer}")
        try:
            inspect_rows = json.loads(docker_inspect["stdout"])
            container_id = inspect_rows[0]["Id"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot bind container ID on {host}") from exc
        if (
            docker_inspect["returncode"] != 0
            or docker_inspect["stderr"] != ""
            or not isinstance(container_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        ):
            raise RuntimeError(f"cannot bind container ID on {host}")
        serving_process = remote(
            host,
            f"docker exec {qcontainer} bash -lc "
            + shlex.quote(
                "pid=$(pgrep -fo '/usr/local/bin/vllm serve'); test -n \"$pid\"; "
                "cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline); "
                "printf 'pid=%s\\ncommand=%s\\n' \"$pid\" \"${cmd% }\""
            ),
        )
        match = re.fullmatch(r"pid=(\d+)\ncommand=(.+)\n?", serving_process["stdout"])
        if serving_process["returncode"] != 0 or match is None:
            raise RuntimeError(f"cannot bind serving PID/command on {host}: {serving_process!r}")
        serving_pid = match.group(1)
        docker_top = remote(host, f"docker top {qcontainer} -eo pid,ppid,pgid,sid,args")
        if docker_top["returncode"] != 0 or not valid_docker_top_inventory(
            docker_top["stdout"], rank
        ):
            raise RuntimeError(f"invalid closed docker top inventory on {host}")
        serving_host_pid = docker_top_runtime_host_pid(docker_top["stdout"], rank)
        wrapper_host_pid = docker_top_wrapper_host_pid(docker_top["stdout"], rank)
        wrapper_parent_host_pid = docker_top_wrapper_parent_host_pid(
            docker_top["stdout"], rank
        )
        pid_namespace = remote(host, pid_namespace_command(serving_host_pid))
        expected_pid_namespace = (
            f"host_pid={serving_host_pid}\ncontainer_pid={serving_pid}\n"
        )
        if (
            pid_namespace["returncode"] != 0
            or pid_namespace["stderr"] != ""
            or pid_namespace["stdout"] != expected_pid_namespace
        ):
            raise RuntimeError(f"serving PID namespace join failed on {host}")
        wrapper_lineage = remote(host, wrapper_lineage_command(wrapper_host_pid))
        try:
            wrapper_lineage_payload = json.loads(wrapper_lineage["stdout"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"wrapper lineage probe failed on {host}") from exc
        if (
            wrapper_lineage["returncode"] != 0
            or wrapper_lineage["stderr"] != ""
            or set(wrapper_lineage_payload) != {
                "wrapper_host_pid", "parent_host_pid", "parent_executable",
                "parent_namespace", "parent_container_id", "parent_address",
            }
            or wrapper_lineage_payload.get("wrapper_host_pid") != wrapper_host_pid
            or wrapper_lineage_payload.get("parent_host_pid")
            != wrapper_parent_host_pid
            or wrapper_lineage_payload.get("parent_executable")
            != "/usr/bin/containerd-shim-runc-v2"
            or wrapper_lineage_payload.get("parent_namespace") != "moby"
            or wrapper_lineage_payload.get("parent_container_id") != container_id
            or wrapper_lineage_payload.get("parent_address")
            != "/run/containerd/containerd.sock"
        ):
            raise RuntimeError(f"wrapper lineage probe failed on {host}")
        serve_markers = remote(host, serve_marker_command(container))
        if serve_markers["returncode"] != 0 or serve_markers["stderr"] != "":
            raise RuntimeError(f"structured serve marker probe failed on {host}")
        serve_log = remote(host, f"docker exec {qcontainer} cat /tmp/sparkrun_serve.log")
        if serve_log["returncode"] != 0:
            raise RuntimeError(f"cannot capture serve log on {host}: {serve_log!r}")
        host_record = {
            "ssh_host": host,
            "rank": rank,
            "container": container,
            "container_discovery": name_result,
            "earlyoom": remote(host, "systemctl is-active earlyoom || true"),
            "docker_inspect": docker_inspect,
            "image_inspect": remote(host, f"docker image inspect {shlex.quote(IMAGE)}"),
            "docker_top": docker_top,
            "pid_namespace": pid_namespace,
            "wrapper_lineage": wrapper_lineage,
            "docker_logs": remote(host, f"docker logs {qcontainer}"),
            "serve_log": serve_log,
            "serve_markers": serve_markers,
            "runtime_env": remote(
                host,
                f"docker exec {qcontainer} bash -lc "
                + shlex.quote(runtime_env_command()),
            ),
            "roce_records": remote(host, roce_records_command(container)),
            "runtime_overlay": remote(host, runtime_overlay_command(container)),
            "runtime_mod_manifest": remote(
                host,
                runtime_mod_manifest_command(container, args.expected_mod_manifest_sha256),
            ),
            "serving_process": serving_process,
            "nccl_loaded": remote(
                host,
                f"docker exec {qcontainer} bash -lc "
                + shlex.quote(
                    f"pid={serving_pid}; test -r /proc/{serving_pid}/maps; "
                    f"lib=$(awk '/libnccl[.]so[.]2/{{print $6; exit}}' /proc/{serving_pid}/maps); "
                    "test -n \"$lib\"; printf 'pid=%s\\npath=%s\\nsha256=' \"$pid\" "
                    "\"$(readlink -f \"$lib\")\"; sha256sum \"$lib\" | cut -d' ' -f1; "
                    "python3 -c \"import importlib.metadata as m; "
                    "print('package_version='+m.version('nvidia-nccl-cu13'))\""
                ),
            ),
            "rdma": remote(
                host,
                "rdma link; for d in /sys/class/infiniband/*; do h=$(basename \"$d\"); "
                "printf '%s rate=' \"$h\"; cat \"$d/ports/1/rate\"; "
                "printf '%s state=' \"$h\"; cat \"$d/ports/1/state\"; done",
            ),
            "memory": remote(
                host,
                "grep -E '^(MemFree|MemAvailable|SwapFree):' /proc/meminfo; "
                "awk '/oom_kill /{print \"oom_kill=\"$2}' /proc/vmstat",
            ),
            "kernel_since_launch": remote(
                host,
                f"journalctl -k --since '@{args.launch_epoch}' --no-pager",
            ),
            "kernel_after_acceptance": remote(
                host,
                f"journalctl -k --since '@{int(acceptance['completed_at'])}' --no-pager",
            ),
        }
        host_record["direct_listener"] = remote(
            host, direct_listener_command(container, host, rank)
        )
        if (
            host_record["direct_listener"]["returncode"] != 0
            or host_record["direct_listener"]["stderr"] != ""
        ):
            raise RuntimeError(f"direct listener probe failed on {host}")
        if host == HOSTS[0]:
            host_record["postready_rc"] = remote(host, f"docker exec {qcontainer} cat /tmp/glm53-postready.rc")
            host_record["postready_ok"] = remote(host, f"docker exec {qcontainer} cat /tmp/glm53-postready.ok")
            host_record["postready_log"] = remote(host, f"docker exec {qcontainer} cat /tmp/glm53-postready.log")
        record["hosts"][host] = host_record

    record["completed_at"] = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize_evidence(record), indent=2) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
