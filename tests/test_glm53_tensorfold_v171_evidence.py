"""Fail-closed mutation controls for v1.7.1's committed live receipts."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "evidence/glm53-tensorfold-20261005"


def load(name: str):
    spec = importlib.util.spec_from_file_location("v171_" + name, HERE / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipts():
    return (json.loads((HERE / "receipt.json").read_text()),
            json.loads((HERE / "runtime.json").read_text()),
            json.loads((HERE / "staging.json").read_text()))


def test_exact_staged_v171_receipts_pass():
    request, runtime, stage = receipts()
    assert load("verify_staging").verify(stage)["passed"] is True
    assert load("verify_runtime").verify_runtime(request, runtime)["runtime_verified"] is True
    launch = json.loads((HERE / "launch.json").read_text())
    assert load("verify_launch").verify(launch, request)["passed"] is True


@pytest.mark.parametrize("mutation", ["image_id", "missing_rank", "false_returncode", "relabel_rank"])
def test_staging_mutations_fail_closed(mutation):
    stage = receipts()[2]
    bad = copy.deepcopy(stage)
    hosts = ("192.168.178.47", "192.168.178.46")
    if mutation == "image_id":
        image = json.loads(bad["hosts"][hosts[0]]["image"]["stdout"])
        image[0]["Id"] = "sha256:" + "0" * 64
        bad["hosts"][hosts[0]]["image"]["stdout"] = json.dumps(image)
    elif mutation == "missing_rank":
        del bad["hosts"][hosts[1]]
    elif mutation == "false_returncode":
        bad["hosts"][hosts[1]]["snapshot"]["returncode"] = 1
    else:
        bad["hosts"][hosts[0]], bad["hosts"][hosts[1]] = bad["hosts"][hosts[1]], bad["hosts"][hosts[0]]
    with pytest.raises(AssertionError):
        load("verify_staging").verify(bad)


@pytest.mark.parametrize("mutation", ["wrong_model", "weak_privilege", "wrong_container", "bad_final_order"])
def test_runtime_mutations_fail_closed(mutation):
    request, runtime, _ = receipts()
    bad_request, bad_runtime = copy.deepcopy(request), copy.deepcopy(runtime)
    if mutation == "wrong_model":
        bad_request["tests"]["direct"]["response"]["model"] = "wrong"
    elif mutation == "weak_privilege":
        bad_request["hosts"]["192.168.178.46"]["privileged"] = 0
    elif mutation == "wrong_container":
        bad_runtime["hosts"]["192.168.178.46"]["inspect"]["argv"][-1] = "not-the-worker"
    else:
        bad_runtime["final_health"]["started_at"] = bad_runtime["captured_at"]
    with pytest.raises((AssertionError, KeyError)):
        load("verify_runtime").verify_runtime(bad_request, bad_runtime)


@pytest.mark.parametrize("mutation", ["temperature", "proxy_output_cap", "role", "long_thinking",
                                           "vision_control", "tool_description", "listener_owner"])
def test_complete_request_and_listener_mutations_fail_closed(mutation):
    request = receipts()[0]
    bad = copy.deepcopy(request)
    tests = bad["tests"]
    if mutation == "temperature":
        tests["direct"]["request"]["temperature"] = 1
    elif mutation == "proxy_output_cap":
        tests["proxy"]["request"]["max_tokens"] = 1
    elif mutation == "role":
        tests["concurrent"][2]["request"]["messages"][0]["role"] = "system"
    elif mutation == "long_thinking":
        tests["long_needle"]["request"]["chat_template_kwargs"]["enable_thinking"] = True
    elif mutation == "vision_control":
        tests["vision"]["request"]["max_tokens"] = 1
    elif mutation == "tool_description":
        tests["tool"]["request"]["tools"][0]["function"]["description"] = "unrelated"
    else:
        obj = json.loads(bad["listener_after"]["stdout"])
        obj["inode"] = "0"
        bad["listener_after"]["stdout"] = json.dumps(obj)
    with pytest.raises((AssertionError, KeyError)):
        load("verify").verify(bad)


@pytest.mark.parametrize("mutation", ["empty_kernel", "short_log", "nccl_error",
                                           "late_window", "listener_join"])
def test_kernel_log_and_process_mutations_fail_closed(mutation):
    request, runtime, _ = receipts()
    bad = copy.deepcopy(runtime)
    host = "192.168.178.47"
    if mutation == "empty_kernel":
        bad["hosts"][host]["kernel"]["stdout"] = ""
    elif mutation == "short_log":
        bad["hosts"][host]["serve_log"]["stdout"] = "[tensorfold] serving GLM-5.3-Flash-EXL3"
    elif mutation == "nccl_error":
        bad["hosts"][host]["serve_log"]["stdout"] += "\nNCCL ERROR collective failed"
    elif mutation == "late_window":
        bad["kernel_since"] = int(request["captured_at"])
    else:
        process = bad["hosts"][host]["processes"]
        process["stdout"] = process["stdout"].replace("--port 8000", "--port 8001")
    with pytest.raises((AssertionError, KeyError)):
        load("verify_runtime").verify_runtime(request, bad)


@pytest.mark.parametrize("mutation", ["semantic_recipe", "mod_byte", "metadata_mtime", "rank_id"])
def test_persisted_launch_state_mutations_fail_closed(mutation):
    request = receipts()[0]
    launch = json.loads((HERE / "launch.json").read_text())
    bad = copy.deepcopy(launch)
    if mutation == "semantic_recipe":
        import yaml
        state = yaml.safe_load(bad["job_yaml"])
        state["recipe_state"]["_raw"]["env"]["TF_CONTEXT"] = "4096"
        bad["job_yaml"] = yaml.safe_dump(state)
        import hashlib
        bad["job_sha256"] = hashlib.sha256(bad["job_yaml"].encode()).hexdigest()
    elif mutation == "mod_byte":
        row = bad["hosts"]["192.168.178.46"]["mod_sha256"]
        row["stdout"] = "0" + row["stdout"][1:]
    elif mutation == "metadata_mtime":
        bad["job_mtime_ns"] = int(request["captured_at"] * 1e9) + 10**9
    else:
        row = bad["hosts"]["192.168.178.46"]["inspect"]
        obj = json.loads(row["stdout"])
        obj[0]["Id"] = "0" * 64
        row["stdout"] = json.dumps(obj)
    with pytest.raises((AssertionError, KeyError)):
        load("verify_launch").verify(bad, request)


@pytest.mark.parametrize("mutation", ["top_pid", "listener_pid", "listener_start",
                                           "omit_log_line", "omit_kernel_entry", "mapping_start"])
def test_independent_process_and_log_joins_reject_omissions(mutation):
    request, runtime, _ = receipts()
    bad_request, bad_runtime = copy.deepcopy(request), copy.deepcopy(runtime)
    host = "192.168.178.47"
    if mutation == "top_pid":
        top = bad_runtime["hosts"][host]["processes"]
        pid = next(line.split()[0] for line in top["stdout"].splitlines()
                   if "/usr/local/bin/tensorfold serve " in line)
        top["stdout"] = top["stdout"].replace(pid, "9999999", 1)
    elif mutation in ("listener_pid", "listener_start"):
        field = "pid" if mutation == "listener_pid" else "start_ticks"
        for key in ("listener_before", "listener_after"):
            row = bad_request[key]
            value = json.loads(row["stdout"])
            value["owner"][field] += 1
            row["stdout"] = json.dumps(value)
    elif mutation == "omit_log_line":
        row = bad_runtime["hosts"][host]["serve_log"]
        row["stdout"] = row["stdout"].replace(
            "[tensorfold] EXL3 support is experimental: replies are exact, but the MLX checkpoint (Vontra/GLM-5.3-Flash-MLX-4bit-MTP) is tested more and runs faster (docs/recipes/glm-5.3-flash.md)\n", "")
    elif mutation == "omit_kernel_entry":
        row = bad_runtime["hosts"][host]["kernel"]
        lines = row["stdout"].splitlines(keepends=True)
        assert len(lines) >= 3
        row["stdout"] = "".join(lines[1:])
    else:
        row = bad_runtime["hosts"][host]["process_map"]
        mapped = json.loads(row["stdout"])
        mapped["start_ticks"] += 1
        row["stdout"] = json.dumps(mapped)
    with pytest.raises((AssertionError, KeyError)):
        load("verify_runtime").verify_runtime(bad_request, bad_runtime)


@pytest.mark.parametrize("mutation", ["prompt_usage", "tokenize_count", "pool_before", "pool_after"])
def test_exact_published_measurements_fail_closed(mutation):
    receipt = receipts()[0]
    bad = copy.deepcopy(receipt)
    if mutation == "prompt_usage":
        bad["tests"]["long_needle"]["response"]["usage"]["prompt_tokens"] = 50000
    elif mutation == "tokenize_count":
        bad["long_tokenize"]["body"]["count"] = 50000
        bad["long_tokenize"]["body"]["tokens"] = bad["long_tokenize"]["body"]["tokens"][:50000]
    elif mutation == "pool_before":
        bad["health_before"]["body"]["pool_tokens"] = 1048576
    else:
        bad["health_after"]["body"]["pool_tokens"] = 1048576
    with pytest.raises((AssertionError, KeyError)):
        load("verify").verify(bad)


@pytest.mark.parametrize("host", ["192.168.178.47", "192.168.178.46"])
@pytest.mark.parametrize("hca", ["rocep1s0f1", "roceP2p1s0f1"])
def test_each_selected_cx7_rail_must_remain_active(host, hca):
    request, runtime, _ = receipts()
    bad = copy.deepcopy(runtime)
    row = bad["hosts"][host]["rdma"]
    target = f"link {hca}/1 state ACTIVE physical_state LINK_UP"
    assert target in row["stdout"]
    row["stdout"] = row["stdout"].replace(target, f"link {hca}/1 state DOWN physical_state DISABLED")
    with pytest.raises(AssertionError):
        load("verify_runtime").verify_runtime(request, bad)
