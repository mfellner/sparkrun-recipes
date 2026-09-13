#!/usr/bin/env python3
"""Parse the reviewed recipe command and derive rank argv mechanically."""
from __future__ import annotations

import shlex
from pathlib import Path

import yaml

WRAPPER = "/workspace/mods/glm-5.3-flash-exl3-upstream-850k/serve_wrapper.sh"
VLLM_PREFIX = ("/usr/bin/python3", "/usr/local/bin/vllm", "serve")
EXPORT_PRELUDE = (
    "export",
    "PATH=/usr/local/cuda/bin:/usr/local/bin:${PATH:-}",
    ";",
    "export",
    "GLM53_SERVE_PORT=8000",
    ";",
    "export",
    "GLM53_SERVED_MODEL=GLM-5.3-Flash-EXL3",
    ";",
    "exec",
)


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: UniqueKeyLoader, node, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _shell_tokens(command: str) -> list[str]:
    if "\x00" in command or "`" in command or "$(" in command:
        raise ValueError("recipe command contains executable ambiguity")
    # A backslash-newline is shell line continuation, not an argv token.
    rendered = command.replace("\\\n", "")
    logical_lines = [line for line in rendered.splitlines() if line.strip()]
    if len(logical_lines) != 4:
        raise ValueError("recipe command contains executable ambiguity")
    lexer = shlex.shlex(rendered, posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError as exc:
        raise ValueError("recipe command is not valid shell syntax") from exc


def reviewed_recipe_command(path: Path) -> dict[str, object]:
    try:
        recipe = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("reviewed recipe YAML cannot be parsed") from exc
    if not isinstance(recipe, dict) or not isinstance(recipe.get("command"), str):
        raise ValueError("reviewed recipe has no scalar command")
    raw_command = recipe["command"]
    tokens = _shell_tokens(raw_command)
    if tuple(tokens[: len(EXPORT_PRELUDE)]) != EXPORT_PRELUDE:
        raise ValueError("recipe command prelude is not the reviewed executable form")
    executable = tokens[len(EXPORT_PRELUDE) :]
    if len(executable) < 4 or executable[:2] != ["bash", WRAPPER]:
        raise ValueError("recipe command must exec the reviewed wrapper directly")
    if any(token in {";", "&&", "||", "|", "&", "(", ")", "<", ">", ">>", "<<"} for token in executable):
        raise ValueError("recipe command contains more than one executable")
    options = [token for token in executable[3:] if token.startswith("--")]
    if len(options) != len(set(options)):
        raise ValueError("recipe command contains duplicate options")
    return {
        "raw_command": raw_command,
        "wrapper_argv": executable,
        "serve_argv": [*VLLM_PREFIX, *executable[2:]],
    }


def rank_suffix(rank: int) -> list[str]:
    if rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    suffix = [
        "--nnodes",
        "2",
        "--node-rank",
        str(rank),
        "--master-addr",
        "192.168.178.47",
        "--master-port",
        "25000",
    ]
    if rank == 1:
        suffix.append("--headless")
    return suffix


def reviewed_rank_argv(path: Path) -> dict[int, list[str]]:
    contract = reviewed_recipe_command(path)
    serve = contract.get("serve_argv")
    if not isinstance(serve, list) or not all(isinstance(item, str) for item in serve):
        raise ValueError("invalid reviewed serve argv")
    return {rank: [*serve, *rank_suffix(rank)] for rank in (0, 1)}


def runtime_command(contract: dict[str, object], rank: int) -> str:
    argv = contract.get("serve_argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("invalid reviewed serve argv")
    return " ".join([*argv, *rank_suffix(rank)])


def wrapper_command(contract: dict[str, object], rank: int) -> str:
    argv = contract.get("wrapper_argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("invalid reviewed wrapper argv")
    return " ".join([*argv, *rank_suffix(rank)])
