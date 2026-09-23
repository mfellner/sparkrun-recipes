"""Create an exclusive opt-in overlay from the pinned source and native artifacts.

Does not download, install, change defaults, or restart anything. Rebuilt binaries
with a different hash require review and numerical validation before repinning.
The DS4.1 cooperative .so must not be substituted: it is locked to DeepSeek-V4.1.
"""

import argparse
import hashlib
from pathlib import Path, PurePosixPath

# Reviewed repin: overlay/exl3.py gained the opt-in GLM53_EXL3_MOE_FAST
# dispatch (#217) and the KDA large-M BF16 path (#233, TP2+TP3). With both
# flags unset the module builds the same pointer tables and takes the same
# Marlin paths as before, so a generated profile behaves as it did. Refusal
# on any further drift is unchanged.
STOCK_SHA = "849e25882ab7901fbdd7227990a4f125809e1f79288ce6506311b6e6a53e6fb2"
BINARY_SHA = "aa3fe5e9387c7e0d42d685fb2ca8a5fb959ad956600236baac078a9076c17a1c"
ADAPTER_SHA = "9427f6a65def09ebdbea231e42f735236e145f3d02c19cf5e5276c2e704ce1ca"


def checked(path, digest):
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if digest.startswith("UNVALIDATED") or got != digest:
        raise ValueError(f"Unvalidated source/binary hash: {path}")
    return data


def make_profile(stock, artifacts, runtime_directory, output):
    base = checked(Path(stock), STOCK_SHA)
    artifacts = Path(artifacts)
    checked(artifacts / "cooperative_moe.so", BINARY_SHA)
    checked(artifacts / "runtime.py", ADAPTER_SHA)
    runtime_root = PurePosixPath(runtime_directory)
    if not runtime_root.is_absolute() or ".." in runtime_root.parts:
        raise ValueError(
            "Use an absolute container runtime directory without parent traversal"
        )
    footer = (
        "\n# Explicit fixed cooperative MoE opt-in; unsupported calls stay stock.\n"
        "import runpy as _coop_runpy\nimport sys as _coop_sys\n"
        f'_coop_setup = _coop_runpy.run_path({str(runtime_root / "runtime.py")!r})\n'
        f'_coop_setup["install"](_coop_sys.modules[__name__], library_root={str(runtime_root)!r}, enabled=True)\n'
    )
    with Path(output).open("xb") as handle:
        handle.write(base + footer.encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--runtime-directory",
        required=True,
        help="Container path containing the verified binary and adapter on BOTH ranks",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_profile(args.stock, args.artifacts, args.runtime_directory, args.output)
    print(f"Wrote opt-in overlay: {args.output}; no service changes made")
