"""Generate an explicit TP3 overlay beside an already measured native bundle.

No downloads, lifecycle commands, model changes or default configuration edits.
"""
import argparse
import hashlib
from pathlib import Path

from manifest import verify_artifacts

# Reviewed repin for the default-off KDA large-M BF16 path (#233) on TP3.
# Cooperative MoE implementation and TP3's unaligned f_b/g_b exclusion are
# unchanged. With GLM53_KDA_BF16_LARGE_M unset/0, no BF16 copy is retained and
# dense projections still use their existing Marlin/base paths. Enabling the
# in_proj path on TP3 retains the padded-head [8726x4096] copy. Refusal on
# further source drift is unchanged.
STOCK_SHA = "849e25882ab7901fbdd7227990a4f125809e1f79288ce6506311b6e6a53e6fb2"


def prepare(stock, bundle):
    stock, bundle = Path(stock), Path(bundle)
    source = stock.read_bytes()
    if hashlib.sha256(source).hexdigest() != STOCK_SHA:
        raise ValueError("stock EXL3 source changed; review compatibility before repinning")
    verify_artifacts(bundle)
    footer = '''
# Explicit TP3 ABI2 cooperative adapter; complete manifest required on all ranks.
import runpy as _coop_runpy
import sys as _coop_sys
_coop_setup = _coop_runpy.run_path("/root/.cache/vllm/cooperative_moe/runtime.py")
_coop_setup["install"](_coop_sys.modules[__name__], library_root="/root/.cache/vllm/cooperative_moe", enabled=True)
'''
    output = bundle / "exl3-tp3.py"
    with output.open("xb") as handle:
        handle.write(source + footer.encode())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.stock, args.bundle))
