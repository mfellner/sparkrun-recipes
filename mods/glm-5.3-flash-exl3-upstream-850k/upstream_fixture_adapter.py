"""Local pytest fixture repairs; never modify the vendored upstream bytes.

The pinned NIC fixture predates two required overlay paths. Supply only those
inputs using its existing temporary placeholder, leaving the real extracted
preflight and every assertion unchanged. This is not a production override.
The scheduler module's __main__ initializes installed source but pytest does
not; run its unchanged installation gate before pytest collects its behavior.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent / "upstream" / "tests"
MAMBA_INPUTS = ("MAMBA_STATE_PATCH_HOST", "MAMBA_CHUNK_PATCH_HOST")


def complete_preflight_env(env):
    """Fill missing fixture inputs only; explicit bad paths must still fail."""
    result = dict(env)
    for key in MAMBA_INPUTS:
        result.setdefault(key, env["STOP_PATCH_HOST"])
    return result


@pytest.fixture(autouse=True, scope="module")
def upstream_fixture_inputs(request):
    module = request.module
    path = Path(module.__file__).resolve()
    with pytest.MonkeyPatch.context() as patch:
        if path == ROOT / "test_nccl_multi_hca.py":
            original_run = module.subprocess.run

            def run(*args, **kwargs):
                kwargs["env"] = complete_preflight_env(kwargs["env"])
                return original_run(*args, **kwargs)

            # Replace only this test module's binding, not global subprocess.
            patch.setattr(module, "subprocess", SimpleNamespace(run=run))
        elif path == ROOT / "test_scheduler_decode_floor.py":
            # No source fallback/skip: installation_tests fails if unavailable.
            patch.setattr(module, "PATCHED_SOURCE", module.installation_tests())
        yield
