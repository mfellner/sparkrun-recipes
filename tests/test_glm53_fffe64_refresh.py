"""CPU contracts for the fffe64fc launcher refresh; not live acceptance."""
import hashlib
import re
import runpy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'mods/glm-5.3-flash-exl3-upstream-850k'
REVISION = 'fffe64fc093f65fa8f033957e16b744dc266c2ec'


def test_launcher_runs_complete_upstream_order_and_pins_revision():
    source = (MOD / 'upstream/start.sh').read_text()
    block = re.search(r'GLM53_OVERLAY_ORDER=\((.*?)\n\)', source, re.S).group(1)
    expected = re.findall(r'^\s*(patch_\w+\.py)\s*$', block, re.M)
    run = (MOD / 'run.sh').read_text()
    actual = re.findall(r'^python3 (?:upstream/overlay/)?(patch_\w+\.py)$', run, re.M)
    assert [x for x in actual if x not in ('patch_suppress_stops_multitoken.py', 'patch_e3_execution_marker.py')] == expected
    assert actual.index('patch_suppress_stops_multitoken.py') == actual.index('patch_suppress_stops_in_reasoning.py') + 1
    assert actual.index('patch_e3_execution_marker.py') + 1 == actual.index('patch_dense_fp8.py')
    assert 'install -m 0644 upstream/overlay/qwen3_dflash2.py /opt/glm53/qwen3_dflash2.py' in run
    assert REVISION in run
    assert yaml.safe_load((ROOT / 'recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml').read_text())['metadata']['source_revision'] == REVISION
    assert REVISION in (MOD / 'verify_upstream_source_parity.py').read_text()


def test_historical_snapshot_is_unchanged():
    snapshot = ROOT / 'tests/fixtures/glm53-3f2be18/recipe-mod.tar.gz'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == 'e40747b096f145d79a4290ea867b5337ea08c922043e9abc4b012f37bc876f47'


def test_manifest_covers_all_current_mod_files_once():
    rows = [line.split('  ', 1) for line in (MOD / 'SHA256SUMS').read_text().splitlines()]
    entries = {name: digest for digest, name in rows}
    files = {p.relative_to(MOD).as_posix() for p in MOD.rglob('*')
             if p.is_file() and p.name != 'SHA256SUMS'
             and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    assert len(entries) == len(rows)
    assert set(entries) == files
    for name, digest in entries.items():
        assert hashlib.sha256((MOD / name).read_bytes()).hexdigest() == digest, name


def test_compatibility_runner_standalone_modules_exist():
    runner = (MOD / 'run_upstream_compatibility_suite.sh').read_text()
    block = re.search(r'for t in (.*?); do', runner, re.S).group(1)
    names = re.findall(r'test_\w+\.py', block)
    assert names
    assert all((MOD / 'upstream/tests' / name).is_file() for name in names)


def test_api_key_fixture_repairs_only_placeholder_nv_cache():
    adapt = runpy.run_path(str(MOD / 'upstream_fixture_adapter.py'))['repair_api_key_fixture']
    script = 'NV_HOST_CACHE=fixture\nCACHE_ROOT=/tmp/example\n'
    assert adapt(script) == 'NV_HOST_CACHE=/tmp/example\nCACHE_ROOT=/tmp/example\n'
    explicit = 'NV_HOST_CACHE=/explicit\nCACHE_ROOT=/tmp/example\n'
    assert adapt(explicit) == explicit
