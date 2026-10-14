"""Release-only admission and packaging regressions; no GPU/model imports."""
from pathlib import Path
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'mods/drama3-gb10-release'

def load_release():
    path = MOD / 'serve.py'
    assert path.is_file(), 'minimal release wrapper must exist'
    spec = importlib.util.spec_from_file_location('drama3_minimal_release',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_release_packaging_is_small_and_fail_closed():
    import subprocess
    docker=MOD/'Dockerfile'; run=MOD/'run.sh'
    assert docker.is_file() and run.is_file(), 'minimal release packaging is missing'
    text=docker.read_text()
    assert 'sha256:5da66fdde5e5efaa54abfc9b4c8bcf8517960d4a7b69b87b32e93e2b1440bdaa' in text
    assert 'USER drama' in text and 'FISH_DRAMA3_COMPACT_HEAD=1' in text
    assert 'COPY mods/drama3-gb10-release/ /opt/drama-release/' not in text
    assert not any(name in text for name in ['drama3-native-streaming','drama3-kernels-experimental','drama3-omni'])
    assert subprocess.run(['bash','-n',str(run)]).returncode == 0
    assert 'sha256sum --quiet -c manifest.sha256' in run.read_text()
    assert 'apply_patch.py --verify-only' in run.read_text()

def test_only_compact_head_is_admitted():
    module=load_release()
    result=module.profile_from_env({'FISH_DRAMA3_COMPACT_HEAD':'1'})
    assert result['configured_flags']['FISH_DRAMA3_COMPACT_HEAD'] is True
    assert module.profile_from_env({'FISH_DRAMA3_COMPACT_HEAD':'0'})['configured_flags']['FISH_DRAMA3_COMPACT_HEAD'] is False
    for name in ['FISH_DRAMA3_SMALL_GRAPH_BUCKETS','FISH_DRAMA3_PREFIX_SNAPSHOT','DRAMA_NATIVE_STREAMING','DRAMA_PUBLIC_REFERENCE_CACHE','DRAMA_STATEFUL_CODEC','FISH_DRAMA3_KERNEL_STATE','FISH_DRAMA3_KERNEL_CONV','FISH_DRAMA3_KERNEL_FP8']:
        with pytest.raises(ValueError):module.profile_from_env({name:'1'})
    for name in ['FISH_DRAMA3_KERNEL_TOWER_UP','FISH_DRAMA3_KERNEL_GDN_QKV']:
        with pytest.raises(ValueError):module.profile_from_env({name:'int8'})
    with pytest.raises(ValueError):module.profile_from_env({'FISH_DRAMA3_COMPACT_HEAD':'true'})
