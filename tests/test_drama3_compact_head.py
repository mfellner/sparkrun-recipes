"""CPU contracts only; no GPU qualification or checkpoint loading."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mods/drama3-gb10-perf"


def runtime():
    path = MOD / "native_perf.py"
    assert path.is_file(), "compact native implementation is missing"
    spec = importlib.util.spec_from_file_location("native_perf_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_once_packed_bf16_projection_retains_values_and_storage():
    perf = runtime()
    torch.manual_seed(21)
    weight = torch.randn(32, 8, dtype=torch.bfloat16)
    original = weight.clone()
    ids = torch.tensor([2, 24, 25, 26, 27])
    head = perf.CompactHead(weight, ids)
    assert head.weight.dtype == torch.bfloat16 and head.weight.is_contiguous()
    assert torch.equal(head.weight, weight[ids])
    pointers = (head.weight.data_ptr(), head.logits.data_ptr(), head.finite.data_ptr())
    with torch.inference_mode():
        for _ in range(3):
            hidden = torch.randn(1, 8, dtype=torch.bfloat16)
            actual = head.project(hidden)
            torch.testing.assert_close(actual, torch.nn.functional.linear(hidden, weight)[:, ids], rtol=0, atol=0)
            assert pointers == (head.weight.data_ptr(), actual.data_ptr(), head.finite.data_ptr())
            assert head.finite.item()
    assert torch.equal(original, weight)
    assert head.weight.data_ptr() != weight.data_ptr()


def pinned_config():
    return SimpleNamespace(text=SimpleNamespace(vocab_size=252288, hidden_size=2048),
                           audio_decoder=SimpleNamespace(vocab_size=4096, dim=2048, num_codebooks=10),
                           eos_token_id=248046, semantic_start_token_id=248099,
                           semantic_end_token_id=252194)


def test_config_gate_and_global_candidate_order(monkeypatch):
    perf = runtime()
    assert perf.feature("FISH_DRAMA3_COMPACT_HEAD") is False
    monkeypatch.setenv("FISH_DRAMA3_COMPACT_HEAD", "1")
    assert perf.feature("FISH_DRAMA3_COMPACT_HEAD") is True
    for bad in ("true", "yes", "", "2"):
        monkeypatch.setenv("FISH_DRAMA3_COMPACT_HEAD", bad)
        with pytest.raises(ValueError):
            perf.feature("FISH_DRAMA3_COMPACT_HEAD")
    config = pinned_config()
    perf.validate_config(config)
    ids = perf.candidate_ids(config, "cpu")
    assert ids.tolist() == [248046, *range(248099, 252195)]
    assert ids[[0, 1, 4096]].tolist() == [248046, 248099, 252194]
    for field in ("eos_token_id", "semantic_start_token_id", "semantic_end_token_id"):
        config = pinned_config()
        setattr(config, field, getattr(config, field) + 1)
        with pytest.raises(ValueError):
            perf.validate_config(config)
    for field in ("vocab_size", "hidden_size"):
        config = pinned_config()
        setattr(config.text, field, getattr(config.text, field) + 1)
        with pytest.raises(ValueError):
            perf.validate_config(config)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_finite_status_never_allows_invalid_hidden_or_candidate(bad):
    perf = runtime()
    weight = torch.ones(32, 8, dtype=torch.bfloat16)
    ids = torch.tensor([2, 24, 25])
    head = perf.CompactHead(weight, ids)
    hidden = torch.ones(1, 8, dtype=torch.bfloat16)
    hidden[0, 0] = bad
    head.project(hidden)
    assert not head.finite.item()
    status = torch.ones((), dtype=torch.bool)
    safe = perf.guard_logits(head.logits, hidden, 0.7, status)
    assert not status.item() and torch.isfinite(safe).all()
    with pytest.raises(FloatingPointError):
        perf.require_finite(status)
    weight[0, 0] = bad  # excluded rows are validated at initialization too
    with pytest.raises(FloatingPointError):
        perf.CompactHead(weight, ids)


def test_finite_guard_covers_scaled_overflow_and_residual_hidden():
    perf = runtime()
    status = torch.ones((), dtype=torch.bool)
    logits = torch.full((1, 4), torch.finfo(torch.float32).max)
    safe = perf.guard_logits(logits, torch.ones(1, 8), 1e-5, status)
    assert not status.item() and torch.isfinite(safe).all()
    status.fill_(True)
    perf.guard_logits(torch.ones(1, 4), torch.full((1, 8), float("nan")), 1.0, status)
    assert not status.item()
    perf.guard_logits(torch.ones(1, 4), torch.ones(1, 8), 1.0, status)
    assert not status.item(), "a later valid codebook must not reset the failure"


@pytest.mark.parametrize("ids", [[2, 2], [24, 2], [-1, 2], [2, 32], []])
def test_candidate_layout_rejects_unsorted_duplicate_and_out_of_range(ids):
    with pytest.raises(ValueError):
        runtime().CompactHead(torch.ones(32, 8, dtype=torch.bfloat16), torch.tensor(ids, dtype=torch.long))


def test_load_gate_does_not_pack_when_off_and_checks_tower_geometry(monkeypatch):
    perf = runtime()
    # Off preserves the full head without looking at a checkpoint/config.
    perf.initialize_compact(None, None, None)
    monkeypatch.setenv("FISH_DRAMA3_COMPACT_HEAD", "1")
    config = pinned_config()
    weight = torch.zeros(1, 2048, dtype=torch.bfloat16).expand(252288, 2048)
    backbone = SimpleNamespace(lm_head=SimpleNamespace(weight=weight))
    tower = SimpleNamespace(device=torch.device("cpu"), codebook_heads=[])
    with pytest.raises(ValueError, match="codebook"):
        perf.initialize_compact(config, backbone, tower)
