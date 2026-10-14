"""Exercise a hash-pinned patch on real upstream tiny CPU modules."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mods/drama3-gb10-perf"
SOURCE = ROOT / "tests/fixtures/drama3-native-source"


@pytest.fixture
def patched(tmp_path, monkeypatch):
    assert (MOD / "apply_patch.py").is_file(), "pinned patch installer is missing"
    target = tmp_path / "fish"
    shutil.copytree(SOURCE, target)
    result = subprocess.run([sys.executable, str(MOD / "apply_patch.py"), str(target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    verified = subprocess.run([sys.executable, str(MOD / "apply_patch.py"), "--verify-only", str(target)], capture_output=True, text=True)
    assert verified.returncode == 0, verified.stderr
    monkeypatch.syspath_prepend(str(target))
    for name in list(sys.modules):
        if name == "fish_speech" or name.startswith("fish_speech."):
            del sys.modules[name]
    yield importlib.import_module("fish_speech.models.text2semantic.fish_omni.tower"), target
    for name in list(sys.modules):
        if name == "fish_speech" or name.startswith("fish_speech."):
            del sys.modules[name]


def tiny_model():
    from fish_speech.models.text2semantic.fish_omni.config import AudioDecoderConfig, FishOmniConfig, FishOmniTextConfig
    from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
    from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder
    text = FishOmniTextConfig(vocab_size=64, hidden_size=16, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, rope_theta=10000., partial_rotary_factor=.5,
        rms_norm_eps=1e-6, max_position_embeddings=1024, linear_conv_kernel_dim=4,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=1,
        linear_num_value_heads=2, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8)
    audio = AudioDecoderConfig(dim=16, intermediate_size=32, n_head=2, n_local_heads=1,
        head_dim=8, norm_eps=1e-6, rope_base=10000., rotary_percent=.5, vocab_size=8,
        num_codebooks=3, shared_enrichment_layers=2, vq_backbone_codebooks=1)
    config = FishOmniConfig(text=text, audio_decoder=audio, semantic_start_token_id=32,
                           semantic_end_token_id=39, eos_token_id=2)
    model = FishOmniModel(config, FishOmniBackbone(text, dtype=torch.bfloat16),
                         FishOmniAudioDecoder(config, dtype=torch.bfloat16), None).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, .05)
    return model


def test_source_fixture_and_fail_closed_installer(tmp_path):
    assert (MOD / "apply_patch.py").is_file(), "pinned patch installer is missing"
    manifest = json.loads((SOURCE / "source-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
    target = tmp_path / "fish"
    shutil.copytree(SOURCE, target)
    path = target / "fish_speech/models/text2semantic/fish_omni/tower.py"
    path.write_text(path.read_text() + "\n# drift\n")
    before = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    result = subprocess.run([sys.executable, str(MOD / "apply_patch.py"), str(target)], capture_output=True, text=True)
    assert result.returncode != 0
    assert before == {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}


def test_compact_tower_eos_ras_and_residual_failure(patched, monkeypatch):
    tower_module, _ = patched
    model = tiny_model()
    tower = model.audio_decoder
    ids = torch.tensor([2, *range(32, 40)])
    tower.main_candidates = ids
    hidden = torch.randn(1, 16, dtype=torch.bfloat16)
    frames = torch.zeros(1, 4, dtype=torch.long)
    positions = torch.tensor([0])
    # A repeated semantic first draw retries; EOS must never retry.
    for first, retry, expected in [(1, 3, 34), (0, 3, 2)]:
        draws = iter([first, retry, 0, 0])
        monkeypatch.setattr(tower_module, "sample_top_k_top_p", lambda logits, *a, **k: torch.full((1,), next(draws), dtype=torch.long))
        status = torch.ones((), dtype=torch.bool)
        out = tower.decode_step(torch.zeros(1, 9, dtype=torch.bfloat16), hidden,
            positions, frames, torch.tensor([2, 32]), temperature=.7, top_k=3, top_p=.7,
            compact_logits=True, finite_status=status)
        assert out[0, 0].item() == expected and status.item()
    monkeypatch.setattr(tower_module, "sample_top_k_top_p", lambda logits, *a, **k: logits.argmax(-1))
    with torch.no_grad():
        tower.codebook_heads[0].weight[0, 0] = float("nan")
    status.fill_(True)
    with pytest.raises(FloatingPointError):
        tower.decode_step(torch.zeros(1, 9, dtype=torch.bfloat16), hidden,
            positions, frames, torch.tensor([32]), temperature=.7, top_k=3, top_p=.7,
            compact_logits=True, finite_status=status)


def test_full_and_compact_eager_generate_share_contract(patched, monkeypatch):
    from fish_speech.models.text2semantic.fish_omni.generate import generate_drama3
    from fish_speech.models.text2semantic.fish_omni.native_perf import CompactHead
    monkeypatch.setenv("FISH_DRAMA3_CUDA_GRAPH", "0")
    torch.manual_seed(19)
    model = tiny_model()
    prompt = torch.tensor([[3, 0, 0, 0], [34, 2, 4, 5], [3, 0, 0, 0], [4, 0, 0, 0]])
    torch.manual_seed(33)
    full, reason = generate_drama3(model, prompt, max_new_tokens=4)
    candidates = torch.tensor([2, *range(32, 40)])
    model.backbone.compact_head = CompactHead(model.backbone.lm_head.weight, candidates)
    model.audio_decoder.main_candidates = candidates
    torch.manual_seed(33)
    compact, compact_reason = generate_drama3(model, prompt, max_new_tokens=4)
    assert torch.equal(full, compact) and reason == compact_reason
    assert compact.dtype == torch.long and compact.device.type == "cpu"


def test_no_per_frame_gather_and_graph_status_source_contract(patched):
    _, target = patched
    path = target / "fish_speech/models/text2semantic/fish_omni"
    generate = (path / "generate.py").read_text()
    assert "compact_head.project(self.hidden)" in generate
    assert "self.finite.logical_and_(compact_head.finite)" in generate
    assert "compact_logits=compact_head is not None" in generate
    assert "finite_status=compact_head.finite" in generate
    assert "index_select" not in generate
    head = (path / "native_perf.py").read_text().split("def project", 1)[1]
    assert "index_select" not in head


def test_real_cpu_graph_run_propagates_residual_failure_before_frame(patched):
    from types import SimpleNamespace
    from fish_speech.models.text2semantic.fish_omni.generate import Drama3DecodeGraphs, ensure_drama3_caches
    from fish_speech.models.text2semantic.fish_omni.native_perf import CompactHead
    model = tiny_model()
    ensure_drama3_caches(model, 256)
    candidates = torch.tensor([2, *range(32, 40)])
    model.backbone.compact_head = CompactHead(model.backbone.lm_head.weight, candidates)
    model.audio_decoder.main_candidates = candidates
    state = SimpleNamespace(model=model, hidden=torch.ones(1, 16, dtype=torch.bfloat16),
        next_hidden=torch.zeros(1, 16, dtype=torch.bfloat16), finite=torch.ones((), dtype=torch.bool),
        position=torch.tensor([0]), frame=torch.zeros(1, 4, dtype=torch.long),
        sampled=torch.zeros(1, 4, dtype=torch.long), history=torch.full((16,), -1, dtype=torch.long),
        temperature=torch.tensor(.7), top_k=torch.tensor(3), top_p=torch.tensor(.7), buckets=(256,))
    for layer in (*model.audio_decoder.shared_layers, *model.audio_decoder.residual_layers):
        layer.transformer_layer.attention.static_kv_len = 256
    state.graphs = {256: SimpleNamespace(replay=lambda: Drama3DecodeGraphs.run(state, 256))}
    with torch.inference_mode():
        frame, main = Drama3DecodeGraphs.step(state, 0)
        assert frame.shape == (1, 4) and main in candidates and state.finite.item()
        model.audio_decoder.codebook_heads[0].weight[0, 0] = float("nan")
        with pytest.raises(FloatingPointError):
            Drama3DecodeGraphs.step(state, 1)
        assert not state.finite.item()
