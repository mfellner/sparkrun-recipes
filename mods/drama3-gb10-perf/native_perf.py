"""Default-off native experiments. SPDX-License-Identifier: MIT.

No generic compute_logits override: compact logits have an explicit tower API.
One serialized model, BF16, batch-one decode only. No private cross-request cache.
"""
import os

import torch
from torch import nn


def feature(name):
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be exactly 0 or 1")
    return value == "1"


def graph_buckets(capacity):
    if type(capacity) is not int or not 2 <= capacity <= 10240:
        raise ValueError("graph capacity must be 2..10240")
    buckets = (1024, 2048, 4096, 10240)
    if feature("FISH_DRAMA3_SMALL_GRAPH_BUCKETS"):
        buckets = (256, 512, *buckets)
    return tuple(sorted({min(b, capacity) for b in buckets}))


def select_bucket(buckets, position):
    if type(position) is not int or position < 0:
        raise ValueError("decode position must be a nonnegative integer")
    for bucket in buckets:
        if bucket >= position + 2:
            return bucket
    raise ValueError("decode position+2 exceeds graph capacity")


def validate_config(config):
    fields = ((config, "eos_token_id", 248046),
              (config, "semantic_start_token_id", 248099),
              (config, "semantic_end_token_id", 252194),
              (config.text, "vocab_size", 252288),
              (config.text, "hidden_size", 2048),
              (config.audio_decoder, "dim", 2048),
              (config.audio_decoder, "vocab_size", 4096),
              (config.audio_decoder, "num_codebooks", 10))
    for obj, name, expected in fields:
        value = getattr(obj, name, None)
        if type(value) is not int or value != expected:
            raise ValueError(f"compact Drama 3 config mismatch: {name}={value!r}, expected {expected}")


def candidate_ids(config, device):
    validate_config(config)
    return torch.cat((torch.tensor([config.eos_token_id], device=device),
                      torch.arange(config.semantic_start_token_id,
                                   config.semantic_end_token_id + 1, device=device)))


def guard_logits(logits, hidden, temperature, status):
    """Accumulate failure on-device; sanitize only to keep failed graph bounded.

    Invalid frames are never released: caller MUST require_finite before emitting.
    Include FP32 sampler scaling overflow, not just the BF16 projection.
    """
    temp = temperature.clamp(min=1e-5) if isinstance(temperature, torch.Tensor) else max(temperature, 1e-5)
    valid = (torch.isfinite(hidden).all() & torch.isfinite(logits).all()
             & torch.isfinite(logits.float() / temp).all())
    status.logical_and_(valid)
    return torch.where(valid, logits, torch.zeros_like(logits))


def require_finite(status):
    if not bool(status):
        raise FloatingPointError("non-finite hidden/candidate/codebook logits; refusing invalid audio")


def initialize_compact(config, backbone, tower):
    """Called only after both checkpoint loaders, before any graph capture."""
    if not feature("FISH_DRAMA3_COMPACT_HEAD"):
        return
    validate_config(config)
    weight = backbone.lm_head.weight
    if weight.shape != (252288, 2048) or weight.device != tower.device:
        raise ValueError("compact GB10 head requires the pinned shape and one head/tower device")
    if getattr(backbone, "compact_head", None) is not None:
        raise ValueError("compact weights may only be initialized once")
    if len(tower.codebook_heads) != 9 or any(h.weight.shape != (4096, 2048) for h in tower.codebook_heads):
        raise ValueError("compact Drama 3 codebook head geometry mismatch")
    head = CompactHead(weight, candidate_ids(config, weight.device))
    for codebook in tower.codebook_heads:
        if codebook.weight.dtype != torch.bfloat16 or not torch.isfinite(codebook.weight).all():
            raise FloatingPointError("invalid BF16 codebook head weights")
    backbone.compact_head = head
    tower.main_candidates = head.candidates


def cache_slots(model):
    """All mutable hybrid prefill caches in the pinned native implementation."""
    slots = {}
    for index, layer in enumerate(model.backbone.model.layers):
        if layer.layer_type == "full_attention":
            for kind in ("k", "v"):
                slots[f"backbone.{index}.{kind}"] = getattr(layer.self_attn, kind + "_cache")
        elif layer.layer_type == "linear_attention":
            for kind in ("conv_state", "recurrent_state"):
                slots[f"backbone.{index}.{kind}"] = getattr(layer.linear_attn, kind)
        else:
            raise ValueError("unknown hybrid cache layer")
    tower = model.audio_decoder
    for index, layer in enumerate((*tower.shared_layers, *tower.residual_layers)):
        for kind in ("k", "v"):
            slots[f"tower.{index}.{kind}"] = getattr(layer.transformer_layer.attention, kind + "_cache")
    if not slots or any(value is None for value in slots.values()):
        raise ValueError("all hybrid caches must be allocated before snapshotting")
    return slots


class RequestPrefixSnapshot:
    """Opt-in, caller-owned prefill-only prototype; NOT a global/model cache.

    Reuse requires identical rows 0..P inclusive: tower row P-1 is labelled
    with row P. P is a full prefill-chunk boundary, so restored suffix GEMM/GDN
    chunk shapes match full re-prefill. No arbitrary longest-prefix splitting.
    No decode state is snapshotted: caller must initialize graph position/frame/
    hidden/RAS history normally AFTER prefill, and owns its RNG/history lifecycle.
    This object must be closed under the request lock on error/cancellation.
    """

    def __init__(self, request_id, rows=128, chunk_size=128):
        if not feature("FISH_DRAMA3_PREFIX_SNAPSHOT"):
            raise ValueError("request-local prefix prototype is disabled")
        if (not isinstance(request_id, str) or not request_id
                or type(rows) is not int or type(chunk_size) is not int
                or rows < 1 or chunk_size < 1 or rows % chunk_size):
            raise ValueError("snapshot needs a request identity and full chunk boundary")
        self.request_id = request_id
        self.rows, self.chunk_size = rows, chunk_size
        self.model = self.prompt = None
        self.state = []
        self.closed = self.last_hit = False

    @staticmethod
    def active(name, value, rows):
        return value[:, :, :rows] if name.endswith((".k", ".v")) else value

    @torch.inference_mode()
    def prefill(self, model, prompt):
        if self.closed or not feature("FISH_DRAMA3_PREFIX_SNAPSHOT"):
            raise ValueError("snapshot is closed or disabled")
        if model.training or prompt.ndim != 2 or len(prompt) <= self.rows:
            raise ValueError("snapshot prefill requires eval mode and a labelled prefix plus suffix")
        if model.cache_capacity < len(prompt):
            raise ValueError("allocate stable cache storage before prefix prefill")
        slots = cache_slots(model)
        self.last_hit = False
        start = 0
        if self.state:
            if model is not self.model:
                raise ValueError("snapshot belongs to another model/request")
            for name, saved, identity in self.state:
                value = slots[name]
                if (value.data_ptr(), tuple(value.shape), value.dtype, value.device) != identity:
                    raise ValueError("snapshot cache storage changed; graph pointers cannot be replaced")
            if torch.equal(prompt[:self.rows + 1].cpu(), self.prompt):
                for name, saved, _ in self.state:
                    self.active(name, slots[name], self.rows).copy_(saved)
                start = self.rows
                self.last_hit = True
        backbone, tower = model.backbone, model.audio_decoder
        frames = prompt.to(tower.device)
        positions = torch.arange(len(frames), device=tower.device)
        for begin in range(start, len(frames), self.chunk_size):
            end = min(begin + self.chunk_size, len(frames))
            chunk = frames[begin:end]
            embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
            hidden = backbone(inputs_embeds=embeddings[None], input_pos=positions[begin:end])[0].to(tower.device)
            paired_end = min(end, len(frames) - 1)
            if paired_end > begin:
                tower.prefill(hidden[:paired_end - begin], positions[begin:paired_end],
                              frames[begin:paired_end], frames[begin + 1:paired_end + 1])
            if end == self.rows and not self.state:
                self.model = model
                self.prompt = prompt[:self.rows + 1].detach().cpu().clone()
                for name, value in slots.items():
                    saved = self.active(name, value, self.rows).clone()
                    if not torch.isfinite(saved).all():
                        self.close()
                        raise FloatingPointError("invalid hybrid prefix state")
                    self.state.append((name, saved, (value.data_ptr(), tuple(value.shape), value.dtype, value.device)))
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("invalid prefix prefill hidden state")
        return hidden[-1:]

    @torch.inference_mode()
    def close(self):
        for _, value, _ in self.state:
            value.zero_()
        if self.prompt is not None:
            self.prompt.zero_()
        self.state.clear()
        self.model = self.prompt = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CompactHead(nn.Module):
    """Retained BF16 candidate rows, packed once; fixed decode output storage."""

    @torch.no_grad()
    def __init__(self, weight, candidates):
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError("compact head requires a BF16 matrix")
        if candidates.ndim != 1 or candidates.dtype != torch.long:
            raise ValueError("candidates must be int64 rows")
        if (not candidates.numel() or candidates.min() < 0 or candidates.max() >= weight.shape[0]
                or not (candidates[1:] > candidates[:-1]).all()):
            raise ValueError("candidates must be nonempty, sorted, unique and in range")
        # Bound temporary validation allocation for the full 1GB source head.
        for block in weight.split(4096):
            if not torch.isfinite(block).all():
                raise FloatingPointError("non-finite source LM head weight")
        self.register_buffer("candidates", candidates.to(weight.device).clone())
        self.register_buffer("weight", weight.detach().index_select(0, self.candidates).contiguous())
        self.register_buffer("logits", weight.new_empty((1, len(candidates))))
        self.register_buffer("finite", torch.ones((), dtype=torch.bool, device=weight.device))

    @torch.no_grad()
    def project(self, hidden):
        if hidden.shape != (1, self.weight.shape[1]) or hidden.dtype != self.weight.dtype:
            raise ValueError("compact decode needs one BF16 hidden row")
        hidden = hidden.to(self.weight.device)
        torch.mm(hidden, self.weight.T, out=self.logits)
        self.finite.copy_(torch.isfinite(hidden).all() & torch.isfinite(self.logits).all())
        return self.logits
