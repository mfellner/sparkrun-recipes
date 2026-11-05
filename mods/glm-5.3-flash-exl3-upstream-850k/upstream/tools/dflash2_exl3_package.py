"""Package the converter output into the servable DFlash2 EXL3 snapshot.

Inputs:
  --src    BF16 draft snapshot (incoai/GLM-5.3-Flash-DFlash2)
  --conv   converter output dir (run_convert.py -o)
  --out    final snapshot dir

Rules:
  - A linear present in the converter output as {key}.trellis is EXL3: carry
    trellis/suh/svh/mul1 (or mcg); drop its .weight and the su/sv training
    tensors (the recipe's Exl3LinearMethod loads exactly those four).
  - Every other source tensor (k_proj/v_proj .weight, norms, conv base
    kernels, selector codebooks, hidden_projection) is copied from the SOURCE
    snapshot, byte-identical BF16 (the converter re-saves unquantized linears
    as fp16; we keep the original bytes).
  - config.json gets the serving quantization_config block with
    non_routed_exl3 served prefixes (model.* as constructed by
    DFlash2Qwen3ForCausalLM with prefix ""); quantization_config.json carries
    the same block plus the converter's tensor_storage ledger.

Writes draft_quant_summary.json (sizes + per-tensor bpw) into --out.
"""

import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

QUANT_PARTS = ("trellis", "suh", "svh", "mul1", "mcg")
DROP_PARTS = ("su", "sv")


def read_all(path):
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conv_file = os.path.join(args.conv, "model.safetensors")
    src_file = os.path.join(args.src, "model.safetensors")
    conv = read_all(conv_file)

    trellis_keys = sorted(k[: -len(".trellis")] for k in conv if k.endswith(".trellis"))
    quant = {}
    for base in trellis_keys:
        parts = {}
        for p in QUANT_PARTS:
            k = f"{base}.{p}"
            if k in conv:
                parts[p] = conv.pop(k)
        assert "trellis" in parts and ("mul1" in parts) != ("mcg" in parts), base
        quant[base] = parts
    for k in list(conv):
        if k.rsplit(".", 1)[-1] in DROP_PARTS:
            conv.pop(k)
    # What remains unclaimed in the converter output: unquantized linear
    # weights re-saved as fp16 (k/v, hidden_projection) — dropped in favour of
    # the source BF16 bytes.
    leftover = sorted(conv)
    print(f" -- dropping {len(leftover)} converter re-saved tensors "
          f"(restored from source BF16): {leftover}")

    src = read_all(src_file)
    out_tensors = {}
    dropped_src = []
    for k, t in src.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base and base in quant:
            dropped_src.append(k)
            continue
        out_tensors[k] = t
    for base, parts in quant.items():
        assert base + ".weight" in src, f"quantized module missing from source: {base}"
        for p, t in parts.items():
            out_tensors[f"{base}.{p}"] = t
    assert len(dropped_src) == len(quant)

    os.makedirs(args.out, exist_ok=True)
    save_file(out_tensors, os.path.join(args.out, "model.safetensors"))

    # per-tensor K from the actual trellis (bits-parameterized variants)
    quant_K = {b: p["trellis"].shape[-1] // 16 for b, p in quant.items()}
    assert len(set(quant_K.values())) == 1, f"mixed bitrates: {sorted(set(quant_K.values()))}"
    main_bits = next(iter(quant_K.values()))
    # served-prefix map for the recipe's Exl3Config (config.json block), derived
    # from what was actually quantized (source key -> served vLLM module prefix).
    def served_entry(base):
        k = quant_K[base]
        if base == "fc":
            return "model.fc", {"bits": k}
        parts = base.split(".")
        layer, mod = ".".join(parts[:2]), ".".join(parts[2:])
        p = f"model.{layer}.{mod}"
        if mod == "self_attn.q_proj":
            return p[: -len("q_proj")] + "qkv_proj", {"bits": k, "bf16_shards": [1, 2]}
        if mod in ("self_attn.k_proj", "self_attn.v_proj"):
            raise AssertionError(f"k/v must stay BF16, got quantized: {base}")
        if mod in ("mlp.gate_proj", "mlp.up_proj"):
            return p.rsplit(".", 1)[0] + ".gate_up_proj", {"bits": k}
        return p, {"bits": k}

    layers_cfg = {}
    for base in sorted(quant):
        prefix, entry = served_entry(base)
        layers_cfg[prefix] = entry  # gate/up collapse onto one entry
    n_layers = sum(1 for b in quant if b.endswith("self_attn.q_proj"))
    with open(os.path.join(args.src, "config.json")) as f:
        n_layers_expected = json.load(f)["num_hidden_layers"]
    assert n_layers == n_layers_expected, (n_layers, n_layers_expected)
    # fc is quantized unless left BF16 (DRAFT_QUANT_FC=0 at convert time)
    if "fc" not in quant:
        assert "fc.weight" in src, "fc tensor missing from source"
    else:
        assert "model.fc" in layers_cfg
    for i in range(n_layers):
        for suf in ("self_attn.qkv_proj", "self_attn.o_proj",
                    "mlp.gate_up_proj", "mlp.down_proj"):
            assert f"model.layers.{i}.{suf}" in layers_cfg, suf
    print(f" -- served EXL3 modules (boot-log count): {len(layers_cfg)}")

    with open(os.path.join(args.conv, "config.json")) as f:
        conv_cfg = json.load(f)
    conv_q = conv_cfg.get("quantization_config", {})
    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)
    qblock = {
        "quant_method": "exl3",
        "bits": main_bits,
        "codebook": "mcg",
        "scope": "dflash2_draft",
        "head_bits": 16,
        "version": conv_q.get("version", "unknown"),
        "converter": "MiaAI-Lab/exllamav3@63b32f0 (v1.4.2), uncalibrated, "
                     "k_proj/v_proj BF16 by design (fused context-KV)",
        "non_routed_exl3": {"layers": layers_cfg},
    }
    cfg["quantization_config"] = qblock
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    ledger = {}
    conv_qc_path = os.path.join(args.conv, "quantization_config.json")
    if os.path.exists(conv_qc_path):
        with open(conv_qc_path) as f:
            ledger = json.load(f).get("tensor_storage", {})
    with open(os.path.join(args.out, "quantization_config.json"), "w") as f:
        json.dump({**qblock, "tensor_storage": ledger}, f, indent=2)

    for f in os.listdir(args.src):
        if f in ("config.json", "model.safetensors", "model.safetensors.index.json"):
            continue
        p = os.path.join(args.src, f)
        if os.path.isfile(p):
            shutil.copy(p, os.path.join(args.out, f))

    # summary: sizes + per-tensor bpw
    DT = {"I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4}
    summary = {"quantized": {}, "bf16_bytes": 0, "exl3_bytes": 0}
    for base, parts in sorted(quant.items()):
        k = parts["trellis"].shape[-1] // 16
        n_bytes = sum(t.nelement() * t.element_size() for t in parts.values())
        src_w = src[base + ".weight"]
        n_w = src_w.nelement()
        summary["quantized"][base] = {
            "K": k,
            "shape": list(src_w.shape),
            "bytes": n_bytes,
            "bpw_effective": round(n_bytes * 8 / n_w, 3),
        }
        summary["exl3_bytes"] += n_bytes
    for k, t in out_tensors.items():
        if not any(k.startswith(b + ".") and k.rsplit(".", 1)[-1] in QUANT_PARTS
                   for b in quant):
            summary["bf16_bytes"] += t.nelement() * t.element_size()
    summary["total_bytes"] = summary["bf16_bytes"] + summary["exl3_bytes"]
    total_w = sum(t.nelement() for t in src.values())
    summary["total_bpw"] = round(summary["total_bytes"] * 8 / total_w, 3)
    with open(os.path.join(args.out, "draft_quant_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "quantized"}, indent=2))
    print(f" -- quantized tensors: {len(quant)}")
    for b, d in summary["quantized"].items():
        print(f"    {b:45s} K={d['K']} {str(d['shape']):20s} "
              f"{d['bytes'] / 2**20:8.2f} MiB  {d['bpw_effective']:.2f} bpw")


if __name__ == "__main__":
    main()
