"""Real GB10 platform probes, no weights. SPDX-License-Identifier: MIT."""
import json
import torch
import triton
from fish_speech.models.text2semantic.fish_omni import fp8_moe as fp8
from fish_speech.models.text2semantic.fish_omni.layers import chunk_gated_delta_rule, recurrent_gated_delta_rule


def main():
    assert torch.cuda.is_available(), "CUDA is required; no CPU fallback"
    assert torch.cuda.get_device_capability(0) == (12, 1), "qualified target is GB10 SM121"
    assert fp8.fp8_gemm_supported(["cuda:0"])
    assert not fp8.fp8_gemm_supported(["cpu"])
    saved = fp8.triton
    fp8.triton = None
    assert not fp8.fp8_gemm_supported(["cuda:0"])
    fp8.triton = saved
    torch.manual_seed(42)
    dev = "cuda"
    x = torch.randn(4, 128, device=dev, dtype=torch.bfloat16) * 0.1
    ids = torch.tensor([[0, 1]] * 4, device=dev)
    weights = torch.full((4, 2), 0.5, device=dev)
    w13 = (torch.randn(2, 256, 128, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w2 = (torch.randn(2, 128, 128, device=dev) * 0.05).to(torch.float8_e4m3fn)
    s13 = torch.ones(2, 2, 1, device=dev)
    s2 = torch.ones(2, 1, 1, device=dev)
    actual = fp8.fp8_moe_forward(x, ids, weights, w13, s13, w2, s2)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for e in range(2):
        gate, up = torch.nn.functional.linear(x.float(), w13[e].float()).chunk(2, -1)
        expected += torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, w2[e].float()) * 0.5
    torch.testing.assert_close(actual.float(), expected, atol=0.003, rtol=0.1)
    q, k, v = [torch.randn(1, 8, 2, 16, device=dev, dtype=torch.bfloat16) for _ in range(3)]
    decay = -torch.rand(1, 8, 2, device=dev)
    beta = torch.rand(1, 8, 2, device=dev)
    a, sa = chunk_gated_delta_rule(q, k, v, decay, beta)
    b, sb = recurrent_gated_delta_rule(q, k, v, decay, beta)
    torch.testing.assert_close(a, b, atol=0.004, rtol=0.02)
    torch.testing.assert_close(sa, sb, atol=0.0001, rtol=0.001)
    conv = torch.nn.Conv1d(64, 64, 7, device=dev, dtype=torch.bfloat16)
    signal = torch.randn(1, 64, 1024, device=dev, dtype=torch.bfloat16)
    out = conv(signal)
    assert torch.backends.cudnn.enabled and torch.isfinite(out).all()
    for _ in range(3):
        captured = fp8.fp8_moe_forward(x, ids, weights, w13, s13, w2, s2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fp8.fp8_moe_forward(x, ids, weights, w13, s13, w2, s2)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, actual)
    print(json.dumps(dict(passed=True, device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)), torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__, cudnn=torch.backends.cudnn.version(), cudnn_enabled=torch.backends.cudnn.enabled, fp8_max_abs_error=(actual.float()-expected).abs().max().item(), probes=["FP8 MoE", "GDN chunk/recurrent parity", "cuDNN Conv1d", "FP8 CUDA graph", "CPU rejection", "missing Triton rejection"])))


if __name__ == "__main__":
    main()
