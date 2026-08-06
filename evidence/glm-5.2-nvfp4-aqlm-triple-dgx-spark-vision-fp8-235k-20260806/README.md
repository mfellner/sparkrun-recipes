# GLM-5.2 Path B acceptance evidence — 2026-08-06

This directory records the evidence underlying the performance and functional claims for `recipes/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-fp8-235k.yaml`.

## Provenance

- Recipe SHA-256 at test time: `6e43eae227476f6bfa2417a0f22b138fcf36921d160f20aa8211551f07ff10d5`
- Repository base commit: `fd6c55fe45b3c0e3ff0f9a6b5ab3ba9eb7fd3ad5`
- SparkRun workload: `66a8e0550db98b58_de8ef6e2d145`
- Cluster and rank order: `dgx03` (`192.168.0.184`, head), `dgx04` (`192.168.0.49`, worker), `gx10` (`192.168.0.73`, worker)
- Upstream source: `MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks@5c85163ccb8d98d395880e71e2dbd03976a3f4ad`
- Model revision: `53e0082eedebd806b63e19779c47905937d768ca`
- Image digest: `sha256:f8f350d46b33858eda4f0c0a5c39a7f0b27005111d393098baaaacae18c10fdb`
- Served model: `glm-5.2`
- API: private-LAN `http://192.168.0.184:8000/v1`

`benchmark.py` and `coding-benchmark.py` contain the complete measured-workload prompt bodies and token-accounting formula. All measured calls were sequential concurrency-1 requests with thinking disabled, temperature 0, streaming enabled, authoritative response usage, and `ignore_eos=true`.

## Performance result

Five discarded 256-token structured warm-ups preceded five measured 1,024-token structured-JSON runs. The discarded warm-up rows were not persisted; the result JSON contains the five measured rows:

- Decode median: **25.426501723394836 tok/s**
- Decode range: **25.36496694104505–25.46870580492912 tok/s**
- Client-observed TTFT median: **1,063.15141300729 ms**
- End-to-end output median: **24.793157330968437 tok/s**

Three measured 512-token TypeScript coding runs:

- Decode median: **20.389034911800554 tok/s**
- Decode range: **20.374664189452147–20.399190895583402 tok/s**
- Client-observed TTFT median: **1,020.8968929946423 ms**

Three measured 512-token high-entropy prose runs produced a 14.235503817158055 tok/s median. This is retained to show speculative decode workload sensitivity rather than hidden as an outlier.

Decode uses `(completion_tokens - 1) / (last_nonempty_content_time - first_nonempty_content_time)`. End-to-end output uses `completion_tokens / total_HTTP_duration`.

## GPU telemetry

Each CSV was collected over one persistent SSH session at approximately two-second intervals using:

```text
nvidia-smi --query-gpu=pstate,utilization.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader,nounits
```

Busy samples are those with utilization at least 50%:

| Node | Samples | Busy | P-state | Utilization busy min/median/max | SM MHz busy min/median/max | Power W busy min/median/max |
|---|---:|---:|---|---|---|---|
| dgx03 | 242 | 184 | P0 | 95 / 95 / 96 | 2437 / 2444 / 2457 | 43.77 / 46.81 / 60.25 |
| dgx04 | 242 | 185 | P0 | 93 / 95 / 96 | 2437 / 2450 / 2476 | 44.16 / 49.62 / 67.26 |
| gx10 | 242 | 184 | P0 | 93 / 96 / 96 | 2457 / 2476 / 2502 | 45.50 / 50.875 / 65.60 |

The CSVs are a separate GPU-load capture from the same acceptance session. Initial and final idle samples are intentionally retained. Because the measured result JSON does not include wall-clock timestamps, the telemetry is not mechanically correlated with individual result rows and is not presented as per-run telemetry.

## Functional acceptance

`acceptance.py` and `acceptance-results.json` record:

- direct semantic completion: `GLM52_SEMANTIC_OK`
- GLM tool call: `get_weather` with `{"city":"Berlin"}`
- Vision: correctly identified the committed 256×256 `vision-test.png` fixture as a white circle centered on a red square/background. `acceptance.py` reads `/tmp/glm52-vision-test.png`; copy the committed fixture to that path before reproducing the test.
- long-context semantic retrieval: 71,537 prompt tokens and exact recovery of `ALPHA_MARKER_7319|OMEGA_MARKER_2846`
- Anthropic-compatible `/v1/messages`: arithmetic result `4`

## Artifact checksums

- `benchmark.py`: `51152f06efb344570eb2c5c48c7fa8ca9b0b6b5675c88e90413199ae006f9cdd`
- `coding-benchmark.py`: `40e38f8aa9226b7faa2c1aba44d792eef6bb2bad070042f379e0f15aa3e45b16`
- `acceptance.py`: `c1d03f2469554ca753911cba3b830f8048c65dc271f3373a6f3adc06e1aa3e1c`
- `structured-and-prose-results.json`: `9de8d5c9d84fb8f0651745613302eabc541e90f9257b31cdf8660451ca2a8f1a`
- `coding-results.json`: `ba4628811c96ebf51612b8aa0957cb0d84fa2281fc2362939f80a1f2e27ac78a`
- `acceptance-results.json`: `93d4f3f54b641f4760aeb0d758af0cfd3149be87006bfe8c2e237a1623d8d5fc`
- `telemetry-dgx03.csv`: `427e198ac47cce406b5edf1d77599843cabe3d3829713c87bdc3ec3d9645ee01`
- `telemetry-dgx04.csv`: `090007180c162d884d614401aeda0d2392a0d6681ad725eb17517bad173d1451`
- `telemetry-gx10.csv`: `f9a813d2e8b2e8efac9cbd91554a8e8d6e3305b392d2563c57c36a4922473911`
- `vision-test.png`: `f9b7fd13f16c38b68115586c37401bf521ab174c78d2677d378217fb8990e89c`

## Acceptance boundary

The server is configured for 235,392 tokens with an approximately 240,640-token KV pool. The largest completed semantic request in this evidence is 71,537 prompt tokens; configured capacity is not represented as an empirically completed near-limit request. Kernel/runtime scan output and the separate LiteLLM/final direct probes are operational session evidence and are intentionally not claimed by this committed artifact set.
