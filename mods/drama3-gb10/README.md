# Drama 3 GB10 runtime mod

Built with Fish Audio. Native Fish at `1ff3c3ed2f80d3106a83909dd25b83129c3d816f`, model at `dd64a85cd82e9521f7202610ede97b367fc83946` (weights external). See [deployment and evidence boundaries](../../docs/drama3-gb10.md).

`Dockerfile` is ARM64 and pins the public CUDA base by digest. `requirements-runtime.txt` lists direct native consumers; `requirements-lock.txt` records the entire selected package inventory. Intentional Torch/Torchaudio/protobuf metadata overrides are in `NOTICE`. No generic vLLM model serving or upstream package resolver downgrade is used. `probe.py` executes actual Fish FP8 MoE/GDN kernels, cuDNN Conv1d and CUDA graph replay before model staging. The server rechecks all immutable model bytes and its packaged manifest at launch.

The separately authored adapter exposes only bounded WAV synthesis, health/models and a small UI. Requests serialize. It uses a checksum-pinned public-domain LJ Speech reference by default because unconditioned output failed full-text ASR; it does not claim unconditional voice quality. Public reference attribution/source are in `reference-provenance.json`. Explicit inline references require consent/legal-right affirmation and never remain in the in-memory reference cache after a request.

`manifest.sha256` covers executable/build/source/config/license/reference inputs, excluding this non-runtime README and the manifest itself. Update it and rebuild after runtime changes; all live evidence then needs requalification. The image source license is `FISH_LICENSE.txt`; MIT `ADAPTER_LICENSE.txt` covers only newly authored adapter/packaging/tests. Do not use this package as a monetized hosted speech-generation service without the required Fish license.
