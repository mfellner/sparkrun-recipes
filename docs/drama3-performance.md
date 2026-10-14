# Drama 3 native GB10 speed profile

Built with Fish Audio. Personal trusted-LAN evaluation only. Fish source/checkpoint remain under Fish Audio Community License v1.0; new wrapper/packaging is MIT. Review hosted-service restrictions and use real voices only with consent or another legal right.

## What changed

The compact BF16 output head projects the 4096 semantic candidates plus EOS instead of all252288 vocabulary rows. Retained weights, native sampling/RAS, FP8 experts, BF16 dense/codebook layers, original BF16 GPU codec and public-domain LJ Speech default reference are unchanged. The source/checkpoint/reference/image are pinned. No new quantization, recurrent/expert kernels, prefix caching, smaller buckets or fine-grained streaming are admitted.

Original recipe `@mfellner/fish-audio-drama3-dgx-spark` remains available for rollback. This distinct profile is `@mfellner/fish-audio-drama3-fast-dgx-spark`.

## Launch and use

After reviewing and trusting the registry:

    sparkrun registry add --trust https://github.com/mfellner/sparkrun-recipes.git
    sparkrun registry update mfellner
    sparkrun run @mfellner/fish-audio-drama3-fast-dgx-spark \
      --hosts 192.168.178.70 -o host=192.168.178.70 \
      --transfer-mode delegated --no-follow --no-sync-tuning

Use an idle single GB10; replace both addresses with that host's explicit RFC1918 IPv4. Stop the current workload before replacing it. Never use `sparkrun stop --dry-run` on0.3.6. The API is unauthenticated and must stay on a trusted LAN, not the Internet. `vllm` is only SparkRun's Docker/lifecycle shim; native Fish serves audio.

UI: http://192.168.178.70:8010/ui
Native API: POST /v1/tts. Configuration/source identities: GET /v1/runtime-profile. Original body/text/frame/reference/consent/origin/admission rules are retained; see `docs/drama3-gb10.md`. No OpenAI audio or LiteLLM TTS compatibility claim.

    curl --fail http://192.168.178.70:8010/health
    curl --fail --max-time 1800 http://192.168.178.70:8010/v1/tts \
      -H 'Content-Type: application/json' \
      -d '{"text":"Good morning. This is Drama three on a single Spark.","seed":42}' \
      -o morning.wav

WAV is mono PCM16 at44100Hz. Streaming remains delivery of completed native text chunks; the existing browser waits for a finalized Blob. Neither is per-frame interactive playback.

## Measured scope

`evidence/drama3-gb10-performance/` preserves actual raw HTTP/WAV, ASR, per-request transport timing, source/runtime/recipe/image identity, target telemetry and logs. Both arms used the same final immutable image on dgx02/GB10, with head off/on and otherwise matching settings.

- Six fixtures: English1/2, German1, expression tag, longer text and completed-chunk streaming.
- Seed42, temperature0.7/top_p0.7, original frame limits/codec/reference. One warmup and five measured repetitions per fixture:30 measured+6warmups per arm.
- Median end-to-end wall/audio RTF: **1.2008836472010322 ->1.0881797649421394**, a **9.385079272382313% reduction**.
- All36 matched WAVs are byte-identical. All36 per-arm full-text/endings ASR checks passed; this is corpus-scoped, not universal byte identity or speaker/emotion/naturalness proof.
- Streaming median first PCM: **19.363558234996162s ->17.343282055997406s**. A WAV header is excluded.
- Ten extra final-image cases cover German2, longer/tagged/default/reference speech and overlapping GPU-serialized requests; all ASR passed. Two postrestart cases also passed.
- Ten HTTP security negatives passed. Direct fabric and target loopback probes did not reach port8010. This is bounded tested exposure, not client authentication or a universal firewall claim.
- Conservative boot-through-acceptance kernel reads were repeated byte-for-byte at collection with cursors and contain no matched Xid/OOM/NV_ERR_NO_MEMORY events in the published window. A later independent bounded requery preserves every original event byte; its journal cursor differs and is retained unmodified. The verifier pins both original complete captures to reviewed digests and requires full event equality with the later requery.
- Real SparkRun stop/relaunch produced a distinct container/process and fresh speech. A prerelease output survived unchanged. Postrestart compiler artifacts are present; no claim is made that graph objects are serialized or that all cache files were compared pre/post.

RTF remains above1: this profile is not faster than playback, and the medianRTF<=0.9 / firstPCM<2s goals are not attained. Earlier baseline multi-seed speech failures remain known; the published performance qualification is limited to this passed corpus/settings. Do not describe it as qualified unconditioned synthesis or general naturalness/style accuracy.

## Why other experiments are disabled

Convolution, prefix reuse, smaller graph buckets, Omni and alternative quantization/recurrent kernels remain excluded. Historical research dispositions led to that selection, but their individual gains and failures are not reproduced or validated by this selected release export. The published evidence here qualifies only the compact-head profile. Native FP32 streaming is a separate qualification effort, not included in this speed profile.

## Reproduce, validate and roll back

    docker build -f mods/drama3-gb10-release/Dockerfile -t drama3-fast:local .
    uv run --with pytest --with pyyaml --with numpy --with soundfile \
      --with httpx --with detect-secrets pytest -q
    uv run --with pyyaml --with numpy --with soundfile python \
      scripts/drama3/verify_fast_evidence.py evidence/drama3-gb10-performance

The verifier is specific to this immutable empirical snapshot, not a generic qualification service. `captured-artifacts.json` freezes the actual collected bytes after collection; its reviewed digest is pinned in verifier source. It excludes publication bindings and verifier source to avoid checksum cycles. This integrity anchor is not an independent measurement and does not replace the semantic request/source/quality checks. Tests separately exercise semantic mutations with an explicitly authorized test-only root and verify that rewriting the production capture map is rejected.

The verifier recomputes request/audio/ASR/RTF/count/source/runtime/log joins and rejects refreshed-manifest semantic mutations. Publication bindings were assembled after collection; each raw batch schedule predates its requests. The independently reviewed committed tree is the external trust root. Recorded capture argv/path identities are historical facts; offline verification does not need those task directories or live hosts.

Public ARM64 image:
`ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@sha256:eb5f38cb8c242dae2162c0618dddae101bf6ed386feb3fdcf0c38971120e511c`.
The legacy package name is artifact transport only; labels identify Drama3 and no old GLM tag was replaced. Anonymous manifest/config access was verified. Weights are not in the image; reuse/distribute the immutable `dd64a85cd82e9521f7202610ede97b367fc83946` snapshot.

Rollback: stop the current workload through SparkRun, confirm no serving process/listener remains, then launch the original namespaced recipe with the same explicit host/binding. Output/cache paths persist. SparkRun's detached serve/sleep supervisor is not automatic process-crash or reboot recovery; do not advertise an uptime guarantee.
