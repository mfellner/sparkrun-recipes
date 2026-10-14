# Native Drama 3 on one GB10

Built with Fish Audio. This is a personal-evaluation deployment, not a public hosted production API. Fish source and checkpoint carry the Fish Audio Community License v1.0; the full license and required attribution are retained in `mods/drama3-gb10/`. Hosted services generally require a separate license. Only clone a real voice with consent or another legal right. Newly authored packaging/adapter/tests are MIT, not a relicensing of Fish materials. NVIDIA/dependency terms remain separate.

## Deployment

Review and trust this registry, then select exactly one idle GB10 host. The service requires an explicit RFC1918 IPv4 that belongs to that host. It rejects wildcard, IPv6, loopback, Tailscale and public bindings. Reserve the address in DHCP; a changed IP requires a reviewed relaunch. Do not expose it to untrusted clients or the Internet.

```sh
sparkrun registry add --trust https://github.com/mfellner/sparkrun-recipes.git
sparkrun registry update mfellner
sparkrun run @mfellner/fish-audio-drama3-dgx-spark \
  --hosts 192.168.178.70 -o host=192.168.178.70 \
  --transfer-mode delegated --no-follow --no-sync-tuning
```

Replace both addresses for another installation. SparkRun 0.3.6 uses `vllm-distributed` as the single-node lifecycle shim; no vLLM model/server is installed or executed. Native Fish's queue, FP8 MoE, hybrid attention, audio tower and CUDA DAC codec perform synthesis. There is no multi-node, TP/EP, NVFP4 or QAT claim. `--compile` is not supported by Drama 3.

Public ARM64 image: `ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@sha256:5da66fdde5e5efaa54abfc9b4c8bcf8517960d4a7b69b87b32e93e2b1440bdaa`.
This intentionally reuses the repository's existing public GHCR package as a
transport, under the distinct `drama3-gb10-20261014-lan-guarded` tag. Its legacy
package name does NOT mean that GLM or vLLM serves audio: the OCI title, version,
source pins and executable hashes identify the new native Drama runtime. The
separate new `drama3-gb10` package defaulted to private; GitHub's visibility API
returned 404 and no browser login was available. No legacy GLM tag was changed.
Anonymous manifest/config retrieval of the published digest is preserved in the
evidence. About 7.52 GB compressed image transfer is separate from model weights.

SparkRun distributes the immutable model snapshot including its codec. Alternatively, pre-stage that same snapshot on the controller and selected target with `python3 scripts/drama3/download.py`. The downloader uses two resumable ordinary HTTP workers, verifies every declared file size/SHA-256 and downloads no S2 weights. Allow about 42 GB for the snapshot plus image/compiler/output space. The launcher rehashes all snapshot files every startup and checks its packaged source manifest; a loader is not ready until a native GPU WAV warmup succeeds.

The documented `--transfer-mode delegated` (supported but hidden in SparkRun
0.3.6's short help) pulls/distributes on the explicit target rather than requiring
a second controller copy. It preserves sole-head capacity and avoids redundant
large image transfers. Do not rewrite a saved multi-node cluster for this solo job.

## Use

Browser: http://192.168.178.70:8010/ui

The default voice is the checksum-pinned public-domain LJ Speech LJ001-0001
reference (Keith Ito / Linda Johnson), bundled with attribution/provenance.
Omitting `references` or sending `[]` selects that reference, not an unconditioned
random voice. Initial native unconditioned English/German synthesis produced
decodable audio but failed full-text ASR; those diagnostic attempts are not passes.
An explicit legal reference replaces the public default. The selected profile
uses native decode CUDA graphs (`FISH_DRAMA3_CUDA_GRAPH=1`); readiness logs must
confirm capture, not just an environment variable.

```sh
curl --fail http://192.168.178.70:8010/health
curl --fail http://192.168.178.70:8010/v1/models
curl --fail --max-time 1800 http://192.168.178.70:8010/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Good morning. This is Drama three on a single Spark.","format":"wav","seed":42}' \
  -o morning.wav
curl --fail --max-time 1800 http://192.168.178.70:8010/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"This is the first sentence. This is the next sentence.","streaming":true}' \
  -o streaming.wav
ffplay -autoexit morning.wav
```

This is native `/v1/tts`, NOT OpenAI `/v1/audio/speech` or a chat API. `/v1/models` exists only for SparkRun discovery/readiness. No LiteLLM TTS route has been qualified. WAV is mono PCM16 at 44.1 kHz. Native streaming is per completed text chunk, not per codec frame; the WAV stream has an unknown-length header. First-audio timing excludes that header. Some clients need conversion to a finalized WAV before seeking.

Bounds: 4 MiB HTTP body, 1–2,000 text characters, WAV only, 1–1,024 max acoustic frames per text chunk, one optional inline reference, 0.5–30 seconds mono/stereo reference WAV at 8–96 kHz. The default is 1,024 frames per chunk (the upstream SDK default); the initial 600-frame streaming probe hit its frame cap and is not an acceptance pass. A cap can truncate speech; inspect the ending and use bounded appropriate controls. Drama 3 uses its native approximately 300-byte text splitter; the legacy `chunk_length` field does not change that splitter. One active GPU request and two queued requests are allowed; excess requests return 429. Requests serialize, including references and RNG state. A disconnected client does not cancel native work: the bounded generation drains before the next one, preventing cache cross-talk. Do not treat client timeout as success.

Reference cloning accepts exactly one `references` element with base64 WAV `audio`, `text` transcript and `consent:true` affirming consent or another legal right. No URL/path/reference-ID fetching, persistent upload, arbitrary codec endpoint or public audio-file route is exposed. Inline reference caches are cleared after each request. A public-domain LJ Speech sample is used by the acceptance harness; no private reference recording is published. Generated WAVs and request-local measurement receipts persist under the target's `~/.cache/huggingface/drama3/outputs/`; they are not HTTP-served. Manage this scoped output directory's retention as needed.

Runnable public-reference example from the repository root (use only a recording
you have consent or another legal right to clone):

```sh
python3 - <<'PY'
import base64, json, pathlib, urllib.request
root = pathlib.Path('mods/drama3-gb10')
payload = {'text': 'This new sentence uses a public domain reference recording.',
           'references': [{'audio': base64.b64encode((root/'default-reference.wav').read_bytes()).decode(),
                           'text': (root/'default-reference.txt').read_text().strip(),
                           'consent': True}]}
request = urllib.request.Request('http://192.168.178.70:8010/v1/tts',
    json.dumps(payload).encode(), {'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=1800) as response:
    pathlib.Path('reference-demo.wav').write_bytes(response.read())
PY
```

The base64 reference is additionally capped at 3,000,000 characters; duration,
sample-rate, channel and HTTP-size bounds must all be satisfied together.

## Lifecycle, logs and persistence

```sh
sparkrun status --hosts 192.168.178.70 --json
sparkrun logs WORKLOAD_ID --hosts 192.168.178.70
sparkrun stop WORKLOAD_ID --hosts 192.168.178.70
# Restart by repeating the reviewed run command above after stop.
```

Never use `sparkrun stop --dry-run` with 0.3.6; it can actually stop the workload. SparkRun's solo container uses `sleep infinity` plus a detached serve process: `docker logs` may be empty; read `/tmp/sparkrun_serve.log` using `docker exec CONTAINER cat /tmp/sparkrun_serve.log`. Containers are retained (`auto_remove:false`) for diagnosis, not a second supervisor. Model, CUDA/Triton compiler caches and generated outputs are persisted in the mounted Hugging Face cache. CUDA graph objects are rebuilt on each process start; they are not serialized compiler-cache artifacts. There is no automatic host-reboot/process-crash restart promise: run the recipe again after checking status and the port. The accepted service is intentionally left running; no cron, gateway or host-wide changes are installed.

The reviewed recipe requests UID/GID 1000, non-privileged Docker, host networking, private IPC, 2 GiB shared memory and no-new-privileges. Re-inspect effective executor mounts/capabilities after SparkRun upgrades. The API is unauthenticated; explicit LAN binding is not user authentication or a firewall. Read-only users of the same trusted LAN can submit synthesis and consume resources. Host cache access is granted to the container. No model weights enter the image.

POST requires `Content-Type: application/json`. Browser requests with an `Origin`
must match the explicit IP/port origin; cross-origin and browser-simple text/form
requests are rejected before reading the body or touching the GPU. This prevents
ordinary drive-by browser synthesis, not an authenticated-client boundary. Use the
documented IP URL for the UI, not an unqualified DNS alias/proxy origin.

Memory receipts report Torch's high-water allocation since native Fish's last
per-chunk reset, plus one-second host-memory samples. They are not an exact
whole-request/system-memory peak; GB10 shares physical memory with the CPU.

## Reproduce and verify

Build from `mods/drama3-gb10/` on ARM64. The Dockerfile pins its public CUDA base digest, exact Fish source, and the complete package inventory. PyTorch 2.13/cu130 and Triton 3.7.1 replace upstream Torch 2.8. Torchaudio 2.13 ARM wheels are unavailable; the qualified Python resampler uses Torchaudio 2.11/cu130 with Torch 2.13. This deliberately overrides its package metadata requirement; general binary-extension compatibility is NOT claimed. `descript-audiotools`' obsolete protobuf constraint is overridden with protobuf 5.29.5 and TensorBoard 2.18, avoiding latest TensorBoard's protobuf-6-generated-code mismatch. CuDNN remains enabled and the GPU codec is not bypassed. Native reference WAV decoding uses soundfile rather than removed Torchaudio backend arguments.

```sh
docker build -t drama3-gb10:local mods/drama3-gb10
docker run --rm --gpus all --entrypoint python drama3-gb10:local /opt/drama/probe.py
uv run --with pytest --with pyyaml --with httpx --with detect-secrets \
  --with numpy --with soundfile pytest -q
uv run --with pyyaml --with numpy --with soundfile python scripts/drama3/verify_evidence.py \
  evidence/drama3-gb10-20261014
```

Evidence includes canonical requests, response status/headers, playable decoded WAVs, actual pinned CPU Whisper ASR, timings, overlapping-request intervals, target telemetry, runtime file hashes, listener/PID/start identity, source/model/image pins, startup logs and repeated bounded kernel reads. ASR consistency and decoded non-silence do not prove general naturalness, emotion accuracy or speaker similarity. No H200/S2 numbers are transferred to GB10.

### Measured acceptance scope

The final immutable image completed eleven real native GPU requests: repeated
English/German examples, an expression-tagged sentence, longer text, incremental
streaming, two overlapping (GPU-serialized) requests, an explicit public reference,
and post-restart synthesis. Pinned CPU Whisper passed every full-text/tail check;
the maximum normalized WER was 0.0625. The reference input is preserved separately
as `reference-input.wav`; `reference.wav` is the generated reference-case output.
An earlier collector filename collision invalidated its reference digest; the
collector was corrected, regression-tested, and the entire final suite recollected.

For these fixtures only, median serial HTTP wall/audio RTF was 1.1695 (lower is
better; this is slower than real time). The streaming case received first PCM at
18.8839 seconds, finished HTTP at 37.7097 seconds, and decoded 32.8795 seconds of
audio. Overlapping-client RTF includes queue wait. Torch's reported high-water
allocation reached about 43.25 GiB under its upstream per-chunk reset semantics;
one-second samples retained at least 69.90 GiB host `MemAvailable`. These are
observations of this profile/fixture set, not broad latency or throughput guarantees.
The browser generated and decoded a playable WAV; nobody claimed to listen to it.

The same immutable image was stopped and relaunched through SparkRun. A distinct
container/process/start identity, matching compiler-cache file digests, and an
unchanged persisted pre-restart output prove the scoped restart/persistence checks.
The model files were independently rehashed in the final live container. Head
gateway/sparkDash start identities predate the mission baseline; sparkDash's HTTP
port was already unavailable at baseline and is not reported as repaired.

See `evidence/drama3-gb10-20261014/summary.json`, `manifest.json` and the verifier for
the mechanically joined receipts. Unconditioned synthesis, subjective naturalness,
emotion accuracy, speaker-similarity scoring, OpenAI/LiteLLM TTS compatibility and
automatic recovery after host reboot/process crash remain unqualified.
