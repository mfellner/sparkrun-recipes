# Selected immutable native speed-release evidence

Actual same-image head-off/head-on GB10 inference, not estimated payload speedups.
Read summary.json and docs/drama3-performance.md for exact scope and unmet goals.

- final-control/ and final-candidate/: six batches, each warm1+measured5, raw WAV/HTTP/ASR/transport events, exact runtime before/after, recipe/harness, telemetry and full source/log bindings.
- final-acceptance/: ten final-image cases; reference-input.wav is the public source, reference.wav is generated output. The first collector invocation lacked the source fixture and stopped before its reference request. The fixture was then staged and only the remaining reference/overlap cases collected; earlier seven outputs were not overwritten. This operator setup failure is not a successful complete first invocation.
- final-post-restart/: fresh speech/streaming, launch-persisted job state, new runtime identity, output persistence and compiler artifact presence. No pre/post whole compiler-cache equality claim.
- http-negative.json: ten security negatives and bounded unintended-interface probes.
- public-image.json: independent anonymous registry manifest/config readback.
- foreground_batch.py: exact operator wrapper used, including bounded joined telemetry. Its recorded task paths are historical; the offline verifier uses committed relative inputs.
- publication-bindings.json: file/source trust bindings assembled after collection. Raw per-batch schedules were written before their requests; the independently approved Git tree anchors these bindings.

All claimed waveform/performance results come from these raw inputs. Profile-only flags are not kernel qualification, ASR is not subjective listening/speaker similarity, and RTF>1/firstPCM17s does not meet realtime or interactive latency targets. `scripts/drama3/verify_fast_evidence.py` and adversarial tests fail closed on missing/altered/misrouted/timestamp-shifted evidence even when hashes are refreshed.
