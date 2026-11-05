# Current live GLM-5.3 TP2 evidence (2026-11-05)

This is an addendum to the static/compatibility package one directory up. It captures a short, read-only snapshot of the running SparkRun job and performs small semantic API checks. It is not a long stress test or full acceptance campaign.

## Bound identity

- Durable SparkRun group: `sparkrun_58201f8c57f8af18_f50636d19f88`, cluster `spark-pair2`; persisted job/run metadata and effective recipe snapshot are in `metadata/`.
- Head/rank 0: `dgx01` (`192.168.178.47`); worker/rank 1: `dgx02` (`192.168.178.70`). Both containers were observed running under root, with matching local Docker image IDs and matching recipe-stamp labels. Exact per-host container/process/overlay data is in `hosts/`.
- Image reference/digest, source/model/draft revisions, launch time, and TP/PP configuration are recorded in `manifest.json` and the raw metadata files.
- The live process argv sets `--max-model-len 850000`. This is configured/advertised only. **No 850k-token prompt or context-capacity test was performed.**

## Post-ready/API checks

The serving logs and `/v1/models` showed the expected served name and configured maximum context; `/health` returned HTTP 200. The small deterministic chat smoke (`api/smoke.py`) passed serially and at three concurrent requests for both direct head API and LAN proxy: 4/4 per path. Raw output includes exact response content, HTTP result, timing and token usage. These prove only short requests, not capacity or endurance.

## Bounded logs and caution

- `metadata/runtime-logs.txt` is a bounded 500-line multi-rank SparkRun log capture.
- `hosts/rank0.txt` and `hosts/rank1.txt` include container/process/overlay identities and kernel journal matches in the fixed host-local window `2026-11-05 21:21:00`–`21:55:00`.
- The head host recorded a memory-cgroup OOM kill at 21:46:45. Its cgroup ID differs from the current rank-0 container ID, so this evidence does not attribute that kill to the GLM job; nevertheless it prevents describing the full interval as incident-free. Rank containers were still running at collection, and API checks passed afterward.
- Container/runtime settings are root-user and host-networked; the API and proxy are unauthenticated HTTP. Keep exposure to a trusted, firewalled private LAN.

## Verification

Run `python3 verify.py` from this directory. It verifies every payload against the live checksum index; checks persisted job, model, image, TP/PP and serving-name identity; and enforces fixed direct/proxy URLs, one serial plus three concurrent result records per route, expected indices, HTTP success and the recorded response text markers. The original request bodies were not captured, so request controls and exact prompt provenance are not independently verifiable. The checksum index is itself bound by the parent package checksum file. The checks establish only consistency with the documented capture and package hashes; they do not independently authenticate the collector or the originating machine. `SHA256SUMS` contains hashes for this subdirectory's evidence files (excluding itself). This verifier does not claim full acceptance, clean host health, absence of unrelated incidents, or 850k context qualification.
