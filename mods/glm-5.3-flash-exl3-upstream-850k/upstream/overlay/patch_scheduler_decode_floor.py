#!/usr/bin/env python3
"""Mixed-prefill policy: skip / cap / off / fair (issue #6).

A decode lane on this backend needs ~8 tokens (1 + DFlash2 k=7). The leftover
MNBT budget otherwise goes to a peer FLASHINFER_MLA_SPARSE_SM120 prefill
chunk. Mixed execution leaves the uniform decode FULL-graph path, and a
128-token cap is still ~10 tok/s at 80k KV. Token caps are not a millisecond
budget.

GLM53_MIXED_PREFILL_CHUNK:
  skip / -1  — do not mix prefill with decode. Starves every
               waiting/running prefill while any peer is decoding; there is
               no age limit. Solo prefill is unchanged.
  N>0        — cap mixed prefill chunks to N tokens while a peer decodes.
               The cap is fed into hybrid Mamba alignment so N < block_size
               still makes sub-block progress. 128 still stalls ~10 tok/s.
  0 / off    — disable the extra isolation policy.
  fair       — service-time mixing (default on TP=2/3/4 since 2026-09-15,
               v5). Decode-only
               steps between prefill turns; at most
               GLM53_FAIR_PREFILL_MAX_CHUNKS chunks per turn (default 1).
               Only prefill that contends with a decoder is charged (solo
               prefill is cost-sampled, not debt). Credit accrues at SHARE
               of accounted engine time. v5 fits a fixed-plus-per-token step
               cost from solo and mixed samples, targets the largest ladder
               rung (128..2048) whose estimated step fits
               GLM53_FAIR_PREFILL_MAX_STEP_MS, and saves credit for that
               rung instead of spending it on small chunks (v4 priced 1024
               tokens off 128-token samples linearly, ~3x too high, then
               stalled at 128-token steps: ~70 tok/s at 20% share). A
               never-served newcomer gets one prompt step-bounded probe;
               afterwards a 2s age override may borrow one such chunk, only
               after all shared debt is repaid. In-flight async prefill
               blocks the next mixed turn. Prefills are selected by last
               completed positive prefill service plus round-robin. Decode
               token/input budget is allocated first by the base scheduler.
               v7: under scheduling-policy=priority, lower numeric priority
               ranks first (FCFS ignores priority). Selection is
               allocation-aware: a runnable prefill already passed when a
               later selected candidate is natively refused KV allocation is
               carried to the next step within its tier. A carried lower-tier
               request is pre-selected ahead of a higher tier only when every
               strictly-higher-tier candidate was natively refused in the
               previous step (one step per refusal); new or unrefused
               higher-tier work always ranks first. DROP-5d: preselection is
               refusal-licensed (a candidate refused in the previous step is
               skipped once); carry is a trace annotation only, created on a
               native refusal and discarded on every release. DROP-5f: under
               the priority policy a candidate whose chosen rung exceeds the
               credit takes the largest affordable rung when a lower tier is
               present, and a lower tier never borrows in a step where a
               higher tier was credit/gap-deferred.
               Solo prefill retains the base scheduler's limits. Timing is a
               host busy-time proxy.

Fair knobs (read at runtime; identical on every rank):
  GLM53_FAIR_PREFILL_CHUNK            default 256 (probe size until timing samples exist)
  GLM53_FAIR_PREFILL_SHARE            default 0.30 (credit accrual fraction)
  GLM53_FAIR_PREFILL_MAX_INTERVAL_MS  default 2000
  GLM53_FAIR_PREFILL_MAX_STEP_MS      default 2000 (estimated mixed-step limit)
  GLM53_FAIR_PREFILL_MAX_CHUNKS       default 1

Versioned installer: `# [glm53-decode-floor:v7]`. Accepted inputs are
authenticated by exact helper bytes (sha256 + length) and exact hook anchors:
pristine, v1, v2, historical v5, current-main v5, priority v5 (#221), warm/
deadline v6 (#180/#80) and carry v6 (#246). v3/v4 and any unknown, drifted,
duplicated or marker-only input are refused without writing.
GLM53_FAIR_TRACE=<path> or SCHED_FAIR_TRACE=<path> (default off) appends JSONL
scheduler events; the latter passes through the launcher's GLM53_EXTRA_ENV.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
import time
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-decode-floor]"
MARK_V2 = "# [glm53-decode-floor:v2]"
MARK_V3 = "# [glm53-decode-floor:v3]"
MARK_V4 = "# [glm53-decode-floor:v4]"
MARK_V5 = "# [glm53-decode-floor:v5]"
MARK_V6 = "# [glm53-decode-floor:v6]"
MARK_V7 = "# [glm53-decode-floor:v7]"

IMPORT_OLD = """import itertools
import time
"""
IMPORT_NEW = """import itertools
import os
import time
"""

# v1 helper + insertions (recipe f906ee9 / this overlay before v2).
V1_HELPER_START = "def _glm53_mixed_prefill_policy(running, current):"
V1_RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
            if mixed_cap is not None and request.num_computed_tokens < request.num_prompt_tokens:
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""
V1_WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
                    if mixed_cap is not None and num_computed_tokens < request.num_prompt_tokens:
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""

# Frozen v2 insertions — used only to unpatch an already-v2 scheduler.
V2_BEGIN_NEW = """        self.current_step += 1
        _GLM53_MIXED.begin_step(self)  # [glm53-decode-floor:v2]
        # NOTE(woosuk) on the scheduling algorithm:
"""
V2_OBS_NEW = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        _GLM53_MIXED.observe_output(self, scheduler_output)  # [glm53-decode-floor:v2]
        pooler_outputs = model_runner_output.pooler_output
"""
V2_RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v2]
            if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""
V2_WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v2]
                    if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""
V2_ALIGN_NEW = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            _align_cap = getattr(self, "_glm53_align_prefill_limit", None)  # [glm53-decode-floor:v2]
            if _align_cap is not None and _align_cap > 0:
                max_prefill_tokens = min(max_prefill_tokens, _align_cap)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""
V2_RUNNING_MAMBA_NEW = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
            _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v2]
"""
V2_WAITING_MAMBA_NEW = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v2]
                        if num_new_tokens == 0:
                            break
"""



# Frozen v3 anchors for migration from the reviewed implementation.
V3_BEGIN_NEW = """        self.current_step += 1
        _GLM53_MIXED.begin_step(self)  # [glm53-decode-floor:v3]
        # NOTE(woosuk) on the scheduling algorithm:
"""
V3_OBS_NEW = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        _GLM53_MIXED.observe_output(self, scheduler_output)  # [glm53-decode-floor:v3]
        pooler_outputs = model_runner_output.pooler_output
"""
V3_FIN_OLD = """        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
"""
V3_FIN_NEW = """        # Check if the scheduling constraints are satisfied.
        _GLM53_MIXED.note_schedule_output(self, num_scheduled_tokens)  # [glm53-decode-floor:v3]
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
"""
V3_RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v3]
            if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                num_new_tokens = min(num_new_tokens, mixed_cap)
            num_new_tokens = _GLM53_MIXED.clip_for_decode_reserve(
                num_new_tokens,
                token_budget,
                int(getattr(self, "_glm53_decode_reserve_tokens", 0) or 0),
                _GLM53_MIXED.needs_prefill_compute(request),
            )

            # Make sure the input position does not exceed the max model len.
"""
V3_WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v3]
                    if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)
                    num_new_tokens = _GLM53_MIXED.clip_for_decode_reserve(
                        num_new_tokens,
                        token_budget,
                        int(getattr(self, "_glm53_decode_reserve_tokens", 0) or 0),
                        _GLM53_MIXED.needs_prefill_compute(request),
                    )
                    if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                        if num_new_tokens <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                    # chunked prefill has to be enabled explicitly to allow
"""
V3_ALIGN_NEW = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            _align_cap = getattr(self, "_glm53_align_prefill_limit", None)  # [glm53-decode-floor:v3]
            if _align_cap is not None and _align_cap > 0:
                max_prefill_tokens = min(max_prefill_tokens, _align_cap)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""
V3_RUNNING_MAMBA_NEW = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
            _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v3]
"""
V3_WAITING_MAMBA_NEW = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v3]
                        if num_new_tokens == 0:
                            break
"""

class _Glm53MixedPrefill:  # [glm53-decode-floor:v7]
    """Bound contention using completion feedback, without synchronizing GPUs."""

    LADDER = (128, 256, 512, 768, 1024, 1536, 2048)
    COLD_TOK_S = 1300.0
    FIT_WINDOW = 24

    def __init__(self, now=None):
        self._now = now or time.monotonic
        self.mode = "skip"
        self.legacy_cap = 0
        self.logged_boot = False
        self.hist_every = 50
        self._parse()
        self.last_service = {}
        self.arrival = {}
        self.rr_seq = {}
        self.served_tokens = {}
        self.rr_n = 0
        self.steps = 0
        self.credit = 0.0
        self.in_contention = False
        self.inflight = {}
        self.mixed_samples = []
        self.solo_samples = []
        self.last_account_mono = None
        self.last_prefill_turn_mono = 0.0
        self.step_tag = None
        self._sched_id = None
        self._open_rec = None
        self.selected = set()
        self._candidates = []
        self._tried = set()
        self._passed = set()
        self._carry = set()
        self._refused = set()
        self._refused_prev = set()
        self._deferred_tiers = set()
        self.priority_mode = False
        self.refusals = 0
        self.step_mode = "solo"
        self.defer_reason = "none"
        self.missed_prefill = 0
        self._model_cache = None
        self._incarn = {}
        self._trace = None
        self.trace_seq = 0
        self.trace_dropped = 0
        # SCHED_FAIR_TRACE is the launcher-forwardable alias: start.sh reserves
        # every GLM53_* name in GLM53_EXTRA_ENV.
        self.trace_path = (os.environ.get("GLM53_FAIR_TRACE", "").strip()
                           or os.environ.get("SCHED_FAIR_TRACE", "").strip())
        if self.trace_path:
            try:
                self._trace = open(self.trace_path, "a", buffering=1 << 16)
            except OSError:
                self.trace_dropped += 1

    def _e(self, name, default):
        v = os.environ.get(name)
        return default if v is None or not str(v).strip() else str(v).strip()

    def _parse(self) -> None:
        raw = self._e("GLM53_MIXED_PREFILL_CHUNK", "skip").strip().lower()
        self.legacy_cap = 0
        if raw in ("0", "off", "no"):
            self.mode = "off"
        elif raw in ("skip", "-1"):
            self.mode = "skip"
        elif raw == "fair":
            self.mode = "fair"
        else:
            try:
                cap = int(raw)
            except ValueError:
                cap = 0
            if cap <= 0:
                self.mode = "off"
            else:
                self.mode = "cap"
                self.legacy_cap = cap
        try:
            self.chunk = int(self._e("GLM53_FAIR_PREFILL_CHUNK", "256"))
        except ValueError:
            self.chunk = 256
        if self.chunk <= 0:
            self.chunk = 256
        try:
            self.share = float(self._e("GLM53_FAIR_PREFILL_SHARE", "0.30"))
        except ValueError:
            self.share = 0.30
        self.share = min(1.0, max(0.0, self.share))
        try:
            self.interval_s = int(self._e("GLM53_FAIR_PREFILL_MAX_INTERVAL_MS", "2000")) / 1000.0
        except ValueError:
            self.interval_s = 2.0
        if self.interval_s <= 0:
            self.interval_s = 2.0
        try:
            self.max_chunks = int(self._e("GLM53_FAIR_PREFILL_MAX_CHUNKS", "1"))
        except ValueError:
            self.max_chunks = 1
        self.max_chunks = max(1, min(self.max_chunks, 16))
        try:
            self.max_step_s = max(0.001, int(self._e("GLM53_FAIR_PREFILL_MAX_STEP_MS", "2000")) / 1000.0)
        except ValueError:
            self.max_step_s = 2.0
        if self.mode == "fair" and not self.logged_boot:
            print(
                f"[glm53-decode-floor] fair v7 probe_chunk={self.chunk} "
                f"ladder={min(self.LADDER)}..{max(self.LADDER)} share={self.share} "
                f"interval_s={self.interval_s} max_step_s={self.max_step_s} "
                f"max_chunks={self.max_chunks}",
                flush=True,
            )
            self.logged_boot = True


    @staticmethod
    def prefill_remaining(request, computed=None):
        prompt = int(getattr(request, "num_prompt_tokens", 0) or 0)
        if computed is None:
            computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        tokens = int(getattr(request, "num_tokens", prompt) or prompt)
        return max(0, max(prompt, tokens - 1) - int(computed))

    def needs_prefill_compute(self, request, computed=None):
        return self.prefill_remaining(request, computed) > 0

    def _iter_waiting(self, sched):
        for name in ("waiting", "skipped_waiting"):
            q = getattr(sched, name, None)
            if not q:
                continue
            try:
                for r in q:
                    yield r
            except TypeError:
                continue

    def _live_ids(self, sched):
        ids = set()
        for r in list(getattr(sched, "running", None) or []):
            rid = getattr(r, "request_id", None)
            if rid is not None:
                ids.add(rid)
        for r in self._iter_waiting(sched):
            rid = getattr(r, "request_id", None)
            if rid is not None:
                ids.add(rid)
        reqs = getattr(sched, "requests", None)
        if isinstance(reqs, dict):
            ids.update(reqs.keys())
        return ids

    def _prune(self, live):
        for store in (self.last_service, self.arrival, self.rr_seq, self.served_tokens):
            dead = [k for k in store if k not in live]
            for k in dead:
                store.pop(k, None)
        for bag in (self._carry, self._refused, self._refused_prev, self._passed):
            bag.intersection_update(live)
        for rid in [k for k in self._incarn if k not in live]:
            self._incarn.pop(rid, None)


    @property
    def inflight_prefill(self):
        return sum(bool(r["prefill_tokens"]) for r in self.inflight.values())

    def _shape(self, decodes, prefills):
        history = max((int(r.num_computed_tokens) for r in decodes), default=0)
        position = max((int(r.num_computed_tokens) for r in prefills), default=0)
        return (len(decodes), (history // 4096).bit_length(),
                (position // 4096).bit_length())

    def _cost_model(self):
        """Fit step cost dt = a + b*n over recent prefill-bearing steps.

        Every prefill-bearing step pays a fixed cost on this kit (~0.3 s host
        time for 82..256 tokens, ~2.7 s for 3584), so a mixed step is close to
        a solo chunk plus a few decode rows: solo samples are pooled for the
        fit. The fit is then scaled so recent mixed samples are not
        underestimated (75th percentile of actual/fit, clamped to [1, 1.5]).
        Needs two distinct chunk sizes; otherwise returns None and the caller
        scales linearly. v4 scaled one sample linearly, priced 1024 tokens off
        128-token samples at ~3x the real cost, and never climbed back.
        """
        if self._model_cache is not None:
            return self._model_cache
        mixed = [(n, dt) for _, n, dt in self.mixed_samples[-self.FIT_WINDOW:]]
        pts = mixed + [(n, dt) for n, dt in self.solo_samples[-self.FIT_WINDOW:]]
        if len(pts) < 2 or len({n for n, _ in pts}) < 2:
            return None
        cnt = float(len(pts))
        sx = float(sum(n for n, _ in pts))
        sy = float(sum(dt for _, dt in pts))
        sxx = float(sum(n * n for n, _ in pts))
        sxy = float(sum(n * dt for n, dt in pts))
        den = cnt * sxx - sx * sx
        b = max(0.0, (cnt * sxy - sx * sy) / den) if den > 0 else 0.0
        a = max(0.0, (sy - b * sx) / cnt)
        if mixed:
            ratios = sorted(dt / max(1e-6, a + b * n) for n, dt in mixed)
            r = ratios[min(len(ratios) - 1, int(0.75 * len(ratios)))]
        else:
            r = 1.1  # decode rows not sampled yet
        r = min(1.5, max(1.0, r))
        self._model_cache = (a * r, b * r)
        return self._model_cache

    def _est_dt(self, tokens, shape=None):
        """Conservative host-time estimate; not a guaranteed execution bound."""
        n = max(1, int(tokens))
        model = self._cost_model()
        if model is not None:
            a, b = model
            return max(0.01, a + b * n)
        samples = self.mixed_samples[-8:]
        if samples:
            # One size only: scale linearly with a fixed-cost floor.
            return max(0.01, max(dt * max(0.5, n / t) for _, t, dt in samples))
        return max(0.05, n / self.COLD_TOK_S)

    def _target(self, remaining, room):
        """Largest rung whose estimated step fits `room` (tokens per accounted
        second rise with size under a fixed per-step cost), or None."""
        rungs = {self.chunk}
        if self.mixed_samples or self._cost_model() is not None:
            rungs.update(self.LADDER)
        if remaining is not None:
            rungs = {min(n, remaining) for n in rungs}
        fitting = [(n, self._est_dt(n)) for n in sorted(rungs)]
        fitting = [(n, cost) for n, cost in fitting if cost <= room + 1e-9]
        return fitting[-1] if fitting else None

    def _credit_limit(self):
        return min(self.max_step_s, self._est_dt(max(self.chunk, max(self.LADDER))))

    def _ev(self, kind, step=None, **fields):
        if not self.trace_path:
            return
        import json  # local: the scheduler module does not import json
        self.trace_seq += 1
        if step is None:
            step = (self._open_rec or {}).get("step_id")
        fields.update(ev=kind, seq=self.trace_seq, t=round(self._now(), 6), step=step,
                      dropped=self.trace_dropped)
        try:
            if self._trace is None:
                raise OSError("trace not open")
            self._trace.write(json.dumps(fields, separators=(",", ":"), default=str) + "\n")
        except (OSError, ValueError, TypeError):
            # Never silently disable: count the loss; seq gaps expose it too.
            self.trace_dropped += 1

    def _trace_flush(self):
        if self._trace is not None:
            try:
                self._trace.flush()
            except (OSError, ValueError):
                self.trace_dropped += 1

    def _forget(self, rid):
        for bag in (self._carry, self._refused, self._refused_prev, self._passed):
            bag.discard(rid)
        for store in (self.last_service, self.arrival, self.rr_seq, self.served_tokens):
            store.pop(rid, None)

    def _track_incarnations(self, reqs):
        # A request id re-bound to a different request object is a new request:
        # it must not inherit carry, refusal, service age or arrival.
        for r in reqs:
            rid, obj = r.request_id, id(r)
            old = self._incarn.get(rid)
            if old is not None and old != obj:
                self._forget(rid)
                self._ev("incarnation_reset", rid=rid)
            self._incarn[rid] = obj

    def _tier(self, r):
        if not self.priority_mode:
            return 0
        try:
            return int(getattr(r, "priority", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _rank_prefills(self, prefills):
        # [glm53-decode-floor:v7] priority tier first (priority policy only),
        # then completed-service age and round-robin within the tier.
        return sorted(prefills, key=lambda r: (
            self._tier(r),
            self.last_service.get(r.request_id, 0.0),
            self.rr_seq.get(r.request_id, 0), self.arrival[r.request_id], r.request_id))

    def _promote_next(self, prefer_passed=False):
        grants = (self._open_rec or {}).get("grants", {})
        candidates = self._candidates
        if prefer_passed:
            # DROP-5e: after a native refusal, prefer a candidate whose native position
            # has NOT passed yet (it can still be scheduled this step), but ONLY within
            # its priority tier: a lower tier never backfills past a higher-tier
            # candidate, even one whose position already passed (that one takes the
            # next step through the refusal licence). Passed ones are annotated as
            # carry for the trace only.
            order = {r.request_id: i for i, r in enumerate(candidates)}
            candidates = sorted(candidates, key=lambda r: (
                self._tier(r), r.request_id in self._passed, order[r.request_id]))
        for r in candidates:
            if len(self.selected | set(grants)) >= self.max_chunks:
                break
            if r.request_id not in self._tried:
                self.selected.add(r.request_id)
                if (prefer_passed and r.request_id in self._passed
                        and r.request_id not in self._refused
                        and r.request_id not in self._refused_prev):
                    # Its scheduler position already passed this step: carry it.
                    self._carry.add(r.request_id)
                    self._ev("carry", rid=r.request_id, tier=self._tier(r))

    def _release(self, rid, reason="allocation"):
        rec = self._open_rec
        if rec is None:
            return
        # A zero reported for an unselected request is a policy pass, not an
        # admission attempt; keep it eligible for carry.
        if rid not in self.selected and rid not in rec["grants"]:
            return
        grant = rec["grants"].pop(rid, None)
        if grant:
            self.credit += grant[1]
            if grant[2]:
                rec["borrowed"] = False
        self.selected.discard(rid)
        self._tried.add(rid)
        # DROP-5b: a released request (credit, gap, zero-progress or allocation) never
        # keeps carry, so a deferral cannot renew its own pre-selection next step.
        self._carry.discard(rid)
        if reason in ("zero_progress", "allocation"):
            # A selected request that could not progress rotates behind its
            # tier peers. Credit/gap deferrals keep their fair rank.
            self.rr_n += 1
            self.rr_seq[rid] = self.rr_n
        self.defer_reason = reason
        self._ev("release", rid=rid, reason=reason, refunded=bool(grant))
        # Only a native allocation refusal may create carry. Credit, gap-budget and
        # zero/alignment deferrals re-rank without carrying anyone.
        self._promote_next(prefer_passed=(reason == "allocation"))

    def note_scheduled(self, request, num_new_tokens):
        # Called after alignment/encoder caps; final allocation is sealed below.
        if num_new_tokens <= 0:
            self._release(request.request_id, "zero_progress")

    def note_alloc_failed(self, request):
        # Native KV allocation refusal (new_blocks is None), distinct from a
        # policy/alignment zero. Only a selected or granted request counts.
        rid = request.request_id
        rec = self._open_rec
        if rec is not None and (rid in self.selected or rid in rec["grants"]):
            self._refused.add(rid)
            self._carry.discard(rid)  # a refused request cannot hold carry
            self.refusals += 1
            self._ev("refusal", rid=rid, tier=self._tier(request))
        self._release(rid, "allocation")

    def protect_decode(self, request):
        return (self.mode == "fair" and self._open_rec is not None
                and self._open_rec["had_decode"]
                and self.needs_prefill_compute(request))

    def begin_step(self, sched):
        now = self._now()
        sid = id(sched)
        tag = (sid, int(sched.current_step))
        if self.step_tag == tag:
            return
        if self._sched_id is not None and self._sched_id != sid:
            self.inflight.clear()
            self._open_rec = None
            self._carry.clear()
            self._refused.clear()
            self._refused_prev = set()
            self.credit = 0.0
            self.in_contention = False
            self.last_account_mono = None
        self._sched_id = sid
        # An unfinished/failed schedule has dispatched nothing: refund its grants.
        if self._open_rec:
            self.credit += sum(g[1] for g in self._open_rec["grants"].values())
        self.step_tag = tag
        self.steps += 1
        self._prune(self._live_ids(sched))
        running = list(sched.running)
        waiting = list(self._iter_waiting(sched))
        prefills = list({r.request_id: r for r in running + waiting
                         if self.needs_prefill_compute(r)}.values())
        decodes = [r for r in running if not self.needs_prefill_compute(r)]
        # The base loop checks eligibility and reserves BOTH token and input/draft
        # capacity by executing these requests first. Preserve order within groups.
        if self.mode == "fair":
            sched.running[:] = decodes + [r for r in running
                                         if self.needs_prefill_compute(r)]
        self._track_incarnations(running + waiting)
        for r in running + waiting:
            self.arrival.setdefault(r.request_id, now)
        policy = getattr(sched, "policy", None)
        self.priority_mode = getattr(policy, "value", policy) == "priority"
        self._candidates = self._rank_prefills(prefills)
        self._tried = set()
        self.selected = set()
        self._passed = set()
        self._deferred_tiers = set()
        live_prefills = {r.request_id for r in prefills}
        self._carry.intersection_update(live_prefills)
        # DROP-5d: carry is a one-step trace annotation only; it never grants
        # preselection and never survives into the next step.
        self._carry.clear()
        # Refusals are evidence for exactly one following step.
        self._refused_prev = self._refused & live_prefills
        self._refused = set()
        # Refused requests keep ordinary ranked retry but may not be carry-preselected.
        self._carry.difference_update(self._refused_prev)
        sched._glm53_align_prefill_limit = None
        self._open_rec = {
            "step_id": int(sched.current_step), "t_submit": now,
            "had_decode": bool(decodes), "had_prefill_demand": bool(prefills),
            "shape": self._shape(decodes, prefills), "grants": {},
            "borrowed": False,
        }
        # Do not forgive debt while a decoder or its outstanding work remains.
        if not decodes and not any(r["had_decode"] for r in self.inflight.values()):
            self.in_contention = False
            self.credit = 0.0
            self._carry.clear()
        self.step_mode = "legacy" if self.mode != "fair" else "solo"
        self.defer_reason = "none"
        if self.mode != "fair" or not prefills or not decodes:
            return
        if not self.in_contention:
            self.credit = min(self.max_step_s, self._est_dt(self.chunk))
            self.in_contention = True
        if self.inflight_prefill:
            self.step_mode = "decode_only"
            self.defer_reason = "async_inflight"
        else:
            self.step_mode = "prefill_turn"
            # DROP-5d refusal-licensed preselection: walk the ranking and skip each
            # candidate natively refused in the previous step, so the next-ranked one gets
            # the turn. Carry never grants preselection. A lower tier can precede a higher
            # tier only when every higher-ranked candidate was refused in the previous step
            # (the bounded exception); a new or unrefused candidate always keeps its rank.
            # The skipped request is retried in this same step if the selected one defers,
            # and ranks normally next step.
            for r in self._candidates:
                if len(self.selected) >= self.max_chunks:
                    break
                if r.request_id in self._refused_prev:
                    continue
                self.selected.add(r.request_id)
            self._promote_next()
        self._ev("step", mode=self.step_mode, prio=self.priority_mode,
                 cand=[(r.request_id, self._tier(r)) for r in self._candidates],
                 sel=sorted(self.selected), carry=sorted(self._carry),
                 refused_prev=sorted(self._refused_prev))
        self._maybe_log()

    def cap_for(self, sched, request, computed=None):
        self.begin_step(sched)
        sched._glm53_align_prefill_limit = None
        remaining = self.prefill_remaining(request, computed)
        if remaining <= 0 or self.mode == "off":
            return None
        peer_decode = any(r is not request and not self.needs_prefill_compute(r)
                          for r in sched.running)
        if self.mode == "skip":
            return 0 if peer_decode else None
        if self.mode == "cap":
            cap = self.legacy_cap if peer_decode else None
            sched._glm53_align_prefill_limit = cap
            return cap
        if self.step_mode == "solo":
            return None
        rid = request.request_id
        rec = self._open_rec
        if rid in rec["grants"]:
            return rec["grants"][rid][0]
        self._passed.add(rid)
        if self.step_mode != "prefill_turn" or rid not in self.selected:
            return 0
        reserved = sum(g[1] for g in rec["grants"].values())
        gap_room = max(0.0, self.max_step_s - reserved)
        age = self._now() - self.last_service.get(rid, self.arrival[rid])
        pick = self._target(remaining, gap_room)
        my_tier = self._tier(request)
        if pick is None:
            self._deferred_tiers.add(my_tier)   # DROP-5f: a deferred tier blocks lower-tier borrowing
            self._release(rid, "gap_budget")
            if age >= self.interval_s:
                self.missed_prefill += 1
                self._maybe_log()
            return 0
        # Save credit for the most efficient rung that fits the step budget
        # instead of spending it on a smaller chunk now (v4 did, and stalled
        # at 128-token steps once one expensive sample priced 256 too high).
        cap, cost = pick
        borrowed = False
        if cost > self.credit + 1e-9:
            # A never-served newcomer gets a prompt probe; afterwards service
            # ages. Either may borrow ONE step-bounded chunk globally, only
            # after all shared debt is repaid: queue churn or several aged
            # newcomers cannot repeatedly overdraw the decoder.
            due = rid not in self.last_service or age >= self.interval_s
            # DROP-5f: under the priority policy a lower tier may not borrow in a step
            # where a strictly higher-tier candidate was credit- or gap-deferred.
            higher_deferred = any(d < my_tier for d in self._deferred_tiers)
            # DROP-5f affordable-rung admission: when a strictly lower tier is present,
            # a candidate whose chosen (largest step-fitting) rung does not fit the
            # credit takes the largest rung that DOES fit it instead of deferring, so
            # the lower tier cannot borrow past an affordable higher-tier admission.
            # With a single tier (FCFS) the credit-saving behaviour is unchanged.
            lower_present = self.priority_mode and any(
                self._tier(c) > my_tier for c in self._candidates)
            alt = (self._target(remaining, min(gap_room, max(0.0, self.credit)))
                   if lower_present else None)
            if (due and self.credit >= -1e-9 and not rec["grants"]
                    and not rec["borrowed"] and not higher_deferred):
                borrowed = True
            elif alt is not None:
                cap, cost = alt
                self._ev("affordable_rung", rid=rid, cap=cap)
            else:
                self._deferred_tiers.add(my_tier)
                self._release(rid, "credit")
                if age >= self.interval_s:
                    self.missed_prefill += 1
                    self._maybe_log()
                return 0
        self.credit -= cost
        rec["grants"][rid] = (cap, cost, borrowed)
        self._ev("grant", rid=rid, cap=cap, borrowed=borrowed)
        rec["borrowed"] |= borrowed
        self._tried.add(rid)
        sched._glm53_align_prefill_limit = cap
        return cap

    def finish_step(self, sched, scheduler_output):
        """Seal final work before _update_after_schedule advances request state."""
        rec = self._open_rec
        self._open_rec = None
        if rec is None:
            return
        self._trace_flush()
        scheduled = {rid: int(n) for rid, n in scheduler_output.num_scheduled_tokens.items()
                     if n > 0}
        requests = getattr(sched, "requests", {})
        prefill = {}
        for rid, n in scheduled.items():
            request = requests.get(rid)
            if request is not None:
                amount = min(n, self.prefill_remaining(request))
                if amount > 0:
                    prefill[rid] = amount
        reserved = 0.0
        for rid, (_, estimate, _) in rec["grants"].items():
            # Never publish/charge a tentative grant that alignment, allocation,
            # encoder caps or preemption removed from the final scheduler output.
            actual_est = min(estimate, self._est_dt(prefill[rid], rec["shape"])) if rid in prefill else 0.0
            self.credit += estimate - actual_est
            reserved += actual_est
        if not scheduled:
            # Empty schedules do not necessarily have a completion callback.
            return
        rec.update(output=scheduler_output, scheduled=scheduled,
                   prefill_tokens=prefill, reserved=reserved,
                   incarn={rid: self._incarn.get(rid) for rid in prefill})
        del rec["grants"]
        if self.mode == "fair":
            self.inflight[id(scheduler_output)] = rec

    def observe_output(self, sched, scheduler_output):
        rec = self.inflight.pop(id(scheduler_output), None)
        if rec is None or rec["output"] is not scheduler_output:
            return  # Empty, duplicate or unrelated callback; never pop another step.
        now = self._now()
        # Partition observed busy wall time instead of adding overlapping
        # submit-to-completion latencies of queued async steps. Queue residence
        # and engine idle gaps therefore cannot mint decode credit twice.
        start = rec["t_submit"]
        if self.last_account_mono is not None:
            start = max(start, self.last_account_mono)
        dt = max(0.0, now - start)
        self.last_account_mono = max(now, self.last_account_mono or now)
        actual = scheduler_output.num_scheduled_tokens
        served = {rid: min(n, max(0, int(actual.get(rid, 0))))
                  for rid, n in rec["prefill_tokens"].items()
                  if int(actual.get(rid, 0)) > 0}
        stale = {rid for rid in served
                 if rec.get("incarn", {}).get(rid) != self._incarn.get(rid)}
        for rid in stale:
            self._ev("stale_completion", step=rec["step_id"], rid=rid)
        for rid, n in served.items():
            if rid in stale:
                continue  # a replaced request's late completion earns the new one nothing
            self.last_service[rid] = now
            self.rr_n += 1
            self.rr_seq[rid] = self.rr_n
            self.served_tokens[rid] = self.served_tokens.get(rid, 0) + n
            self._carry.discard(rid)
            self._ev("served", step=rec["step_id"], rid=rid, tokens=n,
                     contention=int(rec["had_decode"]))
        n = sum(served.values())
        if rec["had_decode"]:
            # Reservation was already debited; settle once, including overruns.
            self.credit += rec["reserved"] + self.share * dt - (dt if n else 0.0)
            self.credit = min(self.credit, self._credit_limit())
            if n and dt > 0:
                self.mixed_samples.append((rec["shape"], n, dt))
                self.mixed_samples = self.mixed_samples[-64:]
                self.last_prefill_turn_mono = now
                self._model_cache = None
        elif n and dt > 0:
            self.solo_samples.append((n, dt))
            self.solo_samples = self.solo_samples[-32:]
            self._model_cache = None
        if n and self.hist_every > 0:
            print(f"[glm53-decode-floor] completed_step={rec['step_id']} "
                  f"contention={int(rec['had_decode'])} prefill_tokens={served} "
                  f"reserved_s={rec['reserved']:.3f} accounted_s={dt:.3f} "
                  f"credit={self.credit:.3f} timing=host_busy_proxy", flush=True)
        self._prune(self._live_ids(sched))
        self._trace_flush()

    def _maybe_log(self):
        if self.hist_every <= 0 or self.steps % self.hist_every != 1:
            return
        rec = self._open_rec or {}
        remaining = sum(self.prefill_remaining(r) for r in self._candidates)
        target, estimate = (self._target(None, self.max_step_s)
                            or (self.chunk, self._est_dt(self.chunk)))
        rate = target / estimate if estimate > 0 else 0.0
        eta = remaining / (self.share * rate) if self.share * rate > 0 else float("inf")
        model = self._cost_model()
        fit = (f"fit_fixed_s={model[0]:.3f} fit_us_per_tok={model[1] * 1e6:.0f}"
               if model else "fit=none")
        print(f"[glm53-decode-floor] step={rec.get('step_id', self.steps)} "
              f"mode={self.step_mode} defer={self.defer_reason} "
              f"inflight={self.inflight_prefill} credit={self.credit:.3f} "
              f"remaining={remaining} target={target} est_s={estimate:.3f} {fit} "
              f"eta_est_s={eta:.1f} max_step_s={self.max_step_s:.3f} "
              f"missed={self.missed_prefill} timing=host_busy_proxy", flush=True)

    @staticmethod
    def aligned_new_tokens(
        start, num_new, prefill_end, block_size, max_prefill_tokens, policy_cap=None
    ) -> int:
        """Hybrid align clip. policy_cap is an intentional mixed cap, not leftover budget."""
        if policy_cap is not None and policy_cap > 0:
            max_prefill_tokens = min(max_prefill_tokens, policy_cap)
        end = start + num_new
        if end < prefill_end:
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
        return max(0, end - start)



def _helper_text() -> str:
    body = inspect.getsource(_Glm53MixedPrefill)
    return (
        "\n"
        + body
        + "\n_GLM53_MIXED = _Glm53MixedPrefill()  # [glm53-decode-floor:v7]\n\n"
        + "def _glm53_mixed_prefill_policy(sched, request, computed=None):  # [glm53-decode-floor:v7]\n"
        + "    return _GLM53_MIXED.cap_for(sched, request, computed)\n\n\n"
    )


HELPER = None  # filled at apply time so tests can call _helper_text()


BEGIN_OLD = """        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
"""
BEGIN_NEW = """        self.current_step += 1
        _GLM53_MIXED.begin_step(self)  # [glm53-decode-floor:v4]
        # NOTE(woosuk) on the scheduling algorithm:
"""

OBS_OLD = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
"""
OBS_NEW = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        _GLM53_MIXED.observe_output(self, scheduler_output)  # [glm53-decode-floor:v4]
        pooler_outputs = model_runner_output.pooler_output
"""

RUNNING_OLD = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
"""
RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v4]
            if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""

WAITING_OLD = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
"""
WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self, request, num_computed_tokens)  # [glm53-decode-floor:v4]
                    if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request, num_computed_tokens):
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""

ALIGN_OLD = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""
ALIGN_NEW = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            _align_cap = getattr(self, "_glm53_align_prefill_limit", None)  # [glm53-decode-floor:v4]
            if _align_cap is not None and _align_cap > 0:
                max_prefill_tokens = min(max_prefill_tokens, _align_cap)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""

RUNNING_MAMBA_OLD = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
"""
RUNNING_MAMBA_NEW = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
            _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v4]
"""

WAITING_MAMBA_OLD = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        if num_new_tokens == 0:
                            break
"""
WAITING_MAMBA_NEW = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v4]
                        if num_new_tokens == 0:
                            if _GLM53_MIXED.mode == "fair":
                                request_queue.pop_request()
                                step_skipped_waiting.prepend_request(request)
                                continue
                            break
"""

FIN_OLD = """        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
"""
FIN_NEW = """        _GLM53_MIXED.finish_step(self, scheduler_output)  # [glm53-decode-floor:v4]
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
"""

RUNNING_ZERO_OLD = """            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
"""
RUNNING_ZERO_NEW = """            if num_new_tokens == 0:
                _GLM53_MIXED.note_scheduled(request, 0)  # [glm53-decode-floor:v4]
                # The request cannot be scheduled because one of the following
"""

WAITING_ZERO_OLD = """                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break
"""
WAITING_ZERO_NEW = """                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            _GLM53_MIXED.note_scheduled(request, 0)  # [glm53-decode-floor:v4]
                            if _GLM53_MIXED.mode == "fair":
                                request_queue.pop_request()
                                step_skipped_waiting.prepend_request(request)
                                continue
                            break
"""

PREFILL_PREEMPT_OLD = """                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
"""
PREFILL_PREEMPT_NEW = """                    # The request cannot be scheduled.
                    if _GLM53_MIXED.protect_decode(request):  # [glm53-decode-floor:v4]
                        break
                    # Preempt the lowest-priority request.
"""

RUNNING_ALLOC_OLD = """            if new_blocks is None:
                # Cannot schedule this request.
                break
"""
RUNNING_ALLOC_NEW = """            if new_blocks is None:
                # Cannot schedule this request.
                _GLM53_MIXED.note_scheduled(request, 0)  # [glm53-decode-floor:v4]
                if _GLM53_MIXED.protect_decode(request):
                    req_index += 1
                    continue
                break
"""

WAITING_ALLOC_OLD = """                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer:"""
WAITING_ALLOC_NEW = """                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    _GLM53_MIXED.note_scheduled(request, 0)  # [glm53-decode-floor:v4]
                    if _GLM53_MIXED.protect_decode(request):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
                    break

                # KVTransfer:"""

V4_PAIRS = (
    (BEGIN_NEW, BEGIN_OLD, 'begin'),
    (OBS_NEW, OBS_OLD, 'obs'),
    (RUNNING_NEW, RUNNING_OLD, 'running'),
    (WAITING_NEW, WAITING_OLD, 'waiting'),
    (ALIGN_NEW, ALIGN_OLD, 'align'),
    (RUNNING_MAMBA_NEW, RUNNING_MAMBA_OLD, 'running_mamba'),
    (WAITING_MAMBA_NEW, WAITING_MAMBA_OLD, 'waiting_mamba'),
    (FIN_NEW, FIN_OLD, 'fin'),
    (RUNNING_ZERO_NEW, RUNNING_ZERO_OLD, 'running_zero'),
    (WAITING_ZERO_NEW, WAITING_ZERO_OLD, 'waiting_zero'),
    (PREFILL_PREEMPT_NEW, PREFILL_PREEMPT_OLD, 'prefill_preempt'),
    (RUNNING_ALLOC_NEW, RUNNING_ALLOC_OLD, 'running_alloc'),
    (WAITING_ALLOC_NEW, WAITING_ALLOC_OLD, 'waiting_alloc'),
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


V1_PAIRS = (
    (V1_RUNNING_NEW, RUNNING_OLD, 'v1-running'),
    (V1_WAITING_NEW, WAITING_OLD, 'v1-waiting'),
)
V2_PAIRS = (
    (V2_BEGIN_NEW, BEGIN_OLD, 'v2-begin'),
    (V2_OBS_NEW, OBS_OLD, 'v2-obs'),
    (V2_RUNNING_NEW, RUNNING_OLD, 'v2-running'),
    (V2_WAITING_NEW, WAITING_OLD, 'v2-waiting'),
    (V2_ALIGN_NEW, ALIGN_OLD, 'v2-align'),
    (V2_RUNNING_MAMBA_NEW, RUNNING_MAMBA_OLD, 'v2-running-mamba'),
    (V2_WAITING_MAMBA_NEW, WAITING_MAMBA_OLD, 'v2-waiting-mamba'),
)
# v5/v6 hook text is the frozen v4 insertion with the marker advanced; every
# known v5/v6 producer shares it byte-for-byte, so identities differ only in
# their helper bytes.
V5_PAIRS = tuple((new.replace(MARK_V4, MARK_V5), old, label) for new, old, label in V4_PAIRS)
V6_PAIRS = tuple((new.replace(MARK_V4, MARK_V6), old, label) for new, old, label in V4_PAIRS)

_ALLOC_OLD_CALL = "_GLM53_MIXED.note_scheduled(request, 0)  " + MARK_V7
_ALLOC_NEW_CALL = "_GLM53_MIXED.note_alloc_failed(request)  " + MARK_V7


def _v7_pair(new, old, label):
    new = new.replace(MARK_V4, MARK_V7)
    if label in ("running_alloc", "waiting_alloc"):
        # v7 reports native KV allocation refusal distinctly from a zero cap.
        if new.count(_ALLOC_OLD_CALL) != 1:
            raise SystemExit(f"{P}: v7 {label} template drifted")
        new = new.replace(_ALLOC_OLD_CALL, _ALLOC_NEW_CALL, 1)
    return new, old, label


V7_PAIRS = tuple(_v7_pair(*p) for p in V4_PAIRS)

NEEDLE = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
ADAPTIVE_K_HEAD = "class _Glm53AdaptiveK:"
CLASS_HEAD = "class _Glm53MixedPrefill:"

# ---------------------------------------------------------------------------
# Authenticated migration identities. A helper *site* is the exact text from the
# newline before its class/def head up to the next CUDAGraphStat import or
# _Glm53AdaptiveK class; it is compared by sha256 AND length, then the version's
# exact hook insertions must invert once each, then the removal must round-trip
# byte-identically. Marker presence alone authenticates nothing.
#
# provenance (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks public history):
#   v1           f3043c95bbf9 HELPER (as registered by #180 4ffef088)
#   v2           9a23b2d8f061 HELPER (as registered by #180 4ffef088)
#   v5-historical a9bbbadc4c72 _helper_text()
#   v5-main      943912cdcda2 _helper_text()  (published TP2/3/4 fair default)
#   v5-priority  #221 9dbd6873ab45 _helper_text() (in-place priority migration)
#   v6-warm-deadline #180 4ffef088edb1 == #80 ff30124ed456 _helper_text()
#   v6-carry     #246 dc702b4fca1c _helper_text()
#   v1-image-d9758a6  d9758a6 == 9a557cf HELPER, the unversioned 2026-09-13/14
#                variant baked into the published :exl3-instanttensor image
#                (ImageID ef9f5013c41a). #180 refuses it as a *retained* v1 because
#                its baked "0" default would land a held default change. v7
#                accepts it ONLY as a migration source: the helper is removed
#                and replaced by v7, whose mode comes from the launcher's
#                GLM53_MIXED_PREFILL_CHUNK, so that default does not land.
#                Without this entry v7 cannot boot on the published image.
#   v7-drop2     #283 DROP-2 patcher f49d85d8 _helper_text() (installed on .123 Sep 27)
#   v7-drop4     #283 DROP-4 patcher 19e25753 _helper_text() (installed on .123 Sep 27-28)
#                Earlier v7 helper bodies are migration inputs only: a v7 marker is
#                "verified" solely by this installer's exact current helper bytes.
#   v3/v4        no public producer body: refused, never migrated.
# ---------------------------------------------------------------------------
IDENTITIES = (
    # name, marker, head token, pairs, sha256, length
    ("v1", MARK, V1_HELPER_START, V1_PAIRS,
     "c0c10c6385bd7e75bfc0f724378960b9d9cb943cf18c386a5a780cf03cb4c48d", 784),
    ("v1-image-d9758a6", MARK, V1_HELPER_START, V1_PAIRS,
     "9ad0c79991ae3f3f8fa382810dee92c58316a03a27c81774ecd889ad986af29e", 781),
    ("v2", MARK_V2, CLASS_HEAD, V2_PAIRS,
     "e170f2bf6d0c0fe0493a91ff54bd719036f6c4b5ea9fcf2ee1346968a4e3fbed", 13378),
    ("v5-historical", MARK_V5, CLASS_HEAD, V5_PAIRS,
     "c05769276df6da0a24b3b8c21e251f39ee7aa2a74ee92c6d6ce3d85c38151a53", 20866),
    ("v5-main", MARK_V5, CLASS_HEAD, V5_PAIRS,
     "1dbbb945a3414bd12d78b53070424f432a2f07d0092898f9751814f1f9dc40f9", 20866),
    ("v5-priority", MARK_V5, CLASS_HEAD, V5_PAIRS,
     "b8bf9c9cfcb5e0e62bb3e842c3679c9adc0911c2ce66d2705131ac481a129b55", 21170),
    ("v6-warm-deadline", MARK_V6, CLASS_HEAD, V6_PAIRS,
     "4eac19b23a39ffa58c765191fd8d120b2d9cd400dcdbfe9019fcea4dee88e03a", 23415),
    ("v6-carry", MARK_V6, CLASS_HEAD, V6_PAIRS,
     "278b37ad9eea23de047726b054fd7b6929f6bdcfe76e68e21f230e55528bdc15", 22688),
    ("v7-drop2", MARK_V7, CLASS_HEAD, V7_PAIRS,
     "9ed55941e86f51c82845fb32d207acb58183f4bb55642f79d7babfcd18d2aac5", 27780),
    ("v7-drop4", MARK_V7, CLASS_HEAD, V7_PAIRS,
     "45e3786bebb7a0e207989130bfa26feeeae2280467531321c7da9ffa6b013d3d", 27988),
)


def _refuse(reason: str):
    raise SystemExit(f"{P}: refusing to rewrite the scheduler: {reason}")


def _site(text: str, head_token: str, label: str):
    """Return the unique helper site start (the newline before its head), or None."""
    found = text.count(head_token)
    if found == 0:
        return None
    if found != 1:
        _refuse(f"{found} {head_token!r} definitions found; {label} requires exactly one")
    head = text.find(head_token)
    if head < 2 or text[head - 1] != "\n" or text[head - 2] != "\n":
        _refuse(f"{label} helper is not preceded by the installer's blank-line boundary")
    return head - 1


def _span_matches(span: str, sha: str, length: int) -> bool:
    return len(span) == length and hashlib.sha256(span.encode("utf-8")).hexdigest() == sha


def _followed_by_anchor(text: str, end: int) -> bool:
    """After the exact helper bytes only blank lines may precede the next known anchor
    (the CUDAGraphStat import, or _Glm53AdaptiveK which patch_adaptive_k inserts)."""
    # Exactly the established boundaries: the helper directly followed by the
    # CUDAGraphStat import, or by patch_adaptive_k's class after its single
    # separator newline. Nothing else is tolerated.
    rest = text[end:]
    return (rest.startswith(NEEDLE) or rest.startswith(ADAPTIVE_K_HEAD)
            or rest.startswith("\n" + ADAPTIVE_K_HEAD))


# Any decode-floor marker (every version, including unknown ones) or any helper
# name. A pristine scheduler contains none; neither do the sibling overlays.
_LEFTOVER_TOKENS = (MARK.rstrip("]"), CLASS_HEAD, V1_HELPER_START,
                    "_Glm53MixedPrefill", "_GLM53_MIXED", "_glm53_mixed_prefill_policy")


def _leftover(text: str):
    return next((t for t in _LEFTOVER_TOKENS if t in text), None)


def _unpatch(text: str, name, marker, head_token, pairs, sha=None, length=None, exact=None):
    """Invert one identity: exact hook insertions, then its authenticated helper site.

    The site is authenticated as exactly `length` bytes (or `exact`) from its start,
    by sha256; it must then be followed only by blank lines and a known anchor.
    """
    for new, old, label in pairs:
        if text.count(new) != 1:
            _refuse(f"{name}: {label} insertion missing, drifted or duplicated")
        text = text.replace(new, old, 1)
    start = _site(text, head_token, name)
    if start is None:
        _refuse(f"{name}: helper site missing")
    n = len(exact) if exact is not None else length
    span = text[start:start + n]
    ok = (span == exact) if exact is not None else _span_matches(span, sha, length)
    if not ok:
        _refuse(f"{name}: helper site not authenticated (got sha256 "
                f"{hashlib.sha256(span.encode()).hexdigest()[:16]} over {len(span)} bytes)")
    if not _followed_by_anchor(text, start + n):
        _refuse(f"{name}: unexpected text between the helper and its anchor")
    clean = text[:start] + text[start + n:]
    # Fail closed on mixed state: after removing this identity, no decode-floor
    # marker of any version and no helper definition or name may remain.
    if marker in clean or head_token in clean or _leftover(clean):
        _refuse(f"{name}: marker or helper definition left after unpatch")
    return clean, start, span


def _round_trip(original: str, clean: str, start: int, span: str, pairs, name: str) -> None:
    again = clean[:start] + span + clean[start:]
    for new, old, label in pairs:
        again = replace_once(again, old, new, label)
    if again != original:
        _refuse(f"{name}: round trip is not byte-identical; not a canonical {name} install")


def apply_v7(text: str, at: int | None = None) -> str:
    if "import os\n" not in text.split("import time\n", 1)[0]:
        text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import os")
    if text.count(CLASS_HEAD) or text.count(V1_HELPER_START):
        _refuse("a helper definition is still present before v7 install")
    helper = _helper_text()
    if at is None:
        text = replace_once(text, NEEDLE, helper + NEEDLE, "helper")
    else:
        # Reinstall at the removed legacy site so later overlays keep their place.
        text = text[:at] + helper + text[at:]
    for new, old, label in V7_PAIRS:
        text = replace_once(text, old, new, label)
    compile(text, str(P), "exec")
    return text


def _identify(text: str):
    """Authenticate the installed legacy identity (or None for a pristine scheduler)."""
    for unsupported in (MARK_V3, MARK_V4):
        if unsupported in text:
            _refuse(f"{unsupported} is unsupported: no authenticated producer body exists; "
                    "restore an authenticated/pristine scheduler first")
    if MARK_V7 in text:
        marker = MARK_V7
    elif MARK_V6 in text:
        marker = MARK_V6
    elif MARK_V5 in text:
        marker = MARK_V5
    elif MARK_V2 in text:
        marker = MARK_V2
    elif MARK in text or V1_HELPER_START in text:
        marker = MARK
    else:
        if CLASS_HEAD in text:
            _refuse("unmarked _Glm53MixedPrefill helper present")
        token = _leftover(text)
        if token is not None:
            _refuse(f"unrecognized decode-floor state ({token!r} present); "
                    "restore an authenticated/pristine scheduler first")
        return None
    reasons = []
    for name, mark, head, pairs, sha, length in IDENTITIES:
        if mark != marker:
            continue
        try:
            clean, start, span = _unpatch(text, name, mark, head, pairs, sha=sha, length=length)
        except SystemExit as exc:
            reasons.append(str(exc).split("scheduler: ", 1)[-1])
            continue
        _round_trip(text, clean, start, span, pairs, name)
        # Every legacy helper uses `os`; the import is part of the applied state and
        # must be present before a legacy identity is accepted for migration.
        if "import os\n" not in text.split("import time\n", 1)[0]:
            _refuse(f"{name}: import drifted")
        return name, clean, start
    _refuse(f"no authenticated identity for {marker}: " + " | ".join(reasons))


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    original = text
    if MARK_V7 in text:
        # Position-independent verification of the installed v7: exact hooks,
        # exact CURRENT helper bytes, byte-identical round trip. No write.
        try:
            clean, start, span = _unpatch(text, "v7", MARK_V7, CLASS_HEAD, V7_PAIRS,
                                          exact=_helper_text())
        except SystemExit:
            clean = None  # not the current helper: only an authenticated legacy v7 may migrate
        if clean is not None:
            _round_trip(text, clean, start, span, V7_PAIRS, "v7")
            if "import os\n" not in text.split("import time\n", 1)[0]:
                _refuse("v7 import drifted")
            compile(text, str(P), "exec")
            print(f"{P.name}: {MARK_V7} already present - verified")
            return 0
    found = _identify(text)
    if found is None:
        text = apply_v7(text)
        source = "pristine"
    else:
        source, clean, start = found
        text = apply_v7(clean, at=start)
    for old in (MARK_V2, MARK_V3, MARK_V4, MARK_V5, MARK_V6):
        if old in text:
            raise SystemExit(f"{P}: older marker left after migration")
    if text != original:
        P.write_text(text)
    print(f"patched {P.name} ({MARK_V7}) from {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
