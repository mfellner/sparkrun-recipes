#!/usr/bin/env python3
"""CPU regression tests for fair scheduling and versioned source migration."""
from __future__ import annotations

import ast
import hashlib
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_PATCH_CANDIDATES = (
    HERE / 'patch_scheduler_decode_floor.py',
    ROOT / 'overlay' / 'patch_scheduler_decode_floor.py',
)
PATCH = next((p for p in _PATCH_CANDIDATES if p.is_file()), None)
if PATCH is None:
    raise SystemExit(
        'missing patch_scheduler_decode_floor.py (tried '
        + ', '.join(str(p) for p in _PATCH_CANDIDATES)
        + ')'
    )
_FIXTURE_CANDIDATES = (HERE / 'fixtures', ROOT / 'tests' / 'fixtures')  # image layout, then checkout
FIXTURES = next((p for p in _FIXTURE_CANDIDATES if (p / 'legacy_scheduler_helpers.py').is_file()), None)
if FIXTURES is None:
    raise SystemExit('missing fixtures/legacy_scheduler_helpers.py (tried '
                     + ', '.join(str(p) for p in _FIXTURE_CANDIDATES) + ')')
spec = importlib.util.spec_from_file_location('glm53_decode_floor', PATCH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
POLICY = mod._Glm53MixedPrefill
PATCHED_SOURCE = None
FAIR_ENV = {
    'GLM53_MIXED_PREFILL_CHUNK': 'fair',
    'GLM53_FAIR_PREFILL_CHUNK': '256',
    'GLM53_FAIR_PREFILL_SHARE': '0.20',
    'GLM53_FAIR_PREFILL_MAX_INTERVAL_MS': '2000',
    'GLM53_FAIR_PREFILL_MAX_STEP_MS': '1000',
    'GLM53_FAIR_PREFILL_MAX_CHUNKS': '1',
}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Req:
    def __init__(self, rid, prompt=30000, computed=0, decode=False):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_tokens = prompt + int(decode)
        self.spec_token_ids = list(range(7)) if decode else []
        self.num_output_placeholders = 0
        self.next_decode_eligible_step = 0
        self.max_tokens = 4000
        self.has_encoder_inputs = False
        self.is_prefill_chunk = not decode

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


class Sched:
    def __init__(self, running=(), waiting=()):
        self.running = list(running)
        self.waiting = list(waiting)
        self.skipped_waiting = []
        self.current_step = 1
        self.max_model_len = 850000
        self.num_sampled_tokens_per_step = 1
        self.need_mamba_block_aligned_split = False
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=3584)
        self.refresh()

    def refresh(self):
        self.requests = {r.request_id: r for r in self.running + self.waiting + self.skipped_waiting}


class Out:
    def __init__(self, counts):
        self.num_scheduled_tokens = dict(counts)
        self.total_num_scheduled_tokens = sum(counts.values())


class FairTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.a = Req('A', 30000, 30000, decode=True)
        self.b = Req('B')
        self.s = Sched([self.a], [self.b])
        self.p = self.policy()

    def policy(self, **env):
        with patch.dict(os.environ, {**FAIR_ENV, **env}), contextlib.redirect_stdout(io.StringIO()):
            p = POLICY(now=self.clock)
        p.hist_every = 0
        return p

    def submit(self, counts, sched=None):
        s = sched or self.s
        self.p.begin_step(s)
        out = Out(counts)
        self.p.finish_step(s, out)
        s.current_step += 1
        return out

    def complete(self, out, dt, sched=None):
        self.clock.advance(dt)
        self.p.observe_output(sched or self.s, out)

    def learn(self, cost=0.2):
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)
        out = self.submit({'A': 8, 'B': 256})
        self.complete(out, cost)

    def test_legacy_modes_and_alignment(self):
        for mode, expected in [('skip', 0), ('-1', 0), ('0', None), ('off', None), ('128', 128)]:
            p = self.policy(GLM53_MIXED_PREFILL_CHUNK=mode)
            s = Sched([self.a], [self.b])
            self.assertEqual(p.cap_for(s, self.b), expected)
        p = self.policy(GLM53_MIXED_PREFILL_CHUNK='skip')
        for _ in range(10000):
            self.s.current_step += 1
            self.assertEqual(p.cap_for(self.s, self.b), 0)
        self.assertFalse(p.inflight)
        align = POLICY.aligned_new_tokens
        self.assertEqual(align(0, 128, 30000, 3584, 3584), 0)
        for block in (1792, 3584):
            self.assertEqual(align(0, 128, 30000, block, 3584, 128), 128)
            self.assertEqual(align(block - 64, 128, 30000, block, 3584, 128), 64)
        self.assertEqual(align(29900, 100, 30000, 3584, 3584, 128), 100)

    def test_solo_prefill_then_immediate_newcomer_has_no_debt(self):
        pre = Req('A')
        self.s.running, self.s.waiting = [pre], []
        self.s.refresh()
        self.assertIsNone(self.p.cap_for(self.s, pre))
        out = self.submit({'A': 3584})
        self.complete(out, 6.0)
        self.assertEqual(self.p.credit, 0)
        self.assertEqual(self.p.solo_samples, [(3584, 6.0)])
        self.s.running, self.s.waiting = [self.a], [self.b]
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)

    def test_late_solo_completion_keeps_original_classification(self):
        pre = Req('A')
        self.s.running, self.s.waiting = [pre], []
        self.s.refresh()
        solo = self.submit({'A': 3584})
        self.s.running, self.s.waiting = [self.a], [self.b]
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)
        self.assertEqual(self.p.defer_reason, 'async_inflight')
        dec = self.submit({'A': 8})
        credit = self.p.credit
        self.complete(solo, 5.0)
        self.assertEqual(self.p.credit, credit)
        self.assertNotIn('B', self.p.last_service)
        self.complete(dec, 0.05)
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)

    def test_saves_credit_for_target_rung_instead_of_small_chunk(self):
        # One 256@0.2s sample scales linearly: 1024 (0.8s) is the largest rung
        # under max_step_s. An already-served B with 0.21 credit waits for it
        # instead of buying a 256 now (v4 spent greedily and stalled small).
        self.learn()
        self.p.credit = 0.21
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)
        self.assertEqual(self.p.defer_reason, 'credit')
        self.assertAlmostEqual(self.p.credit, 0.21)
        self.s.current_step += 1
        self.p.begin_step(self.s)
        self.p.credit = 0.85
        self.assertEqual(self.p.cap_for(self.s, self.b), 1024)
        self.assertAlmostEqual(self.p.credit, 0.05)
        self.assertAlmostEqual(self.p._open_rec['grants']['B'][1], 0.8)
        self.assertFalse(self.p._open_rec['grants']['B'][2])

    def kit_samples(self):
        # Head-log shaped samples: solo 3584@2.68s and an 82-token tail@0.31s
        # (~0.25s fixed + ~0.68ms/token); three mixed 128@0.34s agree.
        self.p.solo_samples = [(3584, 2.68), (82, 0.31)]
        self.p.mixed_samples = [((1, 3, 0), 128, 0.34)] * 3
        self.p._model_cache = None

    def test_fixed_cost_fit_prices_large_chunks_from_small_samples(self):
        self.kit_samples()
        fixed, per_tok = self.p._cost_model()
        self.assertGreater(fixed, 0.2)
        self.assertLess(per_tok, 0.001)
        self.assertLess(self.p._est_dt(1024), 1.0)   # v4: 2.72s from 128@0.34
        self.assertGreater(self.p._est_dt(2048), 1.0)
        self.assertGreater(self.p._est_dt(128), 0.3)

    def test_solo_samples_alone_price_the_first_probe(self):
        self.p.solo_samples = [(3584, 2.68), (82, 0.31)]
        self.p._model_cache = None
        self.assertGreater(self.p._est_dt(256), 0.35)
        self.assertLess(self.p._est_dt(256), 0.6)

    def test_single_outlier_does_not_dominate_estimate(self):
        self.p.solo_samples = [(3584, 2.68), (82, 0.31)]
        self.p.mixed_samples = [((1, 3, 0), 256, 0.43)] * 5 + [((1, 3, 0), 256, 0.60)]
        self.p._model_cache = None
        self.assertLess(self.p._est_dt(1024), 1.1)
        self.p.mixed_samples = [((1, 3, 0), 256, 0.60)] * 6
        self.p._model_cache = None
        self.assertGreater(self.p._est_dt(1024), 1.0)  # consistently slow mixed steps push 1024 over the 1.0s gate

    def test_ladder_climbs_back_after_small_chunks(self):
        self.kit_samples()
        self.p.begin_step(self.s)
        self.p.last_service['B'] = self.clock()
        self.p.credit = 1.0
        self.assertEqual(self.p.cap_for(self.s, self.b), 1024)

    def test_never_served_newcomer_gets_prompt_step_bounded_probe(self):
        self.kit_samples()
        self.p.begin_step(self.s)
        self.p.credit = 0.1
        self.assertEqual(self.p.cap_for(self.s, self.b), 1024)
        self.assertTrue(self.p._open_rec['grants']['B'][2])
        self.assertLess(self.p.credit, 0)
        out = self.submit({'A': 8, 'B': 1024})
        self.complete(out, 0.95)
        self.assertLess(self.p.credit, 0)
        c = Req('C')
        self.s.waiting.append(c)
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, c), 0)
        self.assertEqual(self.p.defer_reason, 'credit')
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)

    def test_step_budget_caps_the_target_rung(self):
        self.p = self.policy(GLM53_FAIR_PREFILL_MAX_STEP_MS='500')
        self.kit_samples()
        self.p.begin_step(self.s)
        self.p.credit = 1.0
        cap = self.p.cap_for(self.s, self.b)
        self.assertIn(cap, (256, 512))
        self.assertLessEqual(self.p._open_rec['grants']['B'][1], 0.5)

    def test_positive_credit_does_not_override_step_latency(self):
        self.learn(cost=3.0)
        self.p.credit = 10
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)
        self.assertEqual(self.p.defer_reason, 'gap_budget')

    def test_age_and_step_budgets_are_independent(self):
        p = self.policy(GLM53_FAIR_PREFILL_MAX_INTERVAL_MS='10000', GLM53_FAIR_PREFILL_MAX_STEP_MS='100')
        self.assertEqual(p.interval_s, 10)
        self.assertEqual(p.max_step_s, 0.1)
        self.assertEqual(p.cap_for(self.s, self.b), 0)

    def test_cost_feedback_can_shrink_chunks(self):
        self.learn(cost=1.5)
        self.p.credit = 1
        self.assertEqual(self.p.cap_for(self.s, self.b), 128)

    def test_ladder_uses_available_credit(self):
        self.learn(cost=0.2)
        self.p.credit = 0.9
        self.assertEqual(self.p.cap_for(self.s, self.b), 1024)
        self.assertAlmostEqual(self.p.credit, 0.1)

    def test_aggregate_reservations_bound_multiple_newcomers(self):
        self.p = self.policy(GLM53_FAIR_PREFILL_MAX_CHUNKS='3')
        self.s.waiting += [Req('C'), Req('D')]
        self.s.refresh()
        self.p.begin_step(self.s)
        self.p.credit = 0.4
        caps = [self.p.cap_for(self.s, r) for r in self.s.waiting]
        self.assertEqual(caps, [256, 256, 0])
        self.assertGreaterEqual(self.p.credit, 0)
        self.assertLessEqual(sum(g[1] for g in self.p._open_rec['grants'].values()), 0.4)

    def test_age_cannot_borrow_repeatedly_without_repayment(self):
        self.p = self.policy(GLM53_FAIR_PREFILL_MAX_CHUNKS='3')
        self.s.waiting += [Req('C'), Req('D')]
        self.s.refresh()
        self.p.begin_step(self.s)
        self.p.credit = 0.01
        self.clock.advance(3)
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)
        out = self.submit({'A': 8, 'B': 256})
        self.complete(out, 0.4)
        self.assertLess(self.p.credit, 0)
        self.clock.advance(10)
        self.assertEqual(self.p.cap_for(self.s, self.s.waiting[1]), 0)
        self.assertEqual(self.p.cap_for(self.s, self.s.waiting[2]), 0)

    def test_empty_schedule_does_not_block_next_prefill_or_mint_credit(self):
        before = self.p.cap_for(self.s, self.b)
        self.assertEqual(before, 256)
        empty = self.submit({})
        self.assertFalse(self.p.inflight)
        credit = self.p.credit
        self.clock.advance(100)
        self.p.observe_output(self.s, empty)
        self.assertEqual(self.p.credit, credit)
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)

    def test_removed_grant_refunded_and_no_service_credited(self):
        self.p.cap_for(self.s, self.b)
        out = self.submit({'A': 8})
        self.assertEqual(self.p.inflight_prefill, 0)
        self.complete(out, 0.05)
        self.assertNotIn('B', self.p.last_service)
        self.assertGreater(self.p.credit, 0)

    def test_final_counts_override_provisional_grant(self):
        self.p.cap_for(self.s, self.b)
        out = self.submit({'A': 8, 'B': 64})
        self.b.num_computed_tokens = 30000  # Async scheduler has already advanced it.
        self.complete(out, 0.1)
        self.assertEqual(self.p.served_tokens['B'], 64)

    def test_newer_open_step_cannot_contaminate_older_completion(self):
        decode = self.submit({'A': 8})
        self.p.cap_for(self.s, self.b)
        self.complete(decode, 0.1)
        self.assertNotIn('B', self.p.last_service)
        mixed = self.submit({'A': 8, 'B': 256})
        self.complete(mixed, 0.2)
        self.assertEqual(self.p.served_tokens['B'], 256)

    def test_queued_async_time_is_accounted_once(self):
        self.p.begin_step(self.s)
        self.p.credit = 0
        one = self.submit({'A': 8})
        two = self.submit({'A': 8})
        self.complete(one, 0.1)
        self.complete(two, 0.1)
        self.assertAlmostEqual(self.p.credit, 0.04)

    def test_duplicate_and_unrelated_outputs_do_not_pop_records(self):
        self.p.cap_for(self.s, self.b)
        out = self.submit({'A': 8, 'B': 256})
        self.p.observe_output(self.s, Out({'A': 8, 'B': 256}))
        self.assertEqual(self.p.inflight_prefill, 1)
        self.complete(out, 0.2)
        credit = self.p.credit
        self.p.observe_output(self.s, out)
        self.assertEqual(self.p.credit, credit)
        self.assertEqual(self.p.served_tokens['B'], 256)

    def test_idle_time_is_not_decode_service(self):
        first = self.submit({'A': 8})
        self.complete(first, 0.1)
        self.p.credit = 0
        self.clock.advance(100)
        second = self.submit({'A': 8})
        self.complete(second, 0.1)
        self.assertAlmostEqual(self.p.credit, 0.02)

    def test_cancel_and_rearrival_keep_debt_while_a_decodes(self):
        self.learn(cost=1.0)
        debt = self.p.credit
        self.s.waiting = [Req('C')]
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, self.s.waiting[0]), 0)
        self.assertEqual(self.p.credit, debt)
        self.assertNotIn('B', self.p.last_service)

    def test_zero_progress_promotes_next_candidate(self):
        c = Req('C')
        self.s.waiting.append(c)
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)
        self.assertEqual(self.p.cap_for(self.s, c), 0)
        self.p.note_scheduled(self.b, 0)
        self.assertEqual(self.p.cap_for(self.s, c), 256)

    def test_native_kv_refusal_keeps_waiter_retryable_and_peer_progressing(self):
        # #246: a selected waiter that vLLM refuses KV must not stall a runnable
        # prefill, must stay a candidate, and must recover once memory frees.
        c = Req('C')
        self.s.waiting.append(c)
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)
        self.assertEqual(self.p.cap_for(self.s, c), 0)
        self.p.note_alloc_failed(self.b)                      # native refusal (new_blocks is None)
        cap_c = self.p.cap_for(self.s, c)
        self.assertGreater(cap_c, 0)                          # the runnable peer progresses in the same step
        self.complete(self.submit({'A': 8, 'C': cap_c}), 0.2)
        self.p.begin_step(self.s)                             # next step: refusal evidence lasts one step
        self.assertEqual(self.p._refused_prev, {'B'})
        self.assertNotIn('B', self.p.selected)                # not preselected right after its refusal
        self.assertIn('B', [r.request_id for r in self.p._candidates])  # but still queued for retry
        served = {'B': 0, 'C': cap_c}
        for _ in range(20):                                   # memory has freed: no further refusals
            counts = {'A': 8}
            for r in (self.b, c):
                cap = self.p.cap_for(self.s, r)
                if cap:
                    counts[r.request_id] = cap
                    served[r.request_id] += cap
            self.complete(self.submit(counts), 0.2)
            self.p.begin_step(self.s)
            if served['B'] and served['C'] > cap_c:
                break
        self.assertGreater(served['B'], 0)                    # the refused waiter recovers
        self.assertGreater(served['C'], cap_c)                # and the peer keeps progressing

    def test_full_prefix_hit_is_not_blocked_as_cold_prefill(self):
        self.p.begin_step(self.s)
        self.p.credit = -10
        self.assertIsNone(self.p.cap_for(self.s, self.b, computed=30000))

    def test_small_remaining_tail_can_fit_when_base_chunk_cannot(self):
        self.b.num_computed_tokens = 29980
        self.p.begin_step(self.s)
        self.p.credit = 0.05
        self.assertEqual(self.p.cap_for(self.s, self.b), 20)

    def running_loop(self, input_budget, allocate=None):
        if PATCHED_SOURCE is None:
            self.skipTest('source installation required')
        # Execute the pinned scheduler's actual running-loop budget/eligibility
        # code, omitting only KV allocation and post-allocation speculative setup.
        tree = ast.parse(PATCHED_SOURCE)
        schedule = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
        loop = next(n for n in schedule.body if isinstance(n, ast.While))
        allocate_index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.With))
        append_index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr)
                            and ast.unparse(n).startswith('scheduled_running_reqs.append'))
        increment_index = next(i for i in range(append_index, len(loop.body))
                               if ast.unparse(loop.body[i]) == 'req_index += 1')
        loop.body = (loop.body[:allocate_index] + loop.body[append_index:increment_index + 1]
                     if allocate is None else loop.body[:increment_index + 1])
        code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), '<actual-running-loop>', 'exec')
        self.s.running, self.s.waiting = [self.b, self.a], []
        self.s.refresh()
        self.s.kv_cache_manager = SimpleNamespace(allocate_slots=allocate)
        self.s.num_lookahead_tokens = 7
        self.p.begin_step(self.s)
        ns = dict(self=self.s, _GLM53_MIXED=self.p,
                  _glm53_mixed_prefill_policy=lambda s, r: self.p.cap_for(s, r),
                  req_index=0, token_budget=7168, input_budget=input_budget, draft_slots=8,
                  defer_prefills=False, encoder_compute_budget=0, prefill_scheduled=False,
                  scheduled_running_reqs=[], req_to_new_blocks={}, num_scheduled_tokens={}, new_blocks=[],
                  record_function_or_nullcontext=lambda _: contextlib.nullcontext())
        exec(code, ns)
        return ns

    def test_decode_order_reserves_real_input_and_draft_capacity(self):
        ns = self.running_loop(16)
        self.assertEqual(ns['num_scheduled_tokens'], {'A': 8})
        self.assertEqual(ns['input_budget'], 0)
        self.assertEqual([r.request_id for r in self.s.running], ['A', 'B'])

    def test_prefill_cannot_preempt_incumbent_for_kv(self):
        ns = self.running_loop(7168, allocate=lambda r, n, **kw: [] if r is self.a else None)
        self.assertEqual(ns['num_scheduled_tokens'], {'A': 8})
        self.assertEqual([r.request_id for r in self.s.running], ['A', 'B'])
        self.assertFalse(self.p._open_rec['grants'])

    def test_finished_decode_does_not_hold_phantom_input_reservation(self):
        self.a.spec_token_ids = []
        self.a.num_computed_tokens = self.a.num_tokens
        ns = self.running_loop(264)
        self.assertEqual(ns['num_scheduled_tokens'], {'B': 256})
        self.assertEqual(ns['input_budget'], 0)


def hook_table_digests_check(legacy):
    """The installer's legacy hook tables must match their independently recorded digests."""
    for version, table in (('v1', mod.V1_PAIRS), ('v2', mod.V2_PAIRS), ('v5', mod.V5_PAIRS), ('v6', mod.V6_PAIRS)):
        got = hashlib.sha256(json.dumps([list(t) for t in table]).encode()).hexdigest()
        assert got == legacy.HOOK_TABLE_DIGESTS[version], f'{version} hook table differs from its recorded digest'


class HookTableDigests(unittest.TestCase):
    def test_legacy_hook_tables_match_recorded_digests(self):
        spec = importlib.util.spec_from_file_location('legacy_scheduler_helpers_digests', FIXTURES / 'legacy_scheduler_helpers.py')
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        hook_table_digests_check(legacy)


def installation_tests():
    """Installer contract for decode-floor v7 (#283; integrates #246/#221/#180).

    Deployed and fixture producers migrate to the same v7 bytes as a fresh install,
    a second run is a verified no-op, and unsupported (v3/v4), drifted, duplicated,
    marker-only or unmarked inputs are refused without writing (#180's fail-closed
    contract). This is a self-contained subset of the 13-producer / 14-refusal
    matrix run for #283 (summarized in the #283 integration PR).
    """
    src = next((p for p in [Path(os.environ.get('GLM53_SCHEDULER_PY_SRC', '/missing')),
                           Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py'),
                           Path('/tmp/sched-live.py')] if p.is_file()), None)
    if src is None:
        raise SystemExit('Set GLM53_SCHEDULER_PY_SRC to the pinned scheduler source')
    image = src.read_text()
    current = found = None
    if mod.MARK_V7 in image:
        # An already-current v7 install verifies exactly (as main() does); any other
        # v7-marked source falls back to legacy identification, also as main() does
        # (the accepted pre-release v7 identities carry the same marker).
        try:
            clean, current_at, _ = mod._unpatch(image, 'v7', mod.MARK_V7, mod.CLASS_HEAD, mod.V7_PAIRS,
                                                exact=mod._helper_text())
            current = image
        except SystemExit:
            pass
    if current is None:
        found = mod._identify(image)
        clean = image if found is None else found[1]
    assert mod.MARK_V7 not in clean and mod.CLASS_HEAD not in clean

    def load_fixture(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # inspect.getsource needs the module registered
        spec.loader.exec_module(module)
        return module

    legacy = load_fixture('legacy_scheduler_helpers', FIXTURES / 'legacy_scheduler_helpers.py')

    # Every accepted public identity, rebuilt from its byte-exact helper text at the
    # installer's anchor plus that version's hook pairs (the installer authenticates
    # each by sha256 and length before migrating it).
    pairs = {'v1': mod.V1_PAIRS, 'v1-image-d9758a6': mod.V1_PAIRS, 'v2': mod.V2_PAIRS,
             'v5-historical': mod.V5_PAIRS, 'v5-main': mod.V5_PAIRS, 'v5-priority': mod.V5_PAIRS,
             'v6-warm-deadline': mod.V6_PAIRS, 'v6-carry': mod.V6_PAIRS}
    helpers = {'v1': legacy.HELPERS[1], 'v2': legacy.HELPERS[2], 'v5-historical': legacy.HELPERS[5],
               **{k: legacy.HELPERS[k] for k in ('v1-image-d9758a6', 'v5-main', 'v5-priority',
                                                 'v6-warm-deadline', 'v6-carry')}}

    def producer(ident):
        text = clean.replace(mod.NEEDLE, helpers[ident] + mod.NEEDLE, 1)
        for new, old, label in pairs[ident]:
            text = mod.replace_once(text, old, new, label)
        return text

    producers = {'pristine': clean, **{ident: producer(ident) for ident in helpers}}
    hook_table_digests_check(legacy)
    # On the pinned source, every rebuilt legacy install must match its recorded digest:
    # this checks the installer's hook tables against history, not against themselves.
    on_pinned = hashlib.sha256(clean.encode()).hexdigest() == legacy.PINNED_CLEAN_SHA256
    if on_pinned:
        for ident in helpers:
            got = hashlib.sha256(producers[ident].encode()).hexdigest()
            assert got == legacy.SOURCE_DIGESTS[ident], f'{ident}: rebuilt install differs from its recorded bytes'
        # Sensitivity: a wrong hook insertion must be caught by that comparison.
        for ident, table in (('v5-main', mod.V5_PAIRS), ('v6-carry', mod.V6_PAIRS)):
            wrong = tuple((new.replace('_GLM53_MIXED.begin_step(self)', '_GLM53_MIXED.broken_begin_step(self)'), old, label)
                          for new, old, label in table)
            assert wrong != table, ident
            text = clean.replace(mod.NEEDLE, helpers[ident] + mod.NEEDLE, 1)
            for new, old, label in wrong:
                text = mod.replace_once(text, old, new, label)
            assert hashlib.sha256(text.encode()).hexdigest() != legacy.SOURCE_DIGESTS[ident], f'{ident}: digest check is blind to a wrong hook'
    else:
        print('note: scheduler source is not the pinned image source; recorded legacy digests not compared')

    def run(text, temp):
        target = Path(temp) / 'scheduler.py'
        target.write_text(text)
        env = {**os.environ, 'GLM53_SCHEDULER_PY': str(target), 'GLM53_MIXED_PREFILL_CHUNK': 'skip'}
        result = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
        return result, target.read_text()

    installed = None
    with tempfile.TemporaryDirectory() as temp:
        for ident, text in producers.items():
            result, after = run(text, temp)
            assert result.returncode == 0, (ident, result.stderr)
            assert f'from {ident}' in result.stdout, (ident, result.stdout)
            compile(after, ident, 'exec')
            assert mod.MARK_V7 in after, ident
            for older in (mod.MARK_V2, mod.MARK_V3, mod.MARK_V4, mod.MARK_V5, mod.MARK_V6):
                assert older not in after, (ident, older)
            if installed is None:
                installed = after
            assert after == installed, f'{ident} did not migrate to the fresh-install bytes'
            again, repeat = run(after, temp)
            assert again.returncode == 0 and repeat == after and 'already present' in again.stdout, ident

        if current is not None:
            # A deployed v7 keeps its helper where it was installed (e.g. before a later
            # overlay such as adaptive-K), so compare at that position, not the default anchor.
            assert current == mod.apply_v7(clean, at=current_at), 'the installed v7 is not canonical'
            again, repeat = run(current, temp)
            assert again.returncode == 0 and repeat == current and 'already present' in again.stdout
        if found is not None:
            # A supplied legacy install (any accepted identity, including the pre-release v7
            # ones) migrates in place: expect the v7 helper at the legacy helper's position.
            ident, found_clean, found_at = found
            result, after = run(image, temp)
            assert result.returncode == 0 and f'from {ident}' in result.stdout, (ident, result.stderr)
            assert after == mod.apply_v7(found_clean, at=found_at), f'{ident}: migration did not keep the helper position'
            again, repeat = run(after, temp)
            assert again.returncode == 0 and repeat == after and 'already present' in again.stdout, ident

        # Composition with patch_adaptive_k, which inserts its class at the same anchor:
        # the helper may sit before it (decode-floor applied first, as deployed) or after
        # it, and both layouts verify as a no-op.
        if mod.ADAPTIVE_K_HEAD in clean:  # a deployed source already carries the real class
            with_adaptive = clean
            adaptive_at = clean.index(mod.ADAPTIVE_K_HEAD)
        else:
            adaptive = mod.ADAPTIVE_K_HEAD + '  # [glm53-adaptive-k]\n    pass\n\n'
            with_adaptive = clean.replace(mod.NEEDLE, adaptive + mod.NEEDLE, 1)
            adaptive_at = with_adaptive.index(adaptive)
        helper_after = mod.apply_v7(with_adaptive)
        helper_before = mod.apply_v7(with_adaptive, at=adaptive_at)
        assert helper_after.index(mod.CLASS_HEAD) > helper_after.index(mod.ADAPTIVE_K_HEAD)
        assert helper_before.index(mod.CLASS_HEAD) < helper_before.index(mod.ADAPTIVE_K_HEAD)
        for layout, text in (('helper-after-adaptive-k', helper_after), ('helper-before-adaptive-k', helper_before)):
            again, repeat = run(text, temp)
            assert again.returncode == 0 and repeat == text and 'already present' in again.stdout, layout
        # A legacy install migrates in place in either layout: the v7 helper takes the
        # legacy helper's position (public v5-main fixture, so no private source needed).
        for layout, at in (('legacy-after-adaptive-k', None), ('legacy-before-adaptive-k', adaptive_at)):
            legacy_src = (with_adaptive[:at] + helpers['v5-main'] + with_adaptive[at:] if at is not None
                          else with_adaptive.replace(mod.NEEDLE, helpers['v5-main'] + mod.NEEDLE, 1))
            for new, old, label in pairs['v5-main']:
                legacy_src = mod.replace_once(legacy_src, old, new, label)
            result, after = run(legacy_src, temp)
            assert result.returncode == 0 and 'from v5-main' in result.stdout, (layout, result.stderr)
            expected = mod.apply_v7(with_adaptive, at=at) if at is not None else mod.apply_v7(with_adaptive)
            assert after == expected, f'{layout}: migration did not keep the helper position'

        v5 = producers['v5-priority']
        refused = {
            'v3-marker': v5.replace(mod.MARK_V5, mod.MARK_V3),
            'v4-marker': v5.replace(mod.MARK_V5, mod.MARK_V4),
            'v5-hook-drift': v5.replace('_GLM53_MIXED.finish_step(self, scheduler_output)',
                                        '_GLM53_MIXED.finish_step_changed(self, scheduler_output)', 1),
            'v5-duplicate-helper': v5.replace(mod.CLASS_HEAD, mod.CLASS_HEAD + ' pass\n\n' + mod.CLASS_HEAD, 1),
            'v5-marker-only': clean.replace(mod.NEEDLE, mod.MARK_V5 + '\n' + mod.NEEDLE, 1),
            'v7-hook-drift': installed.replace('_GLM53_MIXED.note_alloc_failed(request)  # [glm53-decode-floor:v7]',
                                               '_GLM53_MIXED.note_scheduled(request, 0)  # [glm53-decode-floor:v7]', 1),
            'v7-duplicate-helper': installed.replace(mod.CLASS_HEAD, mod.CLASS_HEAD + ' pass\n\n' + mod.CLASS_HEAD, 1),
            'unmarked-helper': clean.replace(mod.NEEDLE, '\n\nclass _Glm53MixedPrefill:\n    pass\n' + mod.NEEDLE, 1),
            # Mixed state: a canonical v7 plus a leftover older marker must not verify.
            'v7-plus-v5-marker': mod.MARK_V5 + '\n' + installed,
            'v7-plus-v1-marker': installed + '\n' + mod.MARK + '\n',
            # A later duplicate wrapper would rebind the global the hooks call.
            'v7-plus-duplicate-wrapper': installed + '\n\ndef _glm53_mixed_prefill_policy(sched, request, computed=None):\n    return 0\n',
            'pristine-plus-wrapper': clean + '\n\ndef _glm53_mixed_prefill_policy(sched, request, computed=None):\n    return 0\n',
            # An unknown version is not pristine: refuse instead of installing beside it.
            'unknown-marker-v99': '# [glm53-decode-floor:v99]\n' + clean,
            # Only the import anchor or adaptive-K may follow the helper.
            'v7-unknown-text-after-helper': installed.replace(
                mod.NEEDLE, 'class _Glm53OtherOverlay:\n    pass\n\n' + mod.NEEDLE, 1),
            # The overlay's `import os` is part of the applied state: a legacy or current
            # install missing it is drift, not something to repair in place.
            **{f'{ident}-import-removed': text.replace('import os\n', '', 1)
               for ident, text in (('v1-image-d9758a6', producers['v1-image-d9758a6']),
                                   ('v5-main', producers['v5-main']), ('v7', installed))},
        }
        for case, text in refused.items():
            assert text != installed and text != v5, f'{case}: mutation did not apply'
            result, after = run(text, temp)
            assert result.returncode != 0 and after == text, f'{case} was not refused without writing'
            assert 'refusing to rewrite' in result.stderr, f'{case}: not a deliberate refusal: {result.stderr[-200:]}'
    return installed


def main():
    global POLICY, PATCHED_SOURCE
    PATCHED_SOURCE = installation_tests()
    begin = PATCHED_SOURCE.index('class _Glm53MixedPrefill:')
    end = PATCHED_SOURCE.index('_GLM53_MIXED = _Glm53MixedPrefill()')
    ns = {'os': os, 'time': __import__('time')}
    exec(PATCHED_SOURCE[begin:end], ns)
    POLICY = ns['_Glm53MixedPrefill']
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FairTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
