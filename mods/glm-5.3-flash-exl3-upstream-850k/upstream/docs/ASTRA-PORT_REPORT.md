# Astra review: GLM cooperative MoE port

Review date: 2026-09-16. Updated after the follow-up comparison with Fable's
assessment and direct reanalysis of the September 8 stock and September 16
cooperative receipts. This report describes the reviewed working tree and
serving evidence available on that date. Recommendations are not implemented
by this report.

**Assessment: the port is structurally faithful and internally consistent for
the intended GLM TP2 configuration, but the configuration now serving is not
fully validated.** No shape-capacity or arithmetic-porting defect was identified
by static review. One runtime initialization bug was reproduced with CPU stubs.
The remaining concerns are validation coverage, numerical fidelity, execution
assumptions, and the strength of the performance claims.

The starting point was [the operator handoff](cooperative-moe-handoff.md).
The upstream comparison used
[DeepSeek PR #8](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/pull/8)
at head commit `e7e06bb88c437c4aaa69375d2c93892f68cdafc2`. The native files were
compared against that PR, rather than assuming the neighboring DeepSeek checkout
was identical. Its local adapter differs from the PR in the pinned binary hash.

**What looks correct**

The native kernel body preserves the original implementation's arithmetic;
changes are the namespace, fixed GLM parameters, ABI names, and host-side
specialization. This does not mean cooperative arithmetic is bit-exact with
stock EXL3, or that changing tile geometry preserves numerical results.

| Invariant | Reviewed value |
|---|---|
| Hidden width | 4096 |
| Local intermediate width | 1024, from packed 2048 with TP2 |
| Routing | Top-k 8, at most 288 local experts |
| Quantization | K4 MCG, codebook 1, no mul1 |
| Cooperative row limit | 32 |
| Routed-slot capacity | 256 = 32 × 8 |
| Counter/run-table allocation | 3,587 int32 entries |
| Native interface | 20 pointers; parameter structure checked as 344 bytes |
| Current selected geometry | 1: A-wide/B-wide |

These values agree between [the adapter](../extensions/cooperative_moe/runtime.py),
[the native launcher](../extensions/cooperative_moe/native/cooperative_moe.cu),
and [the kernel](../extensions/cooperative_moe/native/cooperative_moe_kernel.cuh).
The counter allocation is `2048 + 1024 + 2 + 257 + 256 = 3587`.

The run builder splits an expert's selected slots into groups of at most eight.
Concentrated routing across 32 rows therefore does not require a 32-row MMA tile.
The rank key `(expert << 8) | slot` accommodates slot indices 0–255; 256 slots
is a hard ceiling for the present representation and arrays. The adapter and
native launcher both reject larger cooperative batches. Raising this limit
would require more than increasing scratch allocation.

Pointer ordering, output dtype conversion, invalid-route mapping, and the
no-retry behavior after a native launch failure are consistent with the overlay.
Scratch is allocated after weight loading, before graph capture. Calls above
32 rows return to the stock dispatcher, which retains its E3 path. Dispatch is
based on shape, however: a small prefill can also qualify, so "decode-only" is
an operational description rather than a semantic prefill/decode check.

**Findings and required follow-up**

1. **Confirmed bug: optimized Python skips required native initialization.**

   In `CoopLaunch.__init__`, the statement
   `assert self.library.glm53_coop_info(BITS, GEOMETRY, info) == 0` performs
   initialization inside an assertion. Python optimization removes the call
   itself. The native `prepared[geometry]` flag remains false and the first
   cooperative launch returns `cudaErrorInvalidValue`. SHA, ABI, capability,
   capture-state, and layout checks also disappear.

   An in-memory CPU-stub reproduction observed native initialization calls
   `['abi', 'info']` with normal compilation and `[]` with optimized compilation.
   This issue is inherited from the original adapter. It does not demonstrate
   a failure in the current normally launched service.

   Move initialization calls outside assertions and use explicit exceptions
   for required runtime checks. Add a regression covering optimized execution.
   Source: [runtime.py](../extensions/cooperative_moe/runtime.py),
   `CoopLaunch.__init__`; native readiness check in
   [cooperative_moe.cu](../extensions/cooperative_moe/native/cooperative_moe.cu).

2. **Validation gap: geometry 1 and several live adaptive-k shapes lack the
   recorded GPU gate.**

   The handoff records a head-node gate on geometry 2, while serving uses
   geometry 1. Changing B from narrow to wide selects another template
   instantiation and changes the reduction layout. A geometry-1 repeat should
   be required validation, not described as optional. The worker-node gate is
   also outstanding in the handoff.

   The inspected head configuration captures rows
   `1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 32`.
   The packaged GPU test covers rows `1, 2, 4, 7, 8, 16, 24, 32, 40`.
   Missing live shapes are **3, 5, 6, 9, 10, 12, 15, 20**. Rows 3 and 5 are
   particularly relevant to adaptive verification lengths 2 and 4.

   Run the current geometry on both nodes, including alternating shapes and
   layers sharing scratch, concentrated/spread routing, mixed valid/invalid
   routes, zero-weight routes, and transitions back from empty routing. Include
   a production-sized 288-expert table and the 32/33-row dispatch boundary.
   Assert that the larger concentrated fixture actually executes grouped E3;
   merely observing no cooperative launch does not establish that.

   Sources: [GPU gate](../extensions/cooperative_moe/test_cuda_integration.py),
   [capture-size generation](../start.sh), and
   [handoff validation status](cooperative-moe-handoff.md).

3. **Validation gap: the current numerical screen does not establish model
   fidelity.**

   The test accepts `max(abs(actual - reference)) / max(abs(reference)) <= 0.003`
   over the whole output tensor. A large reference element can mask substantial
   relative errors in smaller elements or rows. The fixtures use synthetic
   weights and 32 experts, with stock fused output as the reference.

   The report records 6,473,169 strict raw differences and 4,140,219 post-BF16
   differences across repeated comparisons. These counts were not independently
   reproduced in this review. Without denominators, distributions, and
   downstream effects, they establish neither acceptable fidelity nor a defect.

   Add actual GLM weights and captured activations, a small independent dense
   reference, per-row error statistics, relative L2 error, and downstream
   logits/task-quality checks. Retain the existing peak-normalized screen as
   one diagnostic rather than the complete acceptance criterion.

   Run the quality suite on stock and cooperative serving with the same
   prompts and scoring. Evaluate generated assistant text or fixed assistant
   continuations; user-turn logprobs alone do not establish assistant-output
   quality. Behavioral scoring complements real-weight activation comparisons:
   the former measures task effects, while the latter helps locate arithmetic
   discrepancies.

   The activation limit is **used for ordinary SiLU**. `act_gate` computes SiLU,
   clamps the activated gate from above and the up projection symmetrically
   when `limit != 0`, then multiplies them. The overlay defaults to `10.0`.
   Tests should deliberately exercise clipping and the zero-limit case.

   Sources: [comparison code](../extensions/cooperative_moe/test_cuda_integration.py),
   [act_gate](../extensions/cooperative_moe/native/cooperative_moe_kernel.cuh),
   and [reported numerical results](cooperative-moe.md).

4. **Missing safety evidence: repeat bounded sanitizer checks for the GLM
   specialization.**

   The original PR's sanitizer evidence does not validate this K4/MCG,
   256-slot specialization. Add compute-sanitizer memcheck and racecheck runs
   covering the current geometry, maximum slots, concentrated routing, and
   invalid/empty routes. Initcheck and synccheck would also be useful targeted
   checks for scratch initialization and synchronization.

   A clean racecheck result is not proof that global scratch is safe across
   concurrent streams: racecheck detects shared-memory access hazards.
   See [NVIDIA's Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
   The execution contract below still needs independent review and enforcement.

5. **Execution assumption: scratch is shared per device, so cooperative calls
   must not overlap.**

   `install()` caches one `CoopLaunch` per device and attaches it to every
   eligible layer. The workspace is separate from stock scratch, but is not
   private to each layer, invocation, or CUDA stream. The stock overlay also
   caches fused temporary buffers across layers in `build_exl3_fused_state`;
   serialized workspace reuse is an existing execution requirement, not a
   concern introduced only by this port.

   Kernel A resets B's counters; kernel B resets A's counters for the next call.
   This supports ordered reuse and sequential graph replay. It does not make
   overlapping cooperative invocations safe. Using the current stream alone
   does not serialize calls made on different streams.

   No current serving race was demonstrated. Shared experts being computed
   outside this adapter also does not, by itself, violate the workspace
   contract. The original DS-specific stream restrictions should not be copied
   blindly. Document the actual non-overlap requirement and reject unsupported
   ubatching or dual-batch overlap at configuration time. Separate workspaces
   and correct synchronization would be needed to support overlapping calls;
   per-layer replication is not the preferred fix on this memory-constrained
   configuration.

6. **Measurement limitation: existing receipts do not isolate the claimed
   kernel speedup.**

   The saved geometry comparisons show:

   | Metric | Geometry 2 | Geometry 1 |
   |---|---:|---:|
   | Prose median throughput | 34.81 tok/s | 37.40 tok/s |
   | Prose accepted drafts per step | 2.059 | 2.256 |
   | Structured median throughput | 78.01 tok/s | 80.01 tok/s |
   | Structured accepted drafts per step | 7.0 | 7.0 |

   Direct reanalysis of the receipts gives the following additional comparison.
   Each entry is the median across runs of
   `1000 * run["decode_s"] / run["spec"]["drafts"]`, rounded to two decimals.

   | Workload | Stock, September 8 | Geometry 2 | Geometry 1 |
   |---|---:|---:|---:|
   | Prose: elapsed decode ms per draft step | 94.83 | 87.28 | 86.72 |
   | Structured: elapsed decode ms per draft step | 112.11 | 102.04 | 99.49 |
   | Prose output tokens per run | 400 | 400 | 400 |
   | Structured output tokens per run | **400** | **200** | **200** |

   This uses `decode_s`, which excludes time to first content, rather than
   total request `wall_s`. The draft counter spans the request, so the timing
   and counter boundaries are not perfectly aligned. These are approximate
   serving-cycle measurements, not CUDA kernel timings. The stock columns use
   the linked lab receipts below, not the historical sparkDash headline.

   Prose throughput increases about 7.4% between geometries, alongside higher
   acceptance, while elapsed decode time per draft step changes by less than
   1%. The structured comparison holds acceptance constant and shows about
   2.6% throughput improvement. With only two cooperative runs per workload,
   these measurements do not establish a substantial prose kernel advantage
   for geometry 1. The handoff's narrow-B explanation overstates the evidence.

   Compared with the historical stock receipts, geometry 1 has approximately
   8.5% lower prose time per step and 11.3% lower structured time per step.
   These observations suggest an improvement, but do not isolate a causal
   cooperative-kernel gain. In particular, **the stock structured runs are
   400 tokens and the cooperative runs are 200 tokens**. Normalizing by steps
   does not fully remove the different output lengths and context progression.

   Dividing by steps also reduces, rather than eliminates, the acceptance
   confound. Adaptive-k changes the amount of verification work per step, and
   different generated text changes subsequent routing and computation. The
   ratio includes drafting, attention, communication, and other serving work.
   Describe the observed acceptance change as an acceptance difference; the
   receipts do not establish that it is entirely random variance. Matching a
   few flags across different dates is insufficient to attribute the headline
   +16% throughput difference to the kernel.

   A supported replacement for the geometry claim is:

   > Geometry 1 improved structured throughput by about 2.6% in two recorded
   > runs. Prose throughput increased alongside speculative acceptance, while
   > average decode time per draft step changed by less than 1%. These
   > measurements do not establish a substantial prose kernel advantage over
   > geometry 2.

   For deployed serving, run stock/cooperative/stock restarts with the same
   model, prompts, output lengths, image, and configuration except the selected
   implementation. Include ×2 concurrency and report elapsed ms per step,
   acceptance, and adaptive verification-size distribution separately. Preserve
   the receipts and configuration provenance for each arm.

   Separate boots are the practical protocol for this launcher, not a CUDA
   requirement. A controlled same-process microbenchmark can capture stock and
   cooperative graphs separately on identical inputs, then use CUDA events to
   isolate MoE execution time. Swapping a Python function after capture does
   not alter an existing graph. Claims that routing offers negligible weight
   reuse, or that all remaining opportunity is outside MoE, need measured
   routing overlap and profiling.

   Receipts: [stock prose](../logs/decode-live-20260908/prose-adaptk-fp8.json),
   [stock structured](../logs/decode-live-20260908/structured-adaptk-fp8.json),
   [prose geometry 1](../logs/coop-decode-prose-geo1.json),
   [prose geometry 2](../logs/coop-decode-prose-geo2.json),
   [structured geometry 1](../logs/coop-decode-structured-geo1.json), and
   [structured geometry 2](../logs/coop-decode-structured-geo2.json).
   These are local evidence files and may not be included in a published checkout.

**Operational improvements**

- Log successful native preparation, eligible layer counts, actual geometry,
  and fallback reasons. The current "enabled" message prints when wrappers
  are installed, before any layer qualifies. The handoff's log check therefore
  proves installation, not execution of the cooperative path.
- A Python diagnostic counter can count eager calls and capture-time selection;
  it will not count graph replays. Log, for each capture size, whether the
  cooperative implementation was selected. That identifies what the captured
  graph contains without requiring a device counter. Use kernel traces or
  optional device-side instrumentation when live replay evidence is required.
  GPU utilization alone cannot identify the executing MoE implementation. The
  existing integration gate already provides stronger fixture-level evidence
  by intercepting `native.launch`.
- A geometry environment variable is reasonable, but select and validate it
  before native preparation and graph capture, forward it to both ranks, and
  log the corresponding tile names. This can avoid repinning the adapter for
  each experiment. Changing geometry still requires preparing the selected
  kernels and recapturing graphs or restarting; it is not a live graph switch.
- Correct the stale statement in `cooperative-moe.md` that distributed serving
  remains open. Distinguish the completed two-node serving smoke test from the
  outstanding worker numerical gate and broader distributed validation.
- Revise the handoff and benchmark report's geometry narrative to reflect the
  measured per-step results and acceptance differences. Identify historical
  comparisons and their output-length mismatch explicitly.
- Preserve the exact validated binary and build manifest in an artifact
  archive, including image digest, toolchain, source pins, build command, and
  checksums. Matching the staged hash does not prove a clean rebuild reproduces
  it. This review did not test reproducibility and does not establish that the
  binary is unreproducible. Before publication, ensure intended sources and
  docs are committed; keep binaries, credentials, and checkpoints out of the
  source change.
- Add compile-time checks around the slot ceiling and derived capacities to
  make future parameter changes fail explicitly rather than silently exceeding
  the run builder's representation.

**Checks performed and limits of this review**

The review ran 49 CPU-stub dispatch checks, eight profile-integrity tests,
Python parse checks, and shell syntax checks for the build script and launcher;
all passed. The optimized-Python initialization defect was reproduced using
in-memory CPU stubs. Native sources were diffed against the PR. Read-only
inspection covered the head's serving configuration and relevant vLLM source.
The follow-up review recalculated per-step metrics from the saved stock and
cooperative JSON receipts and checked their output lengths. This added no new
serving experiment.

The head's staged artifacts matched the documented hashes:

| Artifact | SHA-256 |
|---|---|
| Stock overlay | `fe07cf3cd1928d0a189e793579a7d2dd529f75617a55f620ee14a0a9d3b20121` |
| Adapter | `71111c230c2519d473cc77f215703fbb12b1be2659cde66aa166c98ea3952fc6` |
| Native library | `aa3fe5e9387c7e0d42d685fb2ca8a5fb959ad956600236baac078a9076c17a1c` |
| Generated overlay | `5f28f5543629043117c7506cd3cf47cc8fa66fcd1153ddf67c696bf3e5484c4d` |

The review did not rebuild the binary, run GPU workloads or sanitizers, verify
worker artifacts independently, or restart either service. Existing GPU and
serving results are identified as recorded evidence, not newly executed checks.

Recommended order:

1. Move required initialization and integrity checks out of assertions, and
   enforce the supported serialized execution configuration.
2. Gate geometry 1 on both nodes with the full capture-size row list, explicit
   E3 verification for the oversized case, and bounded memcheck/racecheck runs.
   Preserve the gate and sanitizer logs.
3. Compare stock/cooperative real-weight arithmetic and assistant-output
   quality; retain both numerical and behavioral evidence.
4. Run matched stock/cooperative/stock serving restarts, including ×2, reporting
   ms per step and acceptance separately. Use a fixed-input microbenchmark for
   kernel attribution.
5. Correct geometry and validation claims in the docs and preserve artifact
   and image provenance.
6. Log prepared-layer count and actual geometry after weight loading, plus
   cooperative selection for each capture size. Add geometry configuration
   only with preparation, rank forwarding, and graph lifecycle handled.

## Follow-up experiments (2026-09-16)

This section is **not** part of the static review above. The original
assessment did not run GPU gates, sanitizers, or serving A/B/A. Those were
executed later on this kit. Details and tables:
[astra-results.md](astra-results.md),
[astra-experiment-ledger.md](astra-experiment-ledger.md). Frozen gates
remain [astra-acceptance.md](astra-acceptance.md).

What was done after the review:

1. Required init moved out of `assert`; scratch contract enforced; prepared
   layer count and per-capture-size selection logged; `GLM53_COOP_GEOMETRY`
   selected before prepare and forwarded to both ranks.
2. Geometry-1 GPU gate on **both** Sparks, including live sizes 3 and 5;
   rows 33/40 concentrated executed E3 grouped. Bounded memcheck/racecheck
   on both nodes (0 errors / 0 racecheck hazards).
3. Assistant-output quality on stock and C1 (`tests/eval_coop_quality.py`).
4. Matched stock/C1/stock serving, independent C1 boot, then C1-final
   revalidation (structured and prose ×2, mixed ~8k prefill, 12 sequential
   requests). Kernel microbench: geo1 fastest at 32 rows; geos tied at 3/5.
5. Docs updated to separate **C1 vs stock improvement** from **no further
   kernel win over the initial geo1 `.so`**.

Live C1 hashes (head and worker): `.so` `aa3fe5e9…` (same as review), C1
adapter `9427f6a6…`, generated overlay `5f28f554…`. Rollback:
`logs/astra-goal-20260916/rollback-{c1,geo1,stock}.sh`.

