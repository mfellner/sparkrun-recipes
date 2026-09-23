# `tool_choice:"none"` — decode-time enforcement and the live test plan

Status: offline preparation for thegrill issue #55 obligation 1. The patch
(`overlay/patch_tool_choice_none.py`) is implemented and CPU-verified against
the deployed image's parser bytes; **no live behavior is claimed**. The plan
below runs only after the #182 coordinator restores the deployment and
explicitly releases ownership of the serving nodes. No deployment change,
generation request or publication happens before that release.

## What failed and why (established offline)

The 2026-09-16 authorized retest sent factual steps with the shared
`lookup_fact` declaration and `tool_choice:"none"`. The model opened a tool
call anyway; the strict collector failed closed and no measured waves
completed. Read-only extraction of the local image copy
(`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`, vLLM
`0.1.dev20051+g487ecf187`) shows every file in the bad_words enforcement
chain byte-identical to upstream `487ecf187d3dfe74d2cf6119a92881dba403c219`:

- the tool block stays in the prompt under `none` (the shared-prefix premise,
  168-vs-17 prompt-token class) — correct and required;
- the API layer already suppresses *structured* `tool_calls` under `none`
  three ways (`abstract_parser.py` 427/662, `parser_engine.py` 404-414), and
  the engine path also drops the tool span from content — which is exactly
  the "complete tool call, no answer text" symptom;
- nothing stops the model from **generating** `<tool_call>`. That is the
  defect. `--exclude-tools-when-tool-choice-none` (upstream flag) would
  remove the tool block from the prompt and destroy the shared-prefix claim;
  it was rejected for that reason.

## The fix

`Glm47MoeParser.adjust_request` (the hook upstream itself uses for
"none must not leak the tool special token", `online_renderer.py` 442-460)
appends the parser's own `TOOL_CALL_START` to `request.bad_words` when
`tool_choice == "none"` and tools are declared. The v1 sampler masks that
token at every position (`v1/sample/ops/bad_words.py`), including every
speculative draft position on the verify path (`rejection_sampler.py`
329-332); a drafted opener is rejected, costing one acceptance. Prompt
bytes, usage accounting and all other requests are unchanged. The hook runs
for every chat request **because `--reasoning-parser glm45` is set** — if
that flag is ever dropped, the fix goes inert; gate A3 below catches it.

## Live test plan — phases run in order; every gate is fail-stop

All generation phases use the exact grill body shape: `stream:true`,
`stream_options.include_usage:true`, `temperature:0`, `seed:0`,
`max_tokens:128`, `chat_template_kwargs.enable_thinking:false`, the fixed
`lookup_fact` tools array, and a factual question whose answer is in the
conversation rather than behind the tool. Retain every transcript and the
preflight output (`… |& tee preflight-<rank>.log`).

### Phase A–C: pre-generation gates (no requests, no model load)

Run `scripts/tool-choice-none-preflight.py` inside the API container on each rank (`--phase all --model-dir "$MODEL_DIR"`). It checks, in order:

- **A identity** — the live vLLM version is the analyzed `g487ecf187` build;
  the parser module at the **import-resolved path** carries the patch marker
  (reported `APPLIED`/`NOT APPLIED` — unpatched is a valid state for phase D
  only); the live server process was actually launched with
  `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser
  glm45` (the last one is what keeps the hook live).
- **B loaded sources** — sha256 of all thirteen enforcement-chain files
  (parser, serving, protocol, sampling params, input processor, sampler,
  bad-words op, rejection sampler, input batch) against the digests analyzed
  offline; `glm47_moe.py` accepts either the pre-patch digest or the patch
  marker. Any drift: stop and re-verify before anything else.
- **C tokenizer compatibility** — hard gate: `"<tool_call>"` is a vocab
  entry; `encode("<tool_call>", add_special_tokens=False) == [vocab id]`;
  and a `SamplingParams(bad_words=["<tool_call>"]).update_from_tokenizer`
  dry run derives the `[[id]]` mask. A mismatch means the single-token
  opener would stay unmasked — the fix is **inert** for the special-token
  path and needs a token-id mechanism instead; do not proceed.

Before phase A, also record deployment identity per rank: the running image
`repo:tag@digest` (`docker inspect`), which may differ from the analyzed
local copy `sha256:9bb1557a…` — a difference is not an abort by itself, but
phase B drift then decides; plus `MODEL_DIR` and its `tokenizer_config.json`
/ `tokenizer.json` hashes.

### Phase D: request-level hypothesis test (generation; independent of the patch)

This phase verifies the **mechanism hypothesis** through the public request
field `"bad_words":["<tool_call>"]` — the identical sampler path the patch
uses — and is meaningful on an unpatched server (that is its point). It does
not verify the installed patch; phase E does. The patch being installed does
not make D meaningless (the union dedupes).

- **D0 control:** same body, `none`, *no* `bad_words` — reproduce today's
  failure and retain the transcript.
- **D1 — opener-blocked verdict (mechanical):** with request `bad_words`,
  assert no `tool_calls` deltas in the SSE, no `"<tool_call>"` substring in
  any content chunk, and `finish_reason` in {`stop`,`length`}.
- **D2 — useful-answer verdict (separate):** the concatenated content parses
  as the expected fact JSON (or otherwise contains the correct answer).

D1 and D2 are recorded as **separate verdicts**. `D1 pass, D2 fail` (model
emits junk, immediate EOS or an `<|observation|>`-style token) means the
mask works but the model does not answer: record the transcript and hand
the evidence to the serving maintainers — do not add speculative bans.

### Phase E: installed-patch verification (generation; patched server only)

Same body as D **without** request `bad_words`:

- **E0 enforcement evidence:** the server request log for this request shows
  `bad_words=['<tool_call>']` (the request logger prints sampling params) —
  this is the proof the *installed patch* injected the mask, which D cannot
  show.
- **E1/E2:** repeat the D1/D2 split on the SSE.
- **E3 non-regression:** `tool_choice:"auto"` and named `lookup_fact` on a
  tool-worthy prompt still produce structured tool calls; `none` without
  `tools` logs no `bad_words`.

### Phase F: the grill window

Only after A–E pass: the actual shared-tools acquisition through the grill
collector under its own authorized window (thegrill #55), then the
shared-tools profile requalification.

## Not covered by this fix (unchanged, do not silently narrow)

- Beam search combined with `none` (no `bad_words` on `BeamSearchParams`).
- The Responses API with `none`+tools (no request `bad_words` field; the
  hook guards with an isinstance check).
- The opener spelled out with ordinary tokens beyond the phrase's tokenized
  encodings; `update_from_tokenizer` masks the encoded forms only.
- The hook goes inert if `--reasoning-parser` is dropped or the tool parser
  changes (gate A3).
- Pre-existing API-shape suppression of structured `tool_calls` is not an
  answer guarantee; prompt-token count equality is not rendered-token
  identity (equal counts can hide substitutions).

## Files

- `overlay/patch_tool_choice_none.py` — the patcher (fail-closed anchors on
  this image's `vllm/parser/glm47_moe.py`; idempotent; `--status`).
- `scripts/tool-choice-none-preflight.py` — phases A–C pre-generation gates
  (invoked manually per this plan; deliberately not wired into the build).
- `tests/test_tool_choice_none.py` — CPU regression (apply/idempotence/drift,
  none+tools masking, client `bad_words` union, other requests untouched,
  launcher/build wiring).
- Wiring: `Dockerfile` (COPY/RUN/test chain) and `start.sh`
  (`TOOLCHOICE_PATCH_HOST`, `GLM53_OVERLAY_ORDER`, integrity table, scp,
  both mounts).

## Execution record (2026-09-18, owner-authorized bounded window)

Prerequisites: all gates passed (identity `glm53-upstream:8a14501884d6` /
`sha256:2e579cb2…` both ranks, version `0.1.dev20051+g487ecf187`; 13/13
source digests under the owner-approved amended baseline — the deployed
`sampling_params.py` carries the repo's own `[glm53-apc-no-store]` overlay,
insert-only, zero bad_words-chain changes; tokenizer gate via the engine's
`get_tokenizer` path: `<tool_call>` = single special token 154843,
`update_from_tokenizer` derives `[[154843]]`).

Stage D (unpatched): D0 control reproduced the documented failure exactly
(12 completion tokens swallowed, empty content, `stop_reason` 154829 = the
observation boundary token; prompt 175). D1 opener-blocked PASS and D2
useful-answer PASS (`{"fact": "sapphire"}`) with request-level
`bad_words:["<tool_call>"]`, prompt unchanged at 175.

Stage E (patched, docker-cp install + restart both ranks): E0 PASS with no
request-level `bad_words` (server-side mask; prompt 175); E3 non-regression
PASS (auto and named still produce the correct `lookup_fact` call;
`none`-without-tools answers in text, prompt 24).

Stage F (released grill-perf 0.3.0, checksum-verified, pinned shared-tools
workload): waves 0–1 both answered correctly with shared tools in the
prompt — the original blocker is gone — but the reuse step reported
`cached_prompt_tokens: 0` (`provider_reported_prefix_cache_miss`) and the
collector stopped fail-closed at wave 1/44. That is the separate,
still-open required-hit obligation (never yet passed on GLM), not a patch
defect; retained as-is.

Restoration: canonical `glm47_moe.py` (`ce362931…`) restored both ranks,
containers restarted, health 200, 13/13 digests back to baseline, patch
NOT APPLIED, and a final control request reproduced the original failure
signature byte-identically. Total generation requests: 11 (one 16-token
diagnostic probe during the D0 transport-artifact investigation, outside
the declared stages — reported as a deviation). Evidence retained under
`/tmp/issue55-live/` on the head node and mirrored locally.
