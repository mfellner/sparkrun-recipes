"""Deterministic, CPU-only replay of the pinned image's production detokenizer.

Run with Python -S in the recipe's exact image, no GPU, network=none, memory=768m.
Mount an already composed exact-image fixture read-only and set
GLM53_PATCHED_FIXTURE to its root (containing site/vllm). No model/API is used.
Normal repository collection skips unless explicitly opted in.

Reproduce from the repository root (FIXTURE is the composed source directory)::

    docker run --rm --network none --memory 768m --memory-swap 768m \
      --runtime runc --read-only --cap-drop ALL --security-opt no-new-privileges \
      -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONDONTWRITEBYTECODE=1 \
      -e GLM53_PATCHED_FIXTURE=/fixture -v "$FIXTURE:/fixture:ro" \
      -v "$PWD/tests/test_glm53_detokenizer_replay.py:/test.py:ro" \
      --entrypoint python3 \
      ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d \
      -S /test.py

RED/GREEN uses disposable source copies with targeted mutations, then the
untouched patched fixture; neither the serving runtime nor recipe is modified.

The entire production detokenizer module AND detokenizer_utils are imported,
not extracted/reimplemented. HuggingFace tokenizers/DecodeStream are real.
Only import-heavy vLLM infrastructure and the Transformers backend marker type
are stubbed. A tiny deterministic WordLevel/Fuse vocabulary supplies fixed
pieces; this is a stop-policy replay, not GLM tokenizer or inference acceptance.
"""
import importlib.util
import logging
import os
from pathlib import Path
import sys
import types
import unittest


class DetokenizerReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("GLM53_PATCHED_FIXTURE")
        if not root:
            raise unittest.SkipTest("requires opt-in composed exact-image fixture")
        if not sys.flags.no_site:
            raise RuntimeError("run standalone with python3 -S (no .pth/GPU imports)")
        if not Path("/.dockerenv").exists() or list(Path("/dev").glob("nvidia*")):
            raise RuntimeError("requires disposable CPU-only container")
        # Add the pinned image's packages WITHOUT running site/.pth hooks.
        sys.path.append("/usr/local/lib/python3.12/dist-packages")
        from tokenizers import Tokenizer, decoders, models

        class Backend:
            def __init__(self, tokenizer):
                self._tokenizer = tokenizer

            def convert_tokens_to_ids(self, text):
                return self._tokenizer.token_to_id(text)

        def namespace(name, **attributes):
            module = types.ModuleType(name)
            module.__path__ = []
            module.__dict__.update(attributes)
            sys.modules[name] = module
            return module

        namespace("transformers", TokenizersBackend=Backend)
        namespace("vllm")
        namespace("vllm.logger", init_logger=logging.getLogger)
        namespace("vllm.tokenizers", TokenizerLike=Backend)
        # Not executed by the fast path; fail rather than emulate prompt logic.
        def unused(*args):
            raise AssertionError("unexpected slow-path infrastructure dependency")
        namespace("vllm.utils", length_from_prompt_token_ids_or_embeds=unused)
        namespace("vllm.v1")
        namespace("vllm.v1.engine", EngineCoreRequest=types.SimpleNamespace)
        namespace("vllm.config", get_current_vllm_config_or_none=lambda: None)
        source = Path(root) / "site/vllm"

        def load(name, relative):
            spec = importlib.util.spec_from_file_location(name, source / relative)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        load("vllm.tokenizers.detokenizer_utils", "tokenizers/detokenizer_utils.py")
        cls.production = load("vllm.v1.engine.detokenizer", "v1/engine/detokenizer.py")
        pieces = ["[UNK]", "PROMPT", "<think>", "BEFORE ", "Question", ":", " AFTER",
                  "</", "think", ">", " ANSWER ", "<eos>"]
        cls.vocab = {piece: i for i, piece in enumerate(pieces)}
        tokenizer = Tokenizer(models.WordLevel(cls.vocab, unk_token="[UNK]"))
        tokenizer.decoder = decoders.Fuse()
        cls.tokenizer = Backend(tokenizer)
        os.environ["GLM53_SUPPRESS_STOPS_IN_REASONING"] = "1"
        assert cls.production.USE_FAST_DETOKENIZER
        assert "torch" not in sys.modules

    def new_detokenizer(self, reasoning_ended):
        params = types.SimpleNamespace(stop=["Question:"], min_tokens=0,
            include_stop_str_in_output=False, skip_special_tokens=False,
            spaces_between_special_tokens=True)
        request = types.SimpleNamespace(request_id="fixed-replay", sampling_params=params,
            prompt_token_ids=[self.vocab["PROMPT"]], prompt_embeds=None,
            reasoning_ended=reasoning_ended)
        detok = self.production.IncrementalDetokenizer.from_new_request(self.tokenizer, request)
        self.assertIsInstance(detok, self.production.FastIncrementalDetokenizer)
        return detok

    def replay(self, pieces, boundaries, reasoning_ended=True, eos=False):
        detok = self.new_detokenizer(reasoning_ended)
        deltas = []
        reason = None
        offset = 0
        for end in boundaries:
            chunk = [self.vocab[piece] for piece in pieces[offset:end]]
            stop_terminated = eos and end == len(pieces)
            reason = detok.update(chunk, stop_terminated=stop_terminated)
            deltas.append(detok.get_next_output_text(
                finished=reason is not None or stop_terminated, delta=True))
            offset = end
            if reason is not None or stop_terminated:
                break
        return detok, reason, "".join(deltas)

    def test_disabled_reasoning_truncates_exactly_at_client_stop(self):
        pieces = ["BEFORE ", "Question", ":", " AFTER"]
        for boundaries in ([1, 2, 3, 4], [4], [1, 4], [2, 4], [3, 4]):
            with self.subTest(boundaries=boundaries):
                detok, reason, streamed = self.replay(pieces, boundaries)
                self.assertEqual(reason, "Question:")
                self.assertEqual(detok.output_text, "BEFORE ")
                self.assertEqual(streamed, "BEFORE ")
                self.assertEqual(detok.get_next_output_text(True, False), "BEFORE ")
                self.assertFalse(detok._reasoning_stop_guard)

    def test_eos_before_marker_is_a_separate_valid_terminal(self):
        # The engine supplies stop_terminated for EOS; the detokenizer must NOT
        # invent a matched client stop. This does not test engine EOS detection.
        pieces = ["BEFORE ", "<eos>"]
        for boundaries in ([1, 2], [2]):
            with self.subTest(boundaries=boundaries):
                detok, reason, streamed = self.replay(pieces, boundaries, eos=True)
                self.assertIsNone(reason)
                self.assertEqual(detok.output_text, "BEFORE ")
                self.assertEqual(streamed, "BEFORE ")
                self.assertEqual(detok.output_token_ids[-1], self.vocab["<eos>"])
                self.assertNotIn("Question:", streamed)

    def test_open_reasoning_keeps_client_stop_dormant(self):
        pieces = ["BEFORE ", "Question", ":", " AFTER"]
        for boundaries in ([1, 2, 3, 4], [4], [2, 4]):
            with self.subTest(boundaries=boundaries):
                detok, reason, streamed = self.replay(
                    pieces, boundaries, reasoning_ended=False)
                self.assertIsNone(reason)
                self.assertEqual(detok.output_text, "BEFORE Question: AFTER")
                self.assertTrue(detok._reasoning_stop_guard)
                self.assertFalse(detok._reasoning_closed)
                # Flush the production streaming buffer as a length terminal.
                self.assertEqual(streamed + detok.get_next_output_text(True, True),
                                 "BEFORE Question: AFTER")

    def test_closing_marker_resumes_stops_across_chunk_boundaries(self):
        pieces = ["BEFORE ", "Question", ":", "</", "think", ">",
                  " ANSWER ", "Question", ":", " AFTER"]
        expected = "BEFORE Question:</think> ANSWER "
        # Every two-chunk split, individual tokens, and one speculative chunk:
        # splits inside </think>, inside Question:, and a close+stop same step.
        partitions = [list(range(1, len(pieces) + 1)), [len(pieces)]]
        partitions += [[cut, len(pieces)] for cut in range(1, len(pieces))]
        for boundaries in partitions:
            with self.subTest(boundaries=boundaries):
                detok, reason, streamed = self.replay(
                    pieces, boundaries, reasoning_ended=False)
                self.assertEqual(reason, "Question:")
                self.assertEqual(detok.output_text, expected)
                self.assertEqual(streamed, expected)
                self.assertEqual(detok.get_next_output_text(True, False), expected)
                self.assertTrue(detok._reasoning_closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
