"""Regression contracts for local test-only compatibility adapters."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class AdapterTests(unittest.TestCase):
    def test_gpu_adapter_preserves_current_assertions_and_rejects_drift(self):
        path = ROOT / "run_upstream_gpu_compatibility.py"
        self.assertTrue(path.is_file(), "local GPU adapter is required")
        spec = importlib.util.spec_from_file_location("gpu_adapter", path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        source = (ROOT / "upstream/tests/test_exl3_overlay.py").read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_check_dflash2")
        original = ast.get_source_segment(source, fn)
        adapted = adapter.adapt_dflash_check(original)
        self.assertEqual(adapted.replace("    _check_current_draft_kv()\n", ""), original)
        with self.assertRaises(ValueError):
            adapter.adapt_dflash_check(original.replace("compact_block = _glm53_draft_block_size(", "compact_block = 65"))
        with self.assertRaises(ValueError):
            adapter.adapt_dflash_check(original + adapter.CURRENT_SEAM)

    def test_fixture_only_fills_missing_paths(self):
        from upstream_fixture_adapter import complete_preflight_env, MAMBA_INPUTS
        env = {"STOP_PATCH_HOST": "/fixture/placeholder", MAMBA_INPUTS[0]: "/missing"}
        fixed = complete_preflight_env(env)
        self.assertEqual(fixed[MAMBA_INPUTS[0]], "/missing")
        self.assertEqual(fixed[MAMBA_INPUTS[1]], "/fixture/placeholder")
        self.assertNotIn(MAMBA_INPUTS[1], env)

    def test_tp3_fixture_only_substitutes_original_stock(self):
        from upstream_fixture_adapter import TP3_SHA, TP3_STOCK, tp3_fixture_input
        import hashlib
        original = ROOT / "upstream/overlay/exl3.py"
        other = ROOT / "other.py"
        self.assertEqual(tp3_fixture_input(original, original), TP3_STOCK)
        self.assertEqual(tp3_fixture_input(other, original), other)
        self.assertEqual(hashlib.sha256(TP3_STOCK.read_bytes()).hexdigest(), TP3_SHA)
        self.assertNotEqual(hashlib.sha256(original.read_bytes()).hexdigest(), TP3_SHA)


if __name__ == "__main__":
    unittest.main()
