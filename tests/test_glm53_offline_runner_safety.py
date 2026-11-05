import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mods/glm-5.3-flash-exl3-upstream-850k"
RUNNER = MOD / "run_upstream_compatibility_suite.sh"

class OfflineRunnerSafetyTests(unittest.TestCase):
    def test_requires_opt_in_explicit_cap_headroom_and_live_check(self):
        text = RUNNER.read_text()
        for token in ("GLM53_RUN_CPU_SUITE", "GLM53_TEST_MEMORY", "MemAvailable", "required_kib", "docker ps", "--memory-swap \"$MEMORY\""):
            self.assertIn(token, text)
        self.assertNotRegex(text, r'GLM53_TEST_MEMORY:-')

    def test_exact_empty_modules_have_live_or_standalone_explanation(self):
        allowed = {
            "test_ablit.py": ("torch", "def main", "check_transplant"),
            "test_adaptive_k_patch.py": ("def main", "SCHED_SRC", "policy_tests"),
            "test_dense_fp8_patch.py": ("def main", "KDA_SRC", "classifier_tests"),
            "test_exl3_overlay.py": ("def main", "EXL3_SELFCHECK_GPU", "_check_ext_arch"),
            "test_mixed_prefill_decode.py": ("argparse", "127.0.0.1:8888", "stream_one"),
            "test_scheduler_decode_floor_restart.py": ("def main", "synthetic scheduler", "build_clean_scheduler"),
            "test_tool_concurrency.py": ("argparse", "127.0.0.1:8888", "TEST_TOOLS"),
        }
        body = RUNNER.read_text()
        listed = set(re.findall(r'tests/test_[^|; )]+\.py', body))
        names = {Path(p).name for p in listed}
        self.assertTrue(set(allowed) <= names, f"runner lacks exception(s): {set(allowed)-names}")
        self.assertIn("UNEXPECTED EMPTY MODULE", body)
        for name, needles in allowed.items():
            source = (MOD / "upstream/tests" / name).read_text()
            for needle in needles:
                self.assertIn(needle, source, f"{name} no longer matches role: {needle}")

if __name__ == "__main__":
    unittest.main()
