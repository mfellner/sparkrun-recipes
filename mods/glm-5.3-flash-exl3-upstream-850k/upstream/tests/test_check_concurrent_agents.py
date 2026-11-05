#!/usr/bin/env python3
"""Argument validation of tests/check_concurrent_agents.py (no network: rejected before any request)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

CANARY = Path(__file__).resolve().parent / "check_concurrent_agents.py"


class BaseUrlValidation(unittest.TestCase):
    def rejected(self, url):
        result = subprocess.run([sys.executable, str(CANARY), "--base-url", url, "--timeout", "1"],
                                capture_output=True, text=True, timeout=60)
        return result.returncode == 2 and "without credentials" in result.stderr

    def test_userinfo_in_any_form_is_rejected(self):
        for url in ("http://user@127.0.0.1:8888/v1", "http://user:pw@127.0.0.1:8888/v1",
                    "http://:secret@127.0.0.1:8888/v1", "http://@127.0.0.1:8888/v1"):
            with self.subTest(url=url):
                self.assertTrue(self.rejected(url))

    def test_query_fragment_and_scheme_are_rejected(self):
        for url in ("http://127.0.0.1:8888/v1?x=1", "http://127.0.0.1:8888/v1#f", "ftp://127.0.0.1/v1"):
            with self.subTest(url=url):
                self.assertTrue(self.rejected(url))


if __name__ == "__main__":
    unittest.main()
