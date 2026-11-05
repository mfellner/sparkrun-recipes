#!/usr/bin/env python3
"""The Dockerfile stages the decode-floor installation runner at /opt/glm53 and runs it
at build time; this stages that same layout from the Dockerfile's COPY lines, with no
checkout-relative paths available, and checks the runner finds its overlay and fixture.
With GLM53_SCHEDULER_PY_SRC set, the full runner is executed in that layout as well."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "tests/test_scheduler_decode_floor.py"


def staged_copies():
    """(source, destination) pairs for every COPY into /opt/glm53 in the Dockerfile."""
    pairs = []
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        m = re.match(r"COPY\s+(.+?)\s+(/opt/glm53\S*)\s*$", line)
        if not m:
            continue
        sources, dest = m.group(1).split(), m.group(2)
        for src in sources:
            target = dest + Path(src).name if dest.endswith("/") else dest
            pairs.append((src, target))
    return pairs


class ImageTestLayout(unittest.TestCase):
    def test_runner_resolves_overlay_and_fixture_in_image_layout(self):
        pairs = staged_copies()
        self.assertIn((RUNNER, "/opt/glm53/test_scheduler_decode_floor.py"), pairs)
        with tempfile.TemporaryDirectory() as tmp:
            for src, dest in pairs:
                if src.startswith(("tests/", "overlay/")) and (ROOT / src).is_file():
                    out = Path(tmp) / dest.lstrip("/")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(ROOT / src, out)
            staged = Path(tmp) / "opt/glm53/test_scheduler_decode_floor.py"
            probe = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('r', sys.argv[1]); "
                     "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                     "print(m.PATCH); print(m.FIXTURES / 'legacy_scheduler_helpers.py')")
            result = subprocess.run([sys.executable, "-c", probe, str(staged)], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-800:])
            patch_path, fixture_path = result.stdout.strip().splitlines()[-2:]
            self.assertEqual(Path(patch_path), Path(tmp) / "opt/glm53/patch_scheduler_decode_floor.py")
            self.assertEqual(Path(fixture_path), Path(tmp) / "opt/glm53/fixtures/legacy_scheduler_helpers.py")
            self.assertTrue(Path(fixture_path).is_file())
            if os.environ.get("GLM53_SCHEDULER_PY_SRC"):
                run = subprocess.run([sys.executable, str(staged)], capture_output=True, text=True, timeout=900)
                self.assertEqual(run.returncode, 0, run.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
