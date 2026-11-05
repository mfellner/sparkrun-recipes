"""CPU-only checks for source provenance, ABI shape and profile integrity."""

import hashlib
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import prepare_profile


def elf_fixture(suffix=b""):
    data = bytearray(64)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", data, 18, 183)  # EM_AARCH64
    return bytes(data) + b"\0".join(prepare_profile.REQUIRED_NATIVE_SYMBOLS) + suffix


def manifest_fixture(binary=None):
    if binary is None:
        binary = elf_fixture()
    return {
        "schema": 1,
        "artifact": "glm53-cooperative-moe",
        "target": "sm_121a",
        "upstream": {
            "repository": prepare_profile.UPSTREAM_REPOSITORY,
            "commit": prepare_profile.UPSTREAM_PIN,
            "tree_sha256": prepare_profile.COMBINED_TREE_SHA,
        },
        "inputs": prepare_profile.EXPECTED_INPUTS,
        "build": {
            "source_date_epoch": prepare_profile.SOURCE_DATE_EPOCH,
            "random_seed": "0x474c4d3533434f4f",
            "nvcc": "Cuda compilation tools, release 13.0, V13.0.88",
            "nvcc_sha256": "1" * 64,
            "host_compiler": "g++ (Ubuntu) 14.2.0",
            "system": {
                "sysname": "Linux",
                "release": "6.0",
                "version": "fixture",
                "machine": "aarch64",
            },
            "flags": prepare_profile.EXPECTED_FLAGS,
        },
        "output": {
            "sha256": hashlib.sha256(binary).hexdigest(),
            "size": len(binary),
        },
    }


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        root = Path(self.workspace.name)
        self.stock = root / "stock.py"
        self.artifacts = root / "artifacts"
        self.artifacts.mkdir()
        self.output = root / "selected.py"
        self.stock.write_bytes(b"sentinel = True\n")
        (self.artifacts / "cooperative_moe.so").write_bytes(elf_fixture())
        (self.artifacts / "runtime.py").write_bytes(b"adapter fixture")
        self.write_manifest()
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        pins = patch.multiple(
            prepare_profile,
            STOCK_SHA=digest(self.stock),
            ADAPTER_SHA=digest(self.artifacts / "runtime.py"),
        )
        pins.start()
        self.addCleanup(pins.stop)

    def write_manifest(self):
        binary = (self.artifacts / "cooperative_moe.so").read_bytes()
        (self.artifacts / "build-manifest.json").write_text(
            json.dumps(manifest_fixture(binary)) + "\n"
        )

    def generate(self, runtime="/root/.cache/vllm/cooperative_moe"):
        prepare_profile.make_profile(self.stock, self.artifacts, runtime, self.output)

    def execute_footer(self, runtime):
        self.generate(runtime)
        module = types.ModuleType("profile_fixture")
        install = Mock()
        with (
            patch.dict(sys.modules, {module.__name__: module}),
            patch("runpy.run_path", return_value={"install": install}) as loader,
        ):
            exec(
                compile(self.output.read_bytes(), str(self.output), "exec"),
                module.__dict__,
            )
        self.assertTrue(module.sentinel)
        loader.assert_called_once_with(runtime + "/runtime.py")
        install.assert_called_once_with(module, library_root=runtime, enabled=True)

    def test_verified_profile_selects_explicit_artifacts(self):
        self.execute_footer("/root/.cache/vllm/cooperative_moe")

    def test_quoted_path_is_data_not_python(self):
        self.execute_footer("/runtime/contains'quote")

    def test_existing_output_is_not_overwritten(self):
        self.generate()
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(self.output.read_bytes(), original)

    def test_modified_stock_is_rejected(self):
        self.stock.write_bytes(b"different stock")
        with self.assertRaises(ValueError):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_different_valid_compiler_output_is_accepted(self):
        # The complete ELF hash is deliberately not a repository policy input.
        (self.artifacts / "cooperative_moe.so").write_bytes(elf_fixture(b"other-toolchain"))
        self.write_manifest()
        self.generate()
        self.assertTrue(self.output.exists())

    def test_binary_must_match_its_manifest(self):
        (self.artifacts / "cooperative_moe.so").write_bytes(elf_fixture(b"substituted"))
        with self.assertRaises(ValueError):
            self.generate()

    def test_wrong_architecture_or_symbols_are_rejected(self):
        for binary in (b"not-elf", elf_fixture().replace(b"glm53_coop_launch", b"wrong_symbol____")):
            (self.artifacts / "cooperative_moe.so").write_bytes(binary)
            self.write_manifest()
            with self.subTest(binary=binary[:12]), self.assertRaises(ValueError):
                self.generate()
            self.assertFalse(self.output.exists())

    def test_modified_adapter_is_rejected(self):
        (self.artifacts / "runtime.py").write_bytes(b"different adapter")
        with self.assertRaises(ValueError):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_source_manifest_drift_is_rejected(self):
        for mutate in (
            "repository", "commit", "tree", "input", "target", "epoch",
            "seed", "flags", "compiler",
        ):
            manifest = manifest_fixture()
            if mutate == "repository":
                manifest["upstream"]["repository"] = "https://invalid.example/repo.git"
            elif mutate == "commit":
                manifest["upstream"]["commit"] = "0" * 40
            elif mutate == "tree":
                manifest["upstream"]["tree_sha256"] = "0" * 64
            elif mutate == "input":
                manifest["inputs"] = {**manifest["inputs"], "runtime.py": "0" * 64}
            elif mutate == "target":
                manifest["target"] = "sm_120"
            elif mutate == "epoch":
                manifest["build"]["source_date_epoch"] = 0
            elif mutate == "seed":
                manifest["build"]["random_seed"] = "random"
            elif mutate == "flags":
                manifest["build"]["flags"] = ["-O0"]
            else:
                manifest["build"]["nvcc"] = "unknown"
            (self.artifacts / "build-manifest.json").write_text(json.dumps(manifest))
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.generate()
            self.assertFalse(self.output.exists())

    def test_invalid_runtime_paths_are_rejected(self):
        for path in ("relative", "../relative", "/runtime/../parent"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.generate(path)
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
