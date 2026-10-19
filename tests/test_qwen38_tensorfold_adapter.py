import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'mods/qwen38-tensorfold/serve.py'
spec = importlib.util.spec_from_file_location('qwen38_adapter', PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AliasAdapterTests(unittest.TestCase):
    def test_translates_only_alias(self):
        args = ['snapshot', '--drafter', 'draft_snapshot', '--served-model-name', 'Qwen3.8-27B',
                '--parallel', '8', '--context', '262144', '--vision', '--thinking']
        original = args.copy()
        expected = args.copy()
        expected[3] = '--name'
        self.assertEqual(module.translate(args), expected)
        self.assertEqual(args, original)

    def test_bad_alias_fails_closed(self):
        for args in [[], ['--served-model-name'], ['--served-model-name', ''],
                     ['--served-model-name', '--vision'], ['--name', 'a'],
                     ['--served-model-name', 'a', '--name', 'b'],
                     ['--served-model-name', 'a', '--served-model-name', 'b']]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                module.translate(args)


if __name__ == '__main__':
    unittest.main()
