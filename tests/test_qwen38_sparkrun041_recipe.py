"""SparkRun 0.4.1 integration contract; no engine or model loads."""
import pathlib
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'recipes/qwen3.8-27b-tensorfold-dgx-spark.yaml'


class RecipeIntegrationTests(unittest.TestCase):
    def test_uses_current_sparkrun_lifecycle_and_managed_cache(self):
        recipe = yaml.safe_load(RECIPE.read_text())
        self.assertNotIn('name', recipe)
        self.assertEqual(recipe['runtime'], 'vllm-distributed')
        for name in ('HF_HOME', 'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR'):
            self.assertNotIn(name, recipe['env'])
        self.assertEqual(recipe['runtime_cache'], {
            'enabled': True, 'key_by_model': True, 'key_by_image': True,
            'prune': {'enabled': False}})
        self.assertNotIn('pre_exec', recipe)
        self.assertIn('serve.py {model}', recipe['command'])
        self.assertIn('$HF_HUB_CACHE/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4', recipe['command'])
        self.assertEqual(recipe['defaults']['max_model_len'], 262144)
        self.assertEqual(recipe['defaults']['max_num_seqs'], 8)
        self.assertEqual(recipe['env']['TENSORFOLD_KV_POOL_GIB'], '78')
        self.assertEqual(recipe['metadata']['source_revision'], 'c11ff138871cf43d12898d79dbfa4dc952d17ca4')


if __name__ == '__main__':
    unittest.main()
