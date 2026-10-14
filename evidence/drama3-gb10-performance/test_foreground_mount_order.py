"""Regression for Docker's unordered Mounts presentation, no GPU/network."""
import ast
import copy
import json
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name('foreground_batch.py')

class MountIdentity(unittest.TestCase):
    def test_source_is_committed_neighbor(self):
        self.assertEqual(SOURCE, Path(__file__).with_name('foreground_batch.py'))

    def test_order_only_is_same_but_real_changes_fail(self):
        tree = ast.parse(SOURCE.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'same_container']
        self.assertEqual(len(nodes), 1, 'foreground operator needs explicit semantic container comparison')
        ns = {'json': json}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
        original = {'id':'abc','image_id':'def','running':True,'mounts':[{'source':'/a','destination':'/x','rw':True},{'source':'/b','destination':'/y','rw':True}]}
        other = copy.deepcopy(original);other['mounts'].reverse()
        self.assertTrue(ns['same_container'](original,other))
        for field,value in [('id','different'),('image_id','different'),('running',False)]:
            changed=copy.deepcopy(other);changed[field]=value
            self.assertFalse(ns['same_container'](original,changed))
        changed=copy.deepcopy(other);changed['mounts'][0]['rw']=False
        self.assertFalse(ns['same_container'](original,changed))
        changed=copy.deepcopy(other);changed['mounts'].append(changed['mounts'][0].copy())
        self.assertFalse(ns['same_container'](original,changed))
        changed=copy.deepcopy(other);changed['mounts'][0]['source']='/wrong'
        self.assertFalse(ns['same_container'](original,changed))

if __name__ == '__main__':unittest.main()
