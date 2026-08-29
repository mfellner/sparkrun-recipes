#!/usr/bin/env python3
"""Make MiaAI's reasoning-stop guard deterministic for this default-thinking profile."""
from __future__ import annotations
import ast
from pathlib import Path
P=Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py')
MARK='# [glm53-default-thinking-stop-guard]'
CHECK_MARK='# [glm53-stop-policy]'
OLD='''            think_id = None
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                think_id = convert(start_str)
            if think_id is None or think_id < 0:
                encode = getattr(tokenizer, "encode", None)
                if callable(encode):
                    try:
                        ids = encode(start_str, add_special_tokens=False)
                    except TypeError:
                        ids = encode(start_str)
                    if isinstance(ids, list) and len(ids) == 1:
                        think_id = ids[0]
            if think_id is not None and think_id >= 0 and ptids[-1] == think_id:
                detok._reasoning_stop_guard = True
                detok._reasoning_end_str = end_str
'''
NEW='''            # [glm53-default-thinking-stop-guard] This published profile is
            # default-thinking. Request-level reasoning state is not preserved
            # reliably by the pinned engine, so any supplied client stop remains
            # dormant until </think>. EOS and max_tokens remain unchanged. Set
            # GLM53_SUPPRESS_STOPS_IN_REASONING=0 to restore stock stop behavior.
            detok._reasoning_stop_guard = True
            detok._reasoning_end_str = end_str
'''
CHECK_OLD='''            and (not self._reasoning_stop_guard or self._reasoning_closed)
'''
CHECK_NEW='''            # [glm53-stop-policy] This default-thinking profile ignores
            # client stop strings while the recipe guard is enabled. EOS and
            # max_tokens remain active; env opt-out restores stock checks.
            and not IncrementalDetokenizer._suppress_stops_enabled()
'''
def main()->int:
 if not P.is_file():raise SystemExit(f'missing {P}')
 s=P.read_text()
 if MARK not in s:
  if s.count(OLD)!=1:raise SystemExit(f'{P}: expected one upstream guard anchor, found {s.count(OLD)}')
  s=s.replace(OLD,NEW,1)
 if CHECK_MARK not in s:
  if s.count(CHECK_OLD)!=1:raise SystemExit(f'{P}: expected one stop-check anchor, found {s.count(CHECK_OLD)}')
  s=s.replace(CHECK_OLD,CHECK_NEW,1)
 ast.parse(s,filename=str(P));P.write_text(s);print('patched deterministic default-thinking stop policy');return 0
if __name__=='__main__':raise SystemExit(main())
