#!/usr/bin/env python3
"""Regression gate for deterministic default-thinking stop suppression."""
import os
from pathlib import Path
P=Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py')
def main()->int:
 ns={};source=P.read_text();assert '# [glm53-stop-policy]' in source;exec(compile(source,str(P),'exec'),ns);cls=ns['IncrementalDetokenizer']
 class Detok:stop=['Question:'];_reasoning_stop_guard=False;_reasoning_end_str=''
 class Tok:pass
 class Req:prompt_token_ids=[1,2,999];reasoning_ended=True
 d=Detok();cls._maybe_enable_reasoning_stop_guard(d,Tok(),Req());assert d._reasoning_stop_guard is True;assert d._reasoning_end_str=='</think>'
 old=os.environ.get('GLM53_SUPPRESS_STOPS_IN_REASONING');os.environ['GLM53_SUPPRESS_STOPS_IN_REASONING']='0'
 try:
  d2=Detok();d2._reasoning_stop_guard=False;cls._maybe_enable_reasoning_stop_guard(d2,Tok(),Req());assert d2._reasoning_stop_guard is False
 finally:
  if old is None:os.environ.pop('GLM53_SUPPRESS_STOPS_IN_REASONING',None)
  else:os.environ['GLM53_SUPPRESS_STOPS_IN_REASONING']=old
 print('deterministic default-thinking stop guard OK');return 0
if __name__=='__main__':raise SystemExit(main())
