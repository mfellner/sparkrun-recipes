#!/usr/bin/env python3
"""Live XGrammar reasoning/termination acceptance for the GLM-5.3 recipe."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
MODEL='GLM-5.3-Flash-EXL3'
def post(base,body,timeout=900):
 raw=json.dumps(body).encode();q=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'});t=time.time()
 try:
  with urllib.request.urlopen(q,timeout=timeout) as r:return {'ok':r.status==200,'http':r.status,'started_at':t,'completed_at':time.time(),'request':body,'request_sha256':hashlib.sha256(raw).hexdigest(),'response':json.loads(r.read())}
 except Exception as e:return {'ok':False,'started_at':t,'completed_at':time.time(),'request':body,'request_sha256':hashlib.sha256(raw).hexdigest(),'error':f'{type(e).__name__}: {e}'}
def one(base,i,thinking):
 schema={'type':'object','properties':{'answer':{'type':'integer','const':42},'case':{'type':'integer','const':i}},'required':['answer','case'],'additionalProperties':False}
 body={'model':MODEL,'messages':[{'role':'user','content':f'Case {i}: think briefly, then return the required JSON object with answer 42 and case {i}.'}],'temperature':0,'max_tokens':1024,'chat_template_kwargs':{'enable_thinking':thinking},'response_format':{'type':'json_schema','json_schema':{'name':f'case_{i}','strict':True,'schema':schema}}}
 r=post(base,body)
 try:
  choice=r['response']['choices'][0];obj=json.loads(choice['message'].get('content') or '');passed=obj=={'answer':42,'case':i} and choice.get('finish_reason')=='stop'
 except Exception:passed=False
 r['passed']=passed;return r
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base-url',default='http://127.0.0.1:8000');ap.add_argument('--cluster-id',required=True);ap.add_argument('--recipe',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out)
 rec={'schema':1,'cluster_id':a.cluster_id,'process_role':'exact_final','recipe_sha256':hashlib.sha256(Path(a.recipe).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at':time.time()}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: rows=list(ex.map(lambda i:one(a.base_url,i,True),range(4)))
 control=one(a.base_url,9,False);rec['reasoning_c4']=rows;rec['nonthinking_control']=control;rec['completed_at']=time.time();rec['passed']=all(x['passed'] for x in rows) and control['passed'];out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({'reasoning_c4':[x['passed'] for x in rows],'control':control['passed'],'passed':rec['passed']},indent=2));return 0 if rec['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
