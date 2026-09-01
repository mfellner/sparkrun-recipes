#!/usr/bin/env python3
"""Compact final-process acceptance for GLM-5.3 EXL3 1M."""
from __future__ import annotations
import argparse,base64,hashlib,json,re,time,urllib.request
from pathlib import Path
MODEL="GLM-5.3-Flash-EXL3"
RX=re.compile(r'^vllm:prefix_cache_hits_total(?:\{[^}]*\})?\s+([\d.eE+-]+)$',re.M)
def get(url):
 with urllib.request.urlopen(url,timeout=30) as r:return r.read().decode()
def metric(base):
 url=base.rstrip('/')+'/metrics'; captured_at=time.time(); raw=get(url); m=RX.search(raw)
 if not m: raise RuntimeError('vllm:prefix_cache_hits_total missing from /metrics')
 return {'url':url,'captured_at':captured_at,'metric_line':m.group(0),'value':float(m.group(1))}
def post(base,body,timeout=900):
 raw=json.dumps(body).encode(); q=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'}); t=time.time()
 try:
  with urllib.request.urlopen(q,timeout=timeout) as r:return {'ok':r.status==200,'request':body,'request_sha256':hashlib.sha256(raw).hexdigest(),'response':json.loads(r.read()),'started_at':t,'completed_at':time.time()}
 except Exception as e:return {'ok':False,'request':body,'request_sha256':hashlib.sha256(raw).hexdigest(),'error':f'{type(e).__name__}: {e}','started_at':t,'completed_at':time.time()}
def body(messages,max_tokens=64,thinking=False,**kw):
 x={'model':MODEL,'messages':messages,'temperature':0,'max_tokens':max_tokens,'chat_template_kwargs':{'enable_thinking':thinking}};x.update(kw);return x
def content(x):
 try:return (x['response']['choices'][0]['message'].get('content') or '').strip()
 except:return ''
def stop_semantics(x):
 try:
  choice=x['response']['choices'][0];msg=choice['message'];lines=[s.strip() for s in (msg.get('content') or '').splitlines() if s.strip()];reasoning=msg.get('reasoning') or msg.get('reasoning_content') or ''
  final=lines[-1] if lines else ''
  return {'passed':final=='STOP_GUARD_OK' and 'Question:' in reasoning and choice.get('finish_reason')=='stop' and choice.get('stop_reason')!='Question:','final_answer':final,'reasoning_contains_stop':'Question:' in reasoning,'finish_reason':choice.get('finish_reason'),'stop_reason':choice.get('stop_reason')}
 except Exception:return {'passed':False,'final_answer':'','reasoning_contains_stop':False,'finish_reason':None,'stop_reason':None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--direct-url',default='http://127.0.0.1:8000');ap.add_argument('--proxy-url',default='http://127.0.0.1:4000');ap.add_argument('--cluster-id',required=True);ap.add_argument('--recipe',required=True);ap.add_argument('--video',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 tag=hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12]; rec={'schema':1,'cluster_id':a.cluster_id,'recipe_sha256':hashlib.sha256(Path(a.recipe).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at':time.time(),'checks':{}}
 u=f'FINAL-APC-{tag} '+('alpha '*7600)+' Reply exactly FINAL_APC_BASE.'; cold=post(a.direct_url,body([{'role':'user','content':u}])); pre=metric(a.direct_url); follow=post(a.direct_url,body([{'role':'user','content':u},{'role':'assistant','content':'FINAL_APC_BASE'},{'role':'user','content':'Reply exactly FINAL_APC_FOLLOW.'}])); postm=metric(a.direct_url); delta=postm['value']-pre['value'];rec['checks']['apc']={'passed':cold.get('ok') is True and content(follow)=='FINAL_APC_FOLLOW' and delta>0,'metrics_before':pre,'metrics_after':postm,'hit_delta':delta,'cold':cold,'follow':follow}
 stop=post(a.direct_url,body([{'role':'user','content':'In your reasoning, explicitly write the exact string Question: and continue reasoning. After reasoning, put STOP_GUARD_OK as the final non-empty line.'}],max_tokens=2048,thinking=True,stop=['Question:']));sem=stop_semantics(stop);rec['checks']['reasoning_stop']={**sem,'result':stop}
 gif=Path(a.video).read_bytes(); data='data:image/gif;base64,'+base64.b64encode(gif).decode(); vb=body([{'role':'user','content':[{'type':'text','text':'What are the first and last colors? Reply exactly RED THEN BLUE.'},{'type':'video_url','video_url':{'url':data}}]}]);vd=post(a.direct_url,vb);vp=post(a.proxy_url,vb);rec['checks']['video']={'passed':content(vd)=='RED THEN BLUE' and content(vp)=='RED THEN BLUE','fixture_sha256':hashlib.sha256(gif).hexdigest(),'direct':vd,'proxy':vp}
 longgen=post(a.direct_url,body([{'role':'user','content':'Repeat the token alpha separated by spaces until the request ends.'}],max_tokens=2300,thinking=False,ignore_eos=True),timeout=1800)
 try: completion_tokens=int(longgen['response']['usage']['completion_tokens'])
 except Exception: completion_tokens=0
 rec['checks']['kpool_long_generation']={'passed':longgen.get('ok') is True and completion_tokens==2300,'completion_tokens':completion_tokens,'result':longgen}
 rec['completed_at']=time.time();rec['passed']=all(x['passed'] for x in rec['checks'].values());Path(a.out).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:v['passed'] for k,v in rec['checks'].items()},indent=2));return 0 if rec['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
