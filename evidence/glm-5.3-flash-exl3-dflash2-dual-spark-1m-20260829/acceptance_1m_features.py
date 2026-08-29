#!/usr/bin/env python3
"""GLM-5.3 EXL3 1M-specific APC, occupancy, stop-guard, video and near-limit tests."""
from __future__ import annotations
import argparse, base64, concurrent.futures, hashlib, io, json, re, time, urllib.request
from pathlib import Path
from typing import Any
from PIL import Image

MODEL="GLM-5.3-Flash-EXL3"
PREFIX_RE=re.compile(r'^vllm:prefix_cache_hits_total(?:\{[^}]*\})?\s+([\d.eE+-]+)$',re.M)

def get_text(base:str,path:str)->str:
 with urllib.request.urlopen(base.rstrip('/')+path,timeout=30) as r:return r.read().decode()
def metrics(base:str)->dict:
 t=get_text(base,'/metrics'); m=PREFIX_RE.search(t); return {'prefix_hits':float(m.group(1)) if m else 0.0,'raw_sha256':hashlib.sha256(t.encode()).hexdigest()}
def post(base:str,body:dict[str,Any],timeout:float=2400)->dict:
 raw=json.dumps(body).encode(); req=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'})
 st=time.time()
 try:
  with urllib.request.urlopen(req,timeout=timeout) as r:
   b=r.read().decode(); return {'ok':r.status==200,'http':r.status,'started_at':st,'completed_at':time.time(),'request':body,'request_sha256':hashlib.sha256(raw).hexdigest(),'response':json.loads(b)}
 except Exception as e:return {'ok':False,'started_at':st,'completed_at':time.time(),'request':body,'request_sha256':hashlib.sha256(raw).hexdigest(),'error':f'{type(e).__name__}: {e}'}
def content(x:dict)->str:
 try:return (x['response']['choices'][0]['message'].get('content') or '').strip()
 except:return ''
def usage(x:dict)->dict:
 try:return x['response'].get('usage') or {}
 except:return {}
def body(messages,max_tokens=32,thinking=False,**extra):
 d={'model':MODEL,'messages':messages,'temperature':0,'max_tokens':max_tokens,'chat_template_kwargs':{'enable_thinking':thinking}}; d.update(extra); return d

def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--direct-url',default='http://127.0.0.1:8000'); ap.add_argument('--proxy-url',default='http://192.168.1.95:4000'); ap.add_argument('--recipe',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
 rec={'schema':1,'run_id':hashlib.sha256(f'{time.time_ns()}:{out}'.encode()).hexdigest()[:16],'started_at':time.time(),'cluster_id':'sparkrun_88b110d8c27f0455_73f56777180a','recipe_sha256':hashlib.sha256(Path(a.recipe).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':{}}
 def save():out.write_text(json.dumps(rec,indent=2)+'\n')
 save()

 # Single-session APC cold/follow-up.
 tag=f'APC-{rec["run_id"]}'; first_user=f'{tag} '+('alpha '*7600)+' Reply with exactly APC_BASE_READY.'
 before=metrics(a.direct_url); cold=post(a.direct_url,body([{'role':'user','content':first_user}]),900); mid=metrics(a.direct_url)
 follow=post(a.direct_url,body([{'role':'user','content':first_user},{'role':'assistant','content':'APC_BASE_READY'},{'role':'user','content':'Reply with exactly APC_FOLLOW_OK.'}]),900); after=metrics(a.direct_url)
 rec['checks']['apc_single']={'passed':content(cold)=='APC_BASE_READY' and content(follow)=='APC_FOLLOW_OK' and after['prefix_hits']-mid['prefix_hits']>0,'metrics_before':before,'metrics_after_cold':mid,'metrics_after_follow':after,'hit_delta_follow':after['prefix_hits']-mid['prefix_hits'],'cold':cold,'follow':follow}; save()

 # Four isolated sessions and concurrent follow-ups.
 sessions=[]
 for i in range(4):
  u=f'APC-C4-{rec["run_id"]}-{i} '+('beta '*7400)+f' Reply with exactly APC_COLD_{i}.'; r=post(a.direct_url,body([{'role':'user','content':u}]),900); sessions.append((u,r))
 pre=metrics(a.direct_url)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  fut=[ex.submit(post,a.direct_url,body([{'role':'user','content':u},{'role':'assistant','content':f'APC_COLD_{i}'},{'role':'user','content':f'Reply with exactly APC_FOLLOW_{i}.'}]),900) for i,(u,_) in enumerate(sessions)]
  follows=[f.result() for f in fut]
 postm=metrics(a.direct_url)
 rec['checks']['apc_c4']={'passed':all(content(r)==f'APC_COLD_{i}' for i,(_,r) in enumerate(sessions)) and all(content(r)==f'APC_FOLLOW_{i}' for i,r in enumerate(follows)) and postm['prefix_hits']-pre['prefix_hits']>0,'hit_delta_follow':postm['prefix_hits']-pre['prefix_hits'],'cold':[r for _,r in sessions],'follow':follows}; save()

 # Reasoning stop suppression.
 stop_body=body([{'role':'user','content':'In your reasoning, explicitly write the exact string Question: and continue reasoning. Then answer exactly STOP_GUARD_OK.'}],max_tokens=512,thinking=True,stop=['Question:'])
 stop=post(a.direct_url,stop_body,900); rec['checks']['reasoning_stop_guard']={'passed':'STOP_GUARD_OK' in content(stop),'result':stop}; save()

 # Deterministic animated GIF through video_url, direct and proxy.
 frames=[Image.new('RGB',(224,224),'red') for _ in range(4)]+[Image.new('RGB',(224,224),'blue') for _ in range(4)]; buf=io.BytesIO(); frames[0].save(buf,format='GIF',save_all=True,append_images=frames[1:],duration=250,loop=0); gif=buf.getvalue(); fixture=out.with_name('video-red-blue.gif'); fixture.write_bytes(gif); data='data:image/gif;base64,'+base64.b64encode(gif).decode()
 vb=body([{'role':'user','content':[{'type':'text','text':'What are the first and last colors in this video? Reply exactly RED THEN BLUE.'},{'type':'video_url','video_url':{'url':data}}]}],max_tokens=64)
 vd=post(a.direct_url,vb,900); vp=post(a.proxy_url,vb,900); rec['video_fixture']={'path':fixture.name,'sha256':hashlib.sha256(gif).hexdigest(),'bytes':len(gif)}; rec['checks']['video']={'passed':content(vd)=='RED THEN BLUE' and content(vp)=='RED THEN BLUE','direct':vd,'proxy':vp}; save()

 # Three concurrent ~256K contexts; upstream regression case.
 loads=[]
 for i in range(3):
  prompt=f'OCC-{rec["run_id"]}-{i} '+('gamma '*256000)+f' Reply with exactly OCC_OK_{i}.'; loads.append(body([{'role':'user','content':prompt}],max_tokens=16))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  fs=[ex.submit(post,a.direct_url,b,2400) for b in loads]; load_results=[f.result() for f in fs]
 rec['checks']['occupancy_3x256k']={'passed':all(content(r)==f'OCC_OK_{i}' for i,r in enumerate(load_results)),'results':load_results}; save()

 # Near-limit single semantic retrieval, last because it is the heaviest stress.
 needle='NEEDLE_1M_529841'; prompt=('delta '*475000)+f' Hidden retrieval code: {needle}. '+('delta '*475000)+f' Reply with exactly {needle}.'
 long=post(a.direct_url,body([{'role':'user','content':prompt}],max_tokens=32),3600); rec['checks']['near_1m']={'passed':content(long)==needle,'needle':needle,'prompt_chars':len(prompt),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'result':long}; save()
 rec['completed_at']=time.time(); rec['passed']=all(x.get('passed') is True for x in rec['checks'].values()); save(); print(json.dumps({k:v.get('passed') for k,v in rec['checks'].items()},indent=2)); return 0 if rec['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
