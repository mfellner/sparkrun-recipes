#!/usr/bin/env python3
"""Offline fail-closed verifier for the selected immutable Drama3 speed release. MIT."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import yaml
import shlex

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_performance import verify as verify_batch, compare_rows, validate_asr, ASR_MODEL, ASR_REVISION
from performance import digest, identity, harness_hashes
from fixtures import request_for
from acceptance import inspect_audio
from verify_evidence import reject_secrets

FIXTURES=('english-1','english-2','german-1','expression','longer','streaming')
IMAGE_ID='sha256:1bfb3e4f8d310d70afa9d8fe4020c544387f52567d6cb0db399589700bcba4bb'
IMAGE_DIGEST='sha256:eb5f38cb8c242dae2162c0618dddae101bf6ed386feb3fdcf0c38971120e511c'
MODEL_REV='dd64a85cd82e9521f7202610ede97b367fc83946'
ACCEPTANCE_NAMES={'final-acceptance':['english-1','english-2','german-1','german-2','expression','longer','streaming','overlap-a','overlap-b','reference'],'final-post-restart':['post-restart','streaming']}
KERNEL_CAPTURES={'final-control':'7a7c327e6c0e5ebb5d7db40984fc137a34c3e56c23aecc082c860b0b1c0955b4','final-candidate':'ccdbb7771c05190b2a5d76831ad431e8b88a0fbb946ca89496583b7ea60090bb'}
CAPTURE_ROOT='4299fd078cddc1738825aabd0dfee81c6ad15f7d20903a3a7368927aa4a22174'


def require(condition,message):
    if not condition:
        raise ValueError(message)


def validate_profile(runtime,head):
    require(runtime['container']['running'] is True,'runtime stopped')
    join=runtime['process_join'];listener=runtime['listener'];health=runtime['health']
    require(join['namespace_pids'][0]==join['host_pid'] and join['namespace_pids'][-1]==listener['pid']==health['pid'] and join['host_start_ticks']==listener['start_ticks'] and listener['address']=='192.168.178.70:8010','process/listener/health join')
    recipe=yaml.safe_load((ROOT/'recipes/fish-audio-drama3-fast-dgx-spark.yaml').read_text())
    expected_env={name:str(recipe['env'][name]) for name in runtime['effective_env']}
    require(runtime['effective_env']==expected_env and {'FISH_DRAMA3_FP8_EXPERTS','FISH_DRAMA3_CUDA_GRAPH'} <= set(expected_env),'effective runtime configuration')
    profile=runtime['configured_profile']
    flags={name:False for name in ['FISH_DRAMA3_SMALL_GRAPH_BUCKETS','FISH_DRAMA3_PREFIX_SNAPSHOT','DRAMA_NATIVE_STREAMING','DRAMA_PUBLIC_REFERENCE_CACHE','DRAMA_STATEFUL_CODEC']}
    flags['FISH_DRAMA3_COMPACT_HEAD']=head
    require(profile['configured_flags']==flags and all(type(v) is bool for v in profile['configured_flags'].values()),'unsupported configured profile')
    require(profile['kernel_options']=={'state':False,'conv':False,'fp8':False,'tower_up':'off','gdn_qkv':'off'},'unsupported kernels')
    pin=json.loads((ROOT/'mods/drama3-gb10-perf/source-pin.json').read_text())
    require(profile['source_files']==pin['before']|pin['after'],'physical native source substitution')
    require(profile['package_manifest_sha256']==digest((ROOT/'mods/drama3-gb10-release/manifest.sha256').read_bytes()),'package manifest substitution')
    original=['contract.py','server.py','run.sh','probe.py','model-manifest.json','requirements-lock.txt','default-reference.wav','default-reference.txt','reference-provenance.json','manifest.sha256']
    require(runtime['runtime_files']=={n:digest((ROOT/'mods/drama3-gb10'/n).read_bytes()) for n in original},'original adapter source substitution')


def canonical_container(container):
    return {**container,'mounts':sorted(container['mounts'],key=lambda m:json.dumps(m,sort_keys=True))}


def verify(directory):
    directory=Path(directory).resolve()
    # This verifier qualifies one independently reviewed empirical snapshot, not
    # arbitrary re-signed records. No verifier/source hash occurs in this map.
    captured=(directory/'captured-artifacts.json').read_bytes()
    require(digest(captured)==CAPTURE_ROOT,'unapproved empirical snapshot root')
    for name,sha in json.loads(captured)['files'].items():
        path=directory/name
        require(path.resolve().is_relative_to(directory) and path.is_file() and digest(path.read_bytes())==sha,'frozen empirical artifact changed: '+name)
    prep=json.loads((directory/'publication-bindings.json').read_text())
    require(prep.get('schema')==1,'binding schema')
    require(prep['acceptance_names']==ACCEPTANCE_NAMES,'canonical acceptance corpus')
    required_artifacts={'summary.json','public-image.json','http-negative.json','foreground_batch.py'}
    for arm in ['final-control','final-candidate']:
        required_artifacts.update({arm+'/'+n for n in ['complete-serve.log','kernel-full-boot.log','kernel-repeat.log','kernel-repeat-binding.json','log-bindings.json']})
        for fixture in FIXTURES:
            base=arm+'/'+fixture+'/'
            required_artifacts.update({base+n for n in ['runtime-before.json','runtime-after.json','harness.json','recipe.yaml','telemetry.jsonl','operator-receipt.json','repeated/asr.json','repeated/schedule.json']})
            required_artifacts.update({base+'repeated/'+fixture+'-'+s+ext for s in ['warmup','r1','r2','r3','r4','r5'] for ext in ['.json','.wav']})
    for arm,names in ACCEPTANCE_NAMES.items():
        required_artifacts.add(arm+'/asr.json')
        required_artifacts.update({arm+'/'+n+ext for n in names for ext in ['.json','.wav']})
    required_artifacts.update({'final-post-restart/'+n for n in ['runtime.json','runtime-profile.json','launch-state.json','job-state.yaml','cache-witness.json']})
    required_artifacts.update({'final-acceptance/reference-input.wav','final-acceptance/reference.txt'})
    require(required_artifacts <= set(prep['evidence_files']),'required evidence inventory deleted')
    required_sources={'scripts/drama3/'+n+'.py' for n in ['acceptance','fixtures','performance','verify_performance','verify_fast_evidence','capture_runtime','capture_fast_launch','asr','telemetry']}
    required_sources.update({'mods/drama3-gb10-release/'+n for n in ['Dockerfile','serve.py','run.sh','manifest.sha256','NOTICE']})
    required_sources.update({'mods/drama3-gb10-perf/'+n for n in ['native_perf.py','native.patch','apply_patch.py','source-pin.json']})
    required_sources.update({'recipes/fish-audio-drama3-fast-dgx-spark.yaml','mods/drama3-gb10/server.py','mods/drama3-gb10/contract.py','mods/drama3-gb10/default-reference.wav','mods/drama3-gb10/default-reference.txt'})
    require(required_sources <= set(prep['source_files']),'required source bindings deleted')
    for name,sha in prep['source_files'].items():
        require((ROOT/name).resolve().is_relative_to(ROOT),'unsafe source path')
        require(digest((ROOT/name).read_bytes())==sha,'source substitution: '+name)
    for name,sha in prep['evidence_files'].items():
        path=directory/name
        require(path.resolve().is_relative_to(directory) and path.is_file() and digest(path.read_bytes())==sha,'artifact substitution/deletion: '+name)
    measured={};all_rows={}
    for arm,head in [('final-control',False),('final-candidate',True)]:
        all_rows[arm]=[]
        for fixture in FIXTURES:
            batch=directory/arm/fixture
            runtime=json.loads((batch/'runtime-before.json').read_text())
            after=json.loads((batch/'runtime-after.json').read_text())
            harness=json.loads((batch/'harness.json').read_text())
            require(harness==harness_hashes(),'reviewed collector harness source')
            schedule=json.loads((batch/'repeated/schedule.json').read_text())
            require(schedule['fixtures']==[fixture] and schedule['seed']==42 and schedule['repeats']==5 and schedule['warmups']==1 and schedule['arm']==arm,'canonical release schedule')
            operator=json.loads((batch/'operator-receipt.json').read_text())
            require(operator['completed'] is True and all(type(operator[n]) is int and operator[n]==0 for n in ['collector_rc','telemetry_rc','telemetry_copy_rc']),'failed operator/telemetry collection')
            require(operator['wrapper_sha256']==digest((directory/'foreground_batch.py').read_bytes()) and operator['telemetry_sha256']==digest((ROOT/'scripts/drama3/telemetry.py').read_bytes()),'operator/telemetry source identity')
            require(operator['arm']==arm and operator['fixture']==fixture and operator['runtime_image_id']==IMAGE_ID,'operator identity')
            historical=Path('/home/max/drama3-performance-20261014');history_batch=historical/'foreground-completion'/arm/fixture
            recipe_history=historical/'final-control.yaml' if arm=='final-control' else historical/'release-worktree/recipes/fish-audio-drama3-fast-dgx-spark.yaml'
            require(operator['argv']==[str(historical/'foreground_batch.py'),'--arm',arm,'--fixture',fixture,'--container',runtime['container']['id'],'--image',IMAGE_ID,'--recipe',str(recipe_history)],'operator argv')
            require(operator['collector_argv']==['/home/max/.hermes/cache/scratch/drama3-all-tests311/bin/python','/home/max/sparkrun-recipes/scripts/drama3/performance.py','--endpoint','http://192.168.178.70:8010','--output',str(history_batch/'repeated'),'--arm',arm,'--runtime-json',str(history_batch/'runtime-before.json'),'--fixtures',fixture,'--repeats','5','--seed','42'],'collector argv')
            remote=f'/home/max/.cache/huggingface/drama3/performance/foreground-{arm}-{fixture}'
            require(operator['telemetry_argv']==['ssh','192.168.178.70',shlex.join(['python3','-','--output',remote+'/samples.jsonl','--stop-file',remote+'/stop','--seconds','570'])],'telemetry argv')
            recipe=yaml.safe_load((ROOT/'recipes/fish-audio-drama3-fast-dgx-spark.yaml').read_text())
            if arm=='final-control':recipe['name']='fish-audio-drama3-fast-control-experiment';recipe['env']['FISH_DRAMA3_COMPACT_HEAD']='0'
            require(yaml.safe_load((batch/'recipe.yaml').read_text())==recipe,'batch recipe identity')
            require(runtime['container']['image_id']==IMAGE_ID and runtime['container']['running'] is True,'runtime image/state')
            require(runtime['hostname']=='dgx02' and runtime['host']=='192.168.178.70','host substitution')
            require(runtime['configured_profile']['configured_flags']['FISH_DRAMA3_COMPACT_HEAD'] is head,'head configuration')
            require(runtime['container']['id']==prep['containers'][arm],'container substitution')
            require(runtime['process_join']==after['process_join'] and runtime['runtime_files']==after['runtime_files'],'process/source changed')
            require(runtime['configured_profile']==after['configured_profile'],'profile changed')
            require(runtime['effective_env']==after['effective_env'],'effective environment changed')
            require(runtime['container']['id']==after['container']['id'],'container changed')
            require(canonical_container(runtime['container'])==canonical_container(after['container']),'container state/mounts changed')
            validate_profile(runtime,head);validate_profile(after,head)
            result=verify_batch(batch/'repeated',runtime,harness)
            require(result['measured_count']==5 and result['warmup_count']==1,'batch count')
            telemetry=[json.loads(line) for line in (batch/'telemetry.jsonl').read_text().splitlines()]
            require(bool(telemetry),'telemetry absent')
            telemetry_keys={'timestamp','gpu_util_pct','sm_clock_mhz','power_w','temperature_c','pstate','MemAvailable','SwapFree'}
            require(all(set(s)==telemetry_keys for s in telemetry),'incomplete telemetry measurements')
            bounds={'timestamp':(0,1e11),'gpu_util_pct':(0,100),'sm_clock_mhz':(0,10000),'power_w':(0,1000),'temperature_c':(-30,150),'MemAvailable':(0,2**60),'SwapFree':(0,2**60)}
            require(all(type(s[k]) in (int,float) and math.isfinite(s[k]) and lo<=s[k]<=hi for s in telemetry for k,(lo,hi) in bounds.items()),'invalid telemetry values')
            require(all(s['pstate'] in {f'P{i}' for i in range(16)} for s in telemetry),'invalid telemetry pstate')
            require(all(a['timestamp']<b['timestamp'] for a,b in zip(telemetry,telemetry[1:])),'unordered telemetry')
            require(operator['started_at']<=telemetry[0]['timestamp']<=telemetry[-1]['timestamp']<=operator['finished_at'],'telemetry operator window')
            for suffix in ['warmup','r1','r2','r3','r4','r5']:
                row=json.loads((batch/'repeated'/f'{fixture}-{suffix}.json').read_text())
                require(operator['started_at']<=schedule['configured_at']<=row['started_at'] and row['finished_at']<=operator['finished_at'] and after['captured_at']<=operator['finished_at'],'schedule/operator interval')
                require(row['runtime']['container']['created_at']<row['started_at'],'request predates launch')
                require(runtime['captured_at']<=row['started_at'] and after['captured_at']>=row['finished_at'],'capture interval')
                require(any(row['started_at']<=s['timestamp']<=row['finished_at'] for s in telemetry),'telemetry outside request')
                all_rows[arm].append(row)
        measured[arm]=[r for r in all_rows[arm] if r['warmup'] is False]
        log=directory/arm/'complete-serve.log'
        bindings=json.loads((directory/arm/'log-bindings.json').read_text())
        require(digest(log.read_bytes())==bindings['serve_sha256'],'serve log binding')
        kernel=(directory/arm/'kernel-full-boot.log').read_text()
        require(digest(kernel.encode())==KERNEL_CAPTURES[arm],'reviewed complete kernel capture')
        require(digest(kernel.encode())==bindings['kernel_sha256'] and bindings['kernel_repeat_equal'] is True and bindings['kernel_rc']==0,'kernel binding')
        expected_command=['ssh','192.168.178.70','journalctl','-k','--since','@'+str(bindings['boot_epoch']),'--until','@'+str(bindings['until_epoch']),'--no-pager','-o','short-unix','--show-cursor']
        require(bindings['kernel_argv']==expected_command and bindings['until_epoch']>=max(r['finished_at'] for r in all_rows[arm]),'kernel command/window')
        repeat=(directory/arm/'kernel-repeat.log').read_bytes();recheck=json.loads((directory/arm/'kernel-repeat-binding.json').read_text())
        require(recheck['command']==expected_command and recheck['returncode']==0 and recheck['sha256']==digest(repeat),'independent kernel repeat binding')
        require(repeat.rsplit(b'-- cursor:',1)[0]==kernel.encode().rsplit(b'-- cursor:',1)[0] and b'-- cursor:' in repeat,'kernel event completeness/repeat')
        require('-- cursor:' in kernel and re.search(r'^\d+\.\d+ dgx02 kernel:',kernel,re.M) is not None,'unrecognized kernel capture')
        require(re.search(r'NVRM.*Xid|NV_ERR_NO_MEMORY|Out of memory:|Killed process.*total-vm|oom-kill',kernel,re.I) is None,'fatal kernel event')
        receipts={}
        for line in log.read_text().splitlines():
            try:obj=json.loads(line)
            except ValueError:continue
            if isinstance(obj,dict) and 'request_id' in obj:receipts[obj['request_id']]=obj
        for row in all_rows[arm]:
            require(row['request_id'] in receipts,'server receipt absent')
            server=receipts[row['request_id']]
            require(server['model_revision']==MODEL_REV and server['text_sha256']==digest(row['request']['text'].encode()),'server request mismatch')
            require(server['queued_at']>=row['started_at']-1 and server['finished_at']<=row['finished_at']+1,'server timing outside HTTP')
            require(math.isclose(server['duration_s'],row['duration_s'],abs_tol=1e-9),'server sample duration')
    comparisons=compare_rows(measured['final-control'],measured['final-candidate'])
    before={r['name']:r for r in all_rows['final-control']};after={r['name']:r for r in all_rows['final-candidate']}
    require(set(before)==set(after) and len(before)==36,'matched corpus count')
    require(all(before[n]['audio_sha256']==after[n]['audio_sha256'] and before[n]['request']==after[n]['request'] for n in before),'nonidentical audio or requests')
    summary=json.loads((directory/'summary.json').read_text())
    for name in ['baseline_median_rtf','candidate_median_rtf','rtf_reduction']:
        key='median_rtf_reduction' if name=='rtf_reduction' else name
        require(summary[name]==comparisons[key],'summary accounting: '+name)
    require(summary['measured_per_arm']==30 and summary['warmups_per_arm']==6 and summary['matched36_audio_bytes_equal'] is True,'summary totals')
    require(summary['extra_acceptance_cases']==10 and summary['post_restart_cases']==2,'acceptance totals')
    require(summary['image_id']==IMAGE_ID and summary['image_digest']==IMAGE_DIGEST and summary['asr_per_arm']==36,'summary identity/ASR counts')
    for arm,key in [('final-control','baseline_streaming_first_pcm_median_s'),('final-candidate','candidate_streaming_first_pcm_median_s')]:
        require(summary[key]==statistics.median(r['first_pcm_s'] for r in measured[arm] if r['fixture']=='streaming'),'summary first PCM')
    require(summary['goals']=={'median_rtf_le_0_9':summary['candidate_median_rtf']<=0.9,'first_pcm_lt_2s':summary['candidate_streaming_first_pcm_median_s']<2},'goal accounting')
    for arm in ['final-acceptance','final-post-restart']:
        folder=directory/arm;asr=json.loads((folder/'asr.json').read_text());require(asr['passed'] is True,'acceptance ASR failure')
        require(asr['model']==ASR_MODEL and asr['revision']==ASR_REVISION and asr['device']=='cpu' and asr['compute_type']=='int8' and asr['beam_size']==5 and asr['vad_filter'] is False,'acceptance ASR provenance')
        names=prep['acceptance_names'][arm];require(len(asr['rows'])==len(names),'acceptance count');by_name={r['name']:r for r in asr['rows']}
        require(set(by_name)==set(names),'unique canonical acceptance ASR names')
        for name in names:
            row=json.loads((folder/(name+'.json')).read_text())
            require(row['audio_file']==name+'.wav','unsafe acceptance audio path')
            raw=(folder/row['audio_file']).read_bytes()
            expected=request_for(name)
            if name=='reference':
                require((folder/'reference-input.wav').read_bytes()==(ROOT/'mods/drama3-gb10/default-reference.wav').read_bytes(),'public reference substitution')
                require((folder/'reference.txt').read_text().strip()==(ROOT/'mods/drama3-gb10/default-reference.txt').read_text().strip(),'public transcript substitution')
                expected['references']=[{'audio':'sha256:'+digest((folder/'reference-input.wav').read_bytes()),'text':(folder/'reference.txt').read_text().strip(),'consent':True}]
            require(row['status']==200 and row['model_revision']==MODEL_REV and row['request']==expected,'acceptance request/status')
            require(row['endpoint']=='http://192.168.178.70:8010' and row['route']=='/v1/tts' and row['model']=='fishaudio/drama-3','acceptance model/route')
            require(re.fullmatch('[0-9a-f]{32}',row['request_id']) is not None and row['headers']['x-request-id']==row['request_id'] and row['headers']['x-model-revision']==MODEL_REV and row['headers']['content-type'].split(';')[0]=='audio/wav','acceptance HTTP headers')
            require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ['started_at','finished_at','wall_s']) and row['wall_s']>0 and abs(row['finished_at']-row['started_at']-row['wall_s'])<1,'acceptance timing')
            if arm=='final-acceptance':
                require(row['request_id'] in receipts,'extra server receipt missing')
                server=receipts[row['request_id']]
                require(server['model_revision']==MODEL_REV and server['text_sha256']==digest(row['request']['text'].encode()) and row['started_at']-1<=server['queued_at']<=server['finished_at']<=row['finished_at']+1 and row['finished_at']<=bindings['until_epoch'],'extra request/server/kernel window')
            require(digest(raw)==row['audio_sha256'] and inspect_audio(raw)==row['audio'],'acceptance audio')
            validate_asr(row,by_name[name])
    public=json.loads((directory/'public-image.json').read_text())
    require(public['anonymous'] is True and public['status']==200 and public['manifest_digest']==IMAGE_DIGEST and public['config_digest']==IMAGE_ID and public['architecture']=='arm64','public image identity')
    negative=json.loads((directory/'http-negative.json').read_text())
    expected_status={n:400 for n in ['empty text','oversized text','URL input','unbounded frames','structural token','invalid speaker','malformed JSON']}
    expected_status.update({'oversized body':413,'cross-origin':403,'simple Content-Type':403})
    require(negative['endpoint']=='http://192.168.178.70:8010' and negative['passed'] is True and len(negative['rows'])==10,'HTTP negative count/route')
    require({r['case']:r['status'] for r in negative['rows']}==expected_status and all(r['expected_status']==expected_status[r['case']] and isinstance(json.loads(r['response']).get('detail'),str) for r in negative['rows']),'canonical HTTP security controls')
    require(len(negative['unintended_interfaces'])==3 and {r['host'] for r in negative['unintended_interfaces']}=={'192.168.3.183','192.168.2.183','target loopback 127.0.0.1'} and all(r['reachable'] is False and r['port']==8010 for r in negative['unintended_interfaces']),'unintended exposure')
    post=json.loads((directory/'final-post-restart/runtime.json').read_text());launch=json.loads((directory/'final-post-restart/launch-state.json').read_text());witness=json.loads((directory/'final-post-restart/cache-witness.json').read_text())
    require(post['container']['id']==launch['container_id'] and post['container']['id']!=prep['containers']['final-candidate'] and post['container']['image_id']==IMAGE_ID,'restart identity')
    require(launch['recipe_sha256']==digest((ROOT/'recipes/fish-audio-drama3-fast-dgx-spark.yaml').read_bytes()),'recipe binding')
    profile_record=json.loads((directory/'final-post-restart/runtime-profile.json').read_text())
    require(profile_record['container_id_before']==profile_record['container_id_after']==post['container']['id'] and profile_record['host']=='192.168.178.70' and profile_record['endpoint']=='http://192.168.178.70:8010/v1/runtime-profile','postrestart profile route/container')
    validate_profile({**post,'configured_profile':profile_record['configured_profile']},True)
    job_bytes=(directory/'final-post-restart/job-state.yaml').read_bytes();job=yaml.safe_load(job_bytes)
    require(digest(job_bytes)==launch['persisted_sha256'] and job==launch['status_meta'],'persisted job integrity')
    require(job['recipe_state']['_raw']==yaml.safe_load((ROOT/'recipes/fish-audio-drama3-fast-dgx-spark.yaml').read_text()),'persisted recipe substitution')
    first=min(json.loads((directory/'final-post-restart'/(n+'.json')).read_text())['started_at'] for n in ACCEPTANCE_NAMES['final-post-restart'])
    last=max(json.loads((directory/'final-post-restart'/(n+'.json')).read_text())['finished_at'] for n in ACCEPTANCE_NAMES['final-post-restart'])
    require(post['container']['created_at']<=post['health']['started_at']<=first<=last<=post['captured_at']<=profile_record['captured_at'],'postrestart serving/request/readback order')
    require(job['cluster_id']==launch['workload_id'] and job['hosts']==['192.168.178.70'] and launch['persisted_before_acceptance'] is True and job['started_at']<=launch['persistence_mtime']<=first,'launch identity/persistence ordering')
    require(witness['output_exists'] is True and witness['output_sha256']==json.loads((directory/'final-acceptance/english-1.json').read_text())['audio_sha256'],'output persistence')
    require(bool(witness['compiler_files']),'compiler artifacts absent')
    # Compiler witnesses are only postrestart presence, not a pre/post hash comparison.
    reject_secrets(prep)
    return {'qualified':True,**summary,'comparisons':comparisons}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.directory),indent=2))
