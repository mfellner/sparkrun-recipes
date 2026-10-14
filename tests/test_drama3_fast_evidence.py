"""Real release evidence must pass, and semantic mutations must fail."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def loader():
    file=ROOT/'scripts/drama3/verify_fast_evidence.py'
    assert file.is_file(), 'fast release evidence verifier is missing'
    spec=importlib.util.spec_from_file_location('fast_evidence',file)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_real_fast_evidence_is_complete():
    module=loader()
    result=module.verify(ROOT/'evidence/drama3-gb10-performance')
    assert result['measured_per_arm']==30
    assert result['matched36_audio_bytes_equal'] is True
    assert 0 < result['rtf_reduction'] < 1
    assert (ROOT/'evidence/drama3-gb10-performance/captured-artifacts.json').is_file(), 'exact empirical snapshot anchor is missing'

@pytest.mark.parametrize('mutation', ['http','sampling','summary','missing_asr','kernel','server_receipt','time','missing_telemetry','acceptance_corpus','pcm_summary','public_reference','missing_source_bindings','missing_job_state','runtime_source','unsupported_profile','weakened_http','runtime_stopped_after','late_schedule','fake_harness','failed_operator','wrong_recipe','empty_telemetry_schema','post_asr_provenance','truncated_kernel','foreign_extra','foreign_post','post_stopped','post_join','changed_effective','p999','rewritten_empirical_root'])
def test_refreshed_manifest_semantic_mutations_fail(tmp_path,mutation):
    import json,hashlib,shutil
    module=loader();source=ROOT/'evidence/drama3-gb10-performance';target=tmp_path/'evidence';shutil.copytree(source,target)
    rowpath=target/'final-candidate/english-1/repeated/english-1-r1.json'
    def edit(path,fn):
        value=json.loads(path.read_text());fn(value);path.write_text(json.dumps(value,indent=2)+'\n')
    if mutation=='http':edit(rowpath,lambda d:d.update(status=500))
    elif mutation=='sampling':
        def changed(d):
            d['request']['temperature']=0.6;d['request_identity']=module.identity(d['request'])
        edit(rowpath,changed)
    elif mutation=='summary':edit(target/'summary.json',lambda d:d.update(candidate_median_rtf=0.01))
    elif mutation=='pcm_summary':edit(target/'summary.json',lambda d:d.update(candidate_streaming_first_pcm_median_s=0.01))
    elif mutation=='public_reference':
        folder=target/'final-acceptance';(folder/'reference-input.wav').write_bytes((folder/'reference.wav').read_bytes())
        edit(folder/'reference.json',lambda d:d['request']['references'][0].update(audio='sha256:'+hashlib.sha256((folder/'reference-input.wav').read_bytes()).hexdigest()))
    elif mutation=='time':
        def shifted(d):d['started_at']-=100000;d['finished_at']-=100000
        edit(rowpath,shifted)
    elif mutation=='missing_asr':(target/'final-acceptance/asr.json').unlink()
    elif mutation=='missing_telemetry':(target/'final-candidate/english-1/telemetry.jsonl').unlink()
    elif mutation=='missing_job_state':(target/'final-post-restart/job-state.yaml').unlink()
    elif mutation=='runtime_stopped_after':edit(target/'final-candidate/english-1/runtime-after.json',lambda d:d['container'].update(running=False))
    elif mutation=='late_schedule':edit(target/'final-candidate/english-1/repeated/schedule.json',lambda d:d.update(configured_at=9999999999))
    elif mutation=='wrong_recipe':(target/'final-candidate/english-1/recipe.yaml').write_text('not_the_recipe: true\n')
    elif mutation=='failed_operator':edit(target/'final-candidate/english-1/operator-receipt.json',lambda d:d.update(completed=False,collector_rc=1,telemetry_rc=1,telemetry_copy_rc=1,wrapper_sha256='0'*64))
    elif mutation=='empty_telemetry_schema':
        p=target/'final-candidate/english-1/telemetry.jsonl';p.write_text('\n'.join(json.dumps({'timestamp':json.loads(l)['timestamp']}) for l in p.read_text().splitlines())+'\n')
    elif mutation=='post_asr_provenance':edit(target/'final-post-restart/asr.json',lambda d:d.update(model='invented',revision='0'*40,device='cuda',compute_type='float32',beam_size=1,vad_filter=True))
    elif mutation in ['foreign_extra','foreign_post']:
        p=target/('final-acceptance/english-1.json' if mutation=='foreign_extra' else 'final-post-restart/streaming.json')
        edit(p,lambda d:d.update(endpoint='http://unrelated.invalid',route='/not-tts',request_id='invented',headers={}))
    elif mutation=='post_stopped':edit(target/'final-post-restart/runtime.json',lambda d:d['container'].update(running=False))
    elif mutation=='post_join':edit(target/'final-post-restart/runtime.json',lambda d:d.update(process_join={'host_pid':999999999,'namespace_pids':[999999999,999],'host_start_ticks':1}))
    elif mutation=='rewritten_empirical_root':
        p=target/'final-post-restart/runtime.json';edit(p,lambda d:d['health'].update(status='error'))
        edit(target/'captured-artifacts.json',lambda d:d['files'].update({'final-post-restart/runtime.json':hashlib.sha256(p.read_bytes()).hexdigest()}))
    elif mutation=='changed_effective':edit(target/'final-candidate/english-1/runtime-after.json',lambda d:d['effective_env'].update(FISH_DRAMA3_FP8_EXPERTS='0',FISH_DRAMA3_CUDA_GRAPH='0'))
    elif mutation=='p999':
        p=target/'final-candidate/english-1/telemetry.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()];p.write_text('\n'.join(json.dumps({**d,'pstate':'P999'}) for d in rows)+'\n')
    elif mutation=='truncated_kernel':
        p=target/'final-candidate/kernel-full-boot.log';lines=p.read_text().splitlines();p.write_text(lines[0]+'\n'+lines[-1]+'\n')
        edit(target/'final-candidate/log-bindings.json',lambda d:d.update(kernel_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    elif mutation=='fake_harness':
        batch=target/'final-candidate/english-1';bad={'performance.py':'0'*64};(batch/'harness.json').write_text(json.dumps(bad))
        edit(batch/'repeated/schedule.json',lambda d:d.update(harness=bad))
        for p in (batch/'repeated').glob('english-1-*.json'):edit(p,lambda d:d.update(harness=bad))
    elif mutation=='weakened_http':
        def weaken(d):
            d['rows'][0].update(status=200,expected_status=200,response='success');d['unintended_interfaces']=[]
        edit(target/'http-negative.json',weaken)
    elif mutation in ['runtime_source','unsupported_profile']:
        batch=target/'final-candidate/english-1';runtime=json.loads((batch/'runtime-before.json').read_text())
        if mutation=='runtime_source':runtime['configured_profile']['source_files']['fish_speech/models/text2semantic/fish_omni/generate.py']='0'*64
        else:runtime['configured_profile']['configured_flags']['FISH_DRAMA3_PREFIX_SNAPSHOT']=True
        edit(batch/'runtime-before.json',lambda d:d.update(runtime))
        edit(batch/'runtime-after.json',lambda d:d.update(configured_profile=runtime['configured_profile']))
        edit(batch/'repeated/schedule.json',lambda d:d.update(runtime=runtime))
        for p in (batch/'repeated').glob('english-1-*.json'):edit(p,lambda d:d.update(runtime=runtime,runtime_identity=module.identity(runtime)))
    elif mutation=='kernel':
        p=target/'final-candidate/kernel-full-boot.log';p.write_text(p.read_text()+'\n1792010000.000000 dgx02 kernel: NVRM: Xid (PCI:000f:01:00): 31\n')
        edit(target/'final-candidate/log-bindings.json',lambda d:d.update(kernel_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    elif mutation=='server_receipt':
        p=target/'final-candidate/complete-serve.log';rid=json.loads(rowpath.read_text())['request_id'];p.write_text('\n'.join(l for l in p.read_text().splitlines() if rid not in l)+'\n')
        edit(target/'final-candidate/log-bindings.json',lambda d:d.update(serve_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    prep=target/'publication-bindings.json';value=json.loads(prep.read_text())
    if mutation=='missing_source_bindings':value['source_files']={}
    if mutation=='acceptance_corpus':
        value['acceptance_names']['final-acceptance']=['english-1']
        edit(target/'final-acceptance/asr.json',lambda d:d.update(rows=[r for r in d['rows'] if r['name']=='english-1']))
    # Exercise semantic gates independently of the fixed production snapshot:
    # explicitly authorize only this test copy's recomputed map in memory.
    # The rewritten-root case deliberately does NOT receive that authorization.
    if mutation!='rewritten_empirical_root':
        anchor=target/'captured-artifacts.json';a=json.loads(anchor.read_text());a['files']={n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in a['files'] if (target/n).is_file()};anchor.write_text(json.dumps(a,sort_keys=True,indent=2)+'\n');module.CAPTURE_ROOT=hashlib.sha256(anchor.read_bytes()).hexdigest()
    value['evidence_files']={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file() and p!=prep};prep.write_text(json.dumps(value,indent=2)+'\n')
    with pytest.raises((ValueError,FileNotFoundError)):module.verify(target)
