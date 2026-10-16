from pathlib import Path
import json,subprocess,hashlib,datetime,shutil
source=Path(__file__).resolve().parent;root=source.parents[2];old=source.with_name('paired-32k-128-cbb078bb-topk')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def cmd(*args):return subprocess.check_output(args,text=True).strip()
records=read(source/'batch-validation.json');assert len(records)==6 and all(r['exit']==0 for r in records)
build=read(source/'batch-build.json');commit=build['commit']
for name,digest in build['sources'].items():assert sha(root/name)==digest==hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name])).hexdigest()
assert '310 passed; 0 failed' in (source/'batch-tests-fixed.log').read_text()
assert 'Ran 93 tests' in (source/'python-tests.log').read_text() and (source/'python-tests.log').read_text().rstrip().endswith('OK')
assert read(source/'oracle-v5-verification.json')['status']=='complete'
assert read(source/'conformance-v5.json')['passed']
plan=read(old/'inputs/plan.json');freeze=read(old/'inputs/freeze.json');args=plan['candidate']['adapter_command'];model=Path(args[args.index('--model')+1]);assert sha(model)==freeze['model_sha256'];assert model.stat().st_size==freeze['model_size'] and model.stat().st_mtime_ns==freeze['model_mtime_ns'];print('PASS model hash',flush=True)
args=plan['oracle']['adapter_command'];assert sha(Path(args[args.index('--prompt')+1]))==freeze['prompt_sha256']
w=source.with_name('paired-32k-128-'+commit[:8]+'-v5');assert not w.exists();w.mkdir();inputs=w/'inputs';inputs.mkdir()
candidate=w/'bin/candidate/rust-star';candidate.parent.mkdir(parents=True);subprocess.run(['cp','-c',str(source/'batch-rust-star'),str(candidate)],check=True);candidate.chmod(0o555);assert sha(candidate)==build['executable_sha256']
oracle=w/'bin/oracle/ds4-bench';oracle.parent.mkdir(parents=True);o=source.with_name('oracle-v5-source');shutil.copyfile(o/'ds4-bench',oracle);oracle.chmod(0o555);shutil.copytree(o/'metal',oracle.parent/'metal');reference=read(source/'oracle-v5-benchmark-build.json');assert sha(oracle)==reference['executable_sha256'];freeze['oracle_sha256']=sha(oracle);freeze['oracle_commit']=reference['inference_source_commit'];freeze['benchmark_driver_commit']=reference['inference_source_commit']
checks={}
for name in ['batch-32k-output','batch-32k-internal']:
 r=read(source/(name+'.json'));assert r['complete_32k_c0_claim'] and r['checked_decode_steps']==128
 checks[name]={'sha256':sha(source/(name+'.json')),'full_output_c0':True,'production_gpu_selection_checked':True}
correctness={'schema':'rust-star-correctness-manifest-v1','correctness_class':'C0','candidate_commit':commit,'candidate_executable_sha256':sha(candidate),'oracle_commit':freeze['oracle_commit'],'oracle_executable_sha256':sha(oracle),'context':32768,'gen_tokens':128,'fresh_candidate_full_output_runs':checks,'candidate_build':build,'shorter_controls':{n:sha(source/(n+'.json')) for n in ['batch-12k','batch-2k','batch-8k']},'retained_two_cycle_control_sha256':sha(source/'batch-retained.json'),'rust_tests_passed':310,'python_tests_passed':93,'reference_acceptance_manifest_sha256':sha(source/'oracle-v5-acceptance/manifest.json'),'reference_unchanged':False,'reference_evidence':'oracle-v5-acceptance plus two full129-row traces','reference_trace_sha256':sha(source/'conformance-v5.json'),'internal_hc_accepted_through_tokens':24576,'later_internal_hc_claim':False,'failure_policy':'stop on first invalid attempt; retain all evidence; no favorable retries','model_rehashed_before_plan':True}
write(inputs/'correctness-manifest.json',correctness)
host=read(old/'inputs/host-before.json');host['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();host['power']={'configuration':cmd('pmset','-g','custom'),'thermal':cmd('pmset','-g','therm')};host['memory']={'swap':cmd('sysctl','vm.swapusage'),'pressure':cmd('memory_pressure','-Q')};host['load']['uptime']=cmd('uptime')
displays=json.loads(cmd('system_profiler','SPDisplaysDataType','-json'))['SPDisplaysDataType'];host['displays']=[{'chip':d.get('sppci_model'),'cores':d.get('sppci_cores'),'screens':[{'resolution':x.get('_spdisplays_resolution'),'online':x.get('spdisplays_online')} for x in d.get('spdisplays_ndrvs',[])]} for d in displays];write(inputs/'host-before.json',host)
for role,path in [('candidate',candidate),('oracle',oracle)]:
 e=plan[role];e['working_directory']=str(path.parent);a=e['adapter_command'];a[a.index('--executable')+1]=str(path)
 if role=='candidate':
  identity=e['identity'];identity['executable_sha256']=sha(path);identity['source_commit']=commit;identity['source_tree']=build['tree'];identity['build_configuration']['validated_build_sha256']=sha(source/'batch-build.json');identity['runtime_configuration'].update(even_prefill_tail_rows=4096,even_prefill_attention_tile_rows=32,even_prefill_attention_retention='GPU blit in each existing attention command; consecutive complete chunk required',prefill_stage_profile_collection=False)
identity=plan['oracle']['identity'];identity.update(source_commit=reference['inference_source_commit'],source_tree=reference['inference_source_tree'],executable_sha256=sha(oracle))
identity['build_configuration'].update(command=['make','-j4','ds4-bench'],benchmark_driver_commit=reference['inference_source_commit'],benchmark_driver_sha256=reference['benchmark_sha256'],build_provenance_sha256=sha(source/'oracle-v5-benchmark-build.json'),builder='make in clean pinned oracle-v5 worktree',source_overlay='none')
identity['runtime_configuration'].update(oracle_version='oracle-v5',compressor_dependency='pinned current upstream ordering',indexer_selector='deterministic block argsort and merge')
identity['runtime_configuration'].pop('streaming_topk_count_snapshot_barrier',None)
plan['candidate']['identity']['runtime_configuration']['indexer_selector']='upstream deterministic block argsort and merge'
plan['candidate']['identity']['runtime_configuration'].pop('streaming_topk_count_snapshot_barrier',None)
plan['host_manifest_sha256']=sha(inputs/'host-before.json');plan['correctness_manifest_sha256']=sha(inputs/'correctness-manifest.json');write(inputs/'plan.json',plan)
freeze.update(plan_sha256=sha(inputs/'plan.json'),candidate_sha256=sha(candidate),candidate_commit=commit,source_commit=commit,baseline_plan_sha256=read(old/'inputs/freeze.json')['plan_sha256']);write(inputs/'freeze.json',freeze)
for name in ['run-milestone.py','observe-load.py','audit.py']:shutil.copyfile(old/name,w/name)
# The current profile includes long-prefill rows; audit that coverage separately.
p=w/'audit.py';p.write_text(p.read_text().replace("assert profile['prefill_layer_profile'] is None","assert profile['long_prefill_profile']['rows']"))
write(source/'paired-location.json',{'work':str(w),'plan_sha256':freeze['plan_sha256']});print(w,freeze['plan_sha256'],flush=True)
