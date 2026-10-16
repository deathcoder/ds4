from pathlib import Path
import json,hashlib,subprocess,datetime,copy
work=Path(__file__).resolve().parent;root=work.parents[2];previous=work.with_name('paired-32k-128-89e32745-tail')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def cmd(*args):return subprocess.check_output(args,text=True).strip()
checks=read(work/'batch-validation.json');assert len(checks)==7 and all(x['exit']==0 for x in checks)
assert read(work/'component-audit.json')['passed']
build=read(work/'batch-build.json');commit=build['commit']
for name,digest in build['sources'].items():assert sha(root/name)==digest==hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name])).hexdigest()
assert '312 passed; 0 failed' in (work/'batch-tests-fixed.log').read_text()
accepted=read(work/'accepted-build.json');assert sha(work/'accepted-rust-star')==accepted['executable_sha256']
plan=read(previous/'inputs/plan.json');oldfreeze=read(previous/'inputs/freeze.json')
args=plan['candidate']['adapter_command'];model=Path(args[args.index('--model')+1]);assert sha(model)==oldfreeze['model_sha256'];assert model.stat().st_size==oldfreeze['model_size'] and model.stat().st_mtime_ns==oldfreeze['model_mtime_ns']
prompt=root/'speed-bench/promessi_sposi.txt';assert sha(prompt)==oldfreeze['prompt_sha256'];print('PASS model/prompt hashes',flush=True)
p=work.with_name('paired-rust-memory-'+commit[:8]);assert not p.exists();p.mkdir();(p/'inputs').mkdir()
plan['oracle']=copy.deepcopy(plan['candidate'])
for role,source,source_commit,tree in [('candidate',work/'batch-rust-star',commit,build['tree']),('oracle',work/'accepted-rust-star',cmd('git','rev-parse','912fae1e'),cmd('git','rev-parse','912fae1e^{tree}'))]:
 exe=p/'bin'/role/'rust-star';exe.parent.mkdir(parents=True);subprocess.run(['cp','-c',str(source),str(exe)],check=True);exe.chmod(0o555)
 item=plan[role];item['working_directory']=str(exe.parent);args=item['adapter_command'];args[args.index('--executable')+1]=str(exe)
 identity=item['identity'];identity.update(source_commit=source_commit,source_tree=tree,executable_sha256=sha(exe))
 identity['build_configuration']['validated_build_sha256']=sha(work/('batch-build.json' if role=='candidate' else 'accepted-build.json'))
 identity['runtime_configuration'].update(comparison_role='optimized Rust' if role=='candidate' else 'accepted Rust baseline (not the C0 oracle)',initial_bootstrap_last_completed_layer=2,initial_bootstrap_output_head=False,native_tail_dispatches=25,native_tail_scratch_buffers=22,prefill_stage_profile_collection=False)
for name in ('batch-32k-output','batch-32k-internal','batch-32k-profiled'):
 r=read(work/(name+'.json'));assert r['complete_32k_c0_claim'] and r['checked_decode_steps']==128
reference=read(root/'rust-star/reports/32k-dense-output-92ca3ef3/correctness/conformance-v5.json')
correctness={'schema':'rust-star-correctness-manifest-v1','correctness_class':'C0','comparison':'optimized Rust versus accepted Rust; paired oracle slot is a comparator label only','c0_oracle_source':'ccbd16d36895ef9fe8edf7f793014988ff1ab7f4','c0_oracle_executable_sha256':reference['executable_sha256'],'candidate_build':build,'accepted_build':accepted,'candidate_gates':{x['name']:sha(work/(x['name']+'.json')) for x in checks},'candidate_profile_audit_sha256':sha(work/'component-audit.json'),'accepted_report_manifest_sha256':sha(root/'rust-star/reports/32k-bootstrap-912fae1e/artifact-hashes.json'),'rust_tests_passed':312,'model_rehashed':True,'internal_hc_accepted_through_tokens':24576,'full_32k_prefill_and_128_decode_words_required':True}
correctness['measurement_tools_sha256']={name:sha(root/'rust-star'/name) for name in ['run_paired_benchmark.py', 'paired_runner_lib.py', 'paired_benchmark_lib.py', 'measure_ruststar.py', 'ruststar_measurement_lib.py', 'dwarfstar_measurement_lib.py']}
correctness['optimization_acceptance']={'prefill':'Require median paired candidate/baseline latency <= 1.03. Report every pair and dispersion; no favorable retries.', 'memory':'Require lower median peak RSS and lower RSS in at least four of five pairs. Report all ratios and absolute GiB.','decode':'Require median paired steady decode throughput >= 0.97; report first-step latency and every pair.','failures':'Stop on failure; preserve all attempts; no retries to obtain favorable results.','profile':'Separate diagnostic run after timed block. Raw invalid-counter values must be preserved, not accepted as valid timing.','claim_scope':'32K/128 Rust-to-Rust development improvement; no updated Rust-versus-DwarfStar throughput claim.'}
write(p/'inputs/acceptance-plan.json',correctness['optimization_acceptance'])
write(p/'inputs/correctness-manifest.json',correctness)
host=read(previous/'inputs/host-before.json');host['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();host['power']={'configuration':cmd('pmset','-g','custom'),'thermal':cmd('pmset','-g','therm')};host['memory']={'swap':cmd('sysctl','vm.swapusage'),'pressure':cmd('memory_pressure','-Q')};host['load']['uptime']=cmd('uptime');host['displays']=[{'chip':d.get('sppci_model'),'cores':d.get('sppci_cores'),'screens':[{'resolution':x.get('_spdisplays_resolution'),'online':x.get('spdisplays_online')} for x in d.get('spdisplays_ndrvs',[])]} for d in json.loads(cmd('system_profiler','SPDisplaysDataType','-json'))['SPDisplaysDataType']];host['inherited_static_metadata_source']=str(previous/'inputs/host-before.json');host['recovery_boot_time']=cmd('sysctl','kern.boottime');write(p/'inputs/host-before.json',host)
plan['host_manifest_sha256']=sha(p/'inputs/host-before.json');plan['correctness_manifest_sha256']=sha(p/'inputs/correctness-manifest.json');write(p/'inputs/plan.json',plan)
freeze={k:oldfreeze[k] for k in ('model_size','model_mtime_ns','model_sha256','prompt_sha256')}
freeze.update(acceptance_plan_sha256=sha(p/'inputs/acceptance-plan.json'),plan_sha256=sha(p/'inputs/plan.json'),candidate_commit=commit,candidate_sha256=sha(p/'bin/candidate/rust-star'),oracle_commit=plan['oracle']['identity']['source_commit'],oracle_sha256=sha(p/'bin/oracle/rust-star'),comparison='Rust-to-Rust optimization study; C0 oracle-v5 is separate',c0_oracle_source=correctness['c0_oracle_source'])
write(p/'inputs/freeze.json',freeze)
assert read(p/'inputs/acceptance-plan.json')==correctness['optimization_acceptance']
for name in ('run-milestone.py','observe-load.py'):(p/name).write_text((previous/name).read_text())
write(work/'paired-location.json',{'work':str(p),'plan_sha256':freeze['plan_sha256']});print(p,flush=True)
