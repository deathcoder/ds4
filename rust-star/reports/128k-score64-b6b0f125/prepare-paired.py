from pathlib import Path
import json,hashlib,subprocess,datetime,copy,sys
w=Path(__file__).resolve().parent;root=w.parents[2];sys.path.insert(0,str(root/'rust-star'))
from capture_oracle_v1 import collect_host_manifest
from paired_runner_lib import load_paired_plan
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
subprocess.run([sys.executable,str(w/'audit-gates.py')],check=True)
a=read(w/'validation-acceptance.json');assert a['passed'] and len(a['checks'])==10
b=read(w/'candidate-build.json');commit=b['commit']
for n,h in b['sources'].items():assert sha(root/n)==h==hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=root)).hexdigest(),n
assert sha(w/'candidate-rust-star')==b['executable_sha256']
assert sha(w/'optimization-acceptance.json')==b['acceptance_plan_sha256']
assert read(w/'checks/component-audit.json')['passed'] and read(w/'checks/components-128k-audit.json')['passed']
assert 'Ran 94 tests' in (w/'python-tests.log').read_text() and (w/'python-tests.log').read_text().rstrip().endswith('OK')
prior=root/'rust-star/reports/128k-measure-ef37e566';accepted=read(prior/'candidate-build.json');base=read(w/'baseline.json')
assert accepted['commit']==base['baseline_source_commit'] and sha(w/'baseline-rust-star')==accepted['executable_sha256']==base['executable_sha256']
plan=read(root/'rust-star/.work/paired-128k-ef37e566/inputs/plan.json');plan['oracle']=copy.deepcopy(plan['candidate'])
i=read(w/'preflight.json');o=root/'rust-star/.work/oracle-v5-source'
model=Path(plan['candidate']['adapter_command'][plan['candidate']['adapter_command'].index('--model')+1])
assert sha(model)==i['model_sha256'] and model.stat().st_size==i['model_bytes']
assert sha(root/'speed-bench/promessi_sposi.txt')==i['prompt_sha256'] and sha(o/'ds4-bench')==i['oracle_executable_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip()==i['oracle_source']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=o,text=True).strip()
p=root/('rust-star/.work/paired-128k-score64-'+commit[:8]);assert not p.exists();p.mkdir();(p/'inputs').mkdir()
for role,source,build in [('candidate',w/'candidate-rust-star',b),('oracle',w/'baseline-rust-star',accepted)]:
 exe=p/'bin'/role/source.name;exe.parent.mkdir(parents=True);subprocess.run(['cp','-c',str(source),str(exe)],check=True);exe.chmod(0o555)
 assert sha(exe)==build['executable_sha256']
 item=plan[role];item['working_directory']=str(exe.parent);args=item['adapter_command'];args[args.index('--executable')+1]=str(exe);args[args.index('--timeout-seconds')+1]='7100';item['timeout_seconds']=7200
 identity=item['identity'];identity.update(source_commit=build['commit'],source_tree=build['tree'],executable_sha256=build['executable_sha256'])
 identity['build_configuration']['validated_build_sha256']=sha(w/'candidate-build.json' if role=='candidate' else prior/'candidate-build.json')
 if role=='candidate':
  identity['build_configuration']['rustc']=subprocess.check_output(['rustc','--version'],text=True).strip();identity['build_configuration']['cargo']=subprocess.check_output(['cargo','--version'],text=True).strip()
 identity['runtime_configuration'].update(comparison_role='optimized Rust' if role=='candidate' else 'accepted Rust baseline; not the C0 oracle',prefill_indexer_score_query_tile=8,prefill_indexer_score_columns=64 if role=='candidate' else 32,prefill_indexer_score_threads=256 if role=='candidate' else 128,prefill_indexer_score_shared_bytes=20480 if role=='candidate' else 11264,prefill_stage_profile_collection=False)
plan['configuration']['contexts']=[131072];plan['configuration']['repetitions']=5
(p/'inputs/acceptance-plan.json').write_bytes((w/'optimization-acceptance.json').read_bytes())
(p/'inputs/baseline-build.json').write_bytes((prior/'candidate-build.json').read_bytes())
write(p/'inputs/protocol-scope.json',dict(runner_schema_primary_metric='gen_steady_tps',study_primary_metric='prefill_ms',explanation='The unchanged runner schema fixes its generic primary metric to steady decode. The separately frozen optimization acceptance evaluates prefill latency and guards decode/RSS using independently recomputed metrics.'))
correctness={'schema':'rust-star-correctness-manifest-v1','correctness_class':'C0','context':131072,'gen_tokens':128,'comparison':'optimized Rust versus accepted Rust; the runner oracle slot is only a comparator label','candidate_commit':commit,'candidate_executable_sha256':b['executable_sha256'],'baseline_commit':accepted['commit'],'baseline_executable_sha256':accepted['executable_sha256'],'c0_oracle_commit':i['oracle_source'],'c0_oracle_executable_sha256':i['oracle_executable_sha256'],'model_sha256':i['model_sha256'],'prompt_sha256':i['prompt_sha256'],'candidate_gates':{x['name']:sha(w/'checks'/(x['name']+'.json')) for x in a['checks']},'candidate_build_sha256':sha(w/'candidate-build.json'),'profile_audit_sha256':sha(w/'checks/component-audit.json'),'profile128_audit_sha256':sha(w/'checks/components-128k-audit.json'),'baseline_report_manifest_sha256':sha(prior/'artifact-hashes.json'),'reference_manifest_sha256':sha(root/'rust-star/reports/128k-c0-0eb484ea/reference-131072/manifest.json'),'fixture_manifest_sha256':sha(root/'rust-star/fixtures/measurement-131072-v5/manifest.json'),'rust_tests_passed':317,'python_tests_passed':94,'acceptance_plan_sha256':sha(p/'inputs/acceptance-plan.json'),'protocol_scope_sha256':sha(p/'inputs/protocol-scope.json'),'measurement_tools_sha256':{n:sha(root/'rust-star'/n) for n in ['run_paired_benchmark.py','paired_runner_lib.py','paired_benchmark_lib.py','measure_ruststar.py','ruststar_measurement_lib.py','dwarfstar_measurement_lib.py']}}
write(p/'inputs/correctness-manifest.json',correctness)
host=collect_host_manifest();host['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();host['swap']=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip();host['interactive_host']=True;host=json.loads(json.dumps(host).replace(str(root),'$WORKTREE').replace('$USER_DIRECTORY/','$USER_DIRECTORY/'));write(p/'inputs/host-before.json',host)
plan['host_manifest_sha256']=sha(p/'inputs/host-before.json');plan['correctness_manifest_sha256']=sha(p/'inputs/correctness-manifest.json');write(p/'inputs/plan.json',plan);load_paired_plan(p/'inputs/plan.json')
write(p/'inputs/freeze.json',dict(plan_sha256=sha(p/'inputs/plan.json'),acceptance_plan_sha256=sha(p/'inputs/acceptance-plan.json'),candidate_commit=commit,candidate_sha256=b['executable_sha256'],oracle_commit=accepted['commit'],oracle_sha256=accepted['executable_sha256'],c0_oracle_source=i['oracle_source'],c0_oracle_sha256=i['oracle_executable_sha256'],model_sha256=i['model_sha256'],model_size=model.stat().st_size,model_mtime_ns=model.stat().st_mtime_ns,prompt_sha256=i['prompt_sha256'],comparison='Rust-to-Rust development improvement; unchanged oracle-v5 is separate'))
write(w/'paired-location.json',{'work':str(p),'plan_sha256':sha(p/'inputs/plan.json')});print(p,flush=True)
