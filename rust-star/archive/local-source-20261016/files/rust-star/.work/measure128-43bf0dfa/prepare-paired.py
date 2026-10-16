from pathlib import Path
import json,hashlib,subprocess,datetime,shutil,sys
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
subprocess.run([sys.executable,str(w/'audit-gates.py')],check=True)
a=read(w/'validation-acceptance.json');assert a['passed'];b=read(w/'candidate-build.json');commit=b['commit']
for n,h in b['sources'].items():assert sha(root/n)==h
assert read(w/'checks/component-audit.json')['passed'];assert 'Ran 94 tests' in (w/'python-tests.log').read_text() and (w/'python-tests.log').read_text().rstrip().endswith('OK')
old=w.with_name('paired-32k-128-89e32745-tail');plan=read(old/'inputs/plan.json');i=read(w/'preflight.json')['inputs'];o=w.with_name('oracle-v5-source')
model=Path(plan['candidate']['adapter_command'][plan['candidate']['adapter_command'].index('--model')+1]);assert sha(model)==i['model_sha256'];assert sha(root/'speed-bench/promessi_sposi.txt')==i['prompt_sha256'];assert sha(o/'ds4-bench')==i['oracle_executable_sha256']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=o,text=True).strip()
p=w.with_name('paired-128k-'+commit[:8]);assert not p.exists();p.mkdir();(p/'inputs').mkdir()
for role,source in [('candidate',w/'candidate-rust-star'),('oracle',o/'ds4-bench')]:
 exe=p/'bin'/role/source.name;exe.parent.mkdir(parents=True);subprocess.run(['cp','-c',str(source),str(exe)],check=True);exe.chmod(0o555)
 if role=='oracle':shutil.copytree(o/'metal',exe.parent/'metal')
 item=plan[role];item['working_directory']=str(exe.parent);args=item['adapter_command'];args[args.index('--executable')+1]=str(exe);args[args.index('--timeout-seconds')+1]='7100';item['timeout_seconds']=7200
 identity=item['identity'];assert sha(exe)==(b['executable_sha256'] if role=='candidate' else i['oracle_executable_sha256'])
 if role=='candidate':
  identity.update(source_commit=commit,source_tree=b['tree'],executable_sha256=sha(exe));identity['build_configuration']['validated_build_sha256']=sha(w/'candidate-build.json');identity['build_configuration']['rustc']=subprocess.check_output(['rustc','--version'],text=True).strip();identity['build_configuration']['cargo']=subprocess.check_output(['cargo','--version'],text=True).strip()
  identity['runtime_configuration'].update(prefill_context_tokens=131072,initial_bootstrap_last_completed_layer=2,initial_bootstrap_output_head=False,native_activation_scratch_rows=4096,partial_bootstrap_model_views=18,transcript_fixture='measurement-131072-v5',transcript_fixture_manifest_sha256=sha(root/'rust-star/fixtures/measurement-131072-v5/manifest.json'))
plan['configuration']['contexts']=[131072];plan['configuration']['repetitions']=5
acceptance={'scope':'128K/128 characterization, Rust Star versus unchanged oracle-v5','pairs':5,'warmups_per_engine':1,'order':'alternating AB then BA, fresh processes','correctness':'complete 128K full-logit gate plus timed exact transcripts and shorter controls','retained_scope':'two 128K transcript-checked cycles; fresh-process C0 independent; not a paired sample','primary_metric':'gen_steady_tps','secondary_metrics':['prefill_ms','gen_first_ms','gen_tps','process_peak_bytes','process_wall_ms'],'speedup_required':False,'failure_policy':'stop on first failure; preserve all attempts; no favorable retries','claim_limit':'128K characterization only; 256K headline and coding-agent corpus remain unmeasured'}
write(p/'inputs/acceptance-plan.json',acceptance)
correctness={'schema':'rust-star-correctness-manifest-v1','correctness_class':'C0','context':131072,'gen_tokens':128,'candidate_commit':commit,'candidate_executable_sha256':b['executable_sha256'],'oracle_commit':i['oracle_source'],'oracle_executable_sha256':i['oracle_executable_sha256'],'model_sha256':i['model_sha256'],'prompt_sha256':i['prompt_sha256'],'candidate_gates':{x['name']:sha(w/'checks'/(x['name']+'.json')) for x in a['checks']},'candidate_build_sha256':sha(w/'candidate-build.json'),'profile_audit_sha256':sha(w/'checks/component-audit.json'),'prior_full_repeat_report_manifest_sha256':sha(root/'rust-star/reports/128k-c0-0eb484ea/artifact-hashes.json'),'reference_manifest_sha256':sha(root/'rust-star/reports/128k-c0-0eb484ea/reference-131072/manifest.json'),'fixture_manifest_sha256':sha(root/'rust-star/fixtures/measurement-131072-v5/manifest.json'),'rust_tests_passed':315,'python_tests_passed':94,'acceptance_plan_sha256':sha(p/'inputs/acceptance-plan.json'),'measurement_tools_sha256':{n:sha(root/'rust-star'/n) for n in ['run_paired_benchmark.py','paired_runner_lib.py','paired_benchmark_lib.py','measure_dwarfstar.py','dwarfstar_measurement_lib.py','measure_ruststar.py','ruststar_measurement_lib.py']}}
write(p/'inputs/correctness-manifest.json',correctness)
host=collect_host_manifest();host['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();host['swap']=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip();host['interactive_host']=True;host=json.loads(json.dumps(host).replace(str(root),'$WORKTREE').replace('/Users/deathcodevision/','$USER_DIRECTORY/'));write(p/'inputs/host-before.json',host)
plan['host_manifest_sha256']=sha(p/'inputs/host-before.json');plan['correctness_manifest_sha256']=sha(p/'inputs/correctness-manifest.json');write(p/'inputs/plan.json',plan);load_paired_plan(p/'inputs/plan.json')
write(p/'inputs/freeze.json',dict(plan_sha256=sha(p/'inputs/plan.json'),candidate_commit=commit,candidate_sha256=b['executable_sha256'],oracle_sha256=i['oracle_executable_sha256'],model_sha256=i['model_sha256'],model_size=model.stat().st_size,model_mtime_ns=model.stat().st_mtime_ns,prompt_sha256=i['prompt_sha256']))
write(p/'inputs/oracle-metal-hashes.json',{str(f.relative_to(p/'bin/oracle')):sha(f) for f in sorted((p/'bin/oracle/metal').rglob('*')) if f.is_file()})
write(w/'paired-location.json',{'work':str(p),'plan_sha256':sha(p/'inputs/plan.json')});print(p,flush=True)
