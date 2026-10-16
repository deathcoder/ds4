from pathlib import Path
import datetime,hashlib,json,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];old=w.with_name('paired-32k-128-7406473');inputs=w/'inputs';inputs.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def cmd(*args):return subprocess.check_output(args,text=True).strip()
assert read(w/'conformance.json')['passed'] is True
plan=read(old/'inputs/plan.json');freeze=read(old/'inputs/freeze.json');model=Path(plan['candidate']['adapter_command'][plan['candidate']['adapter_command'].index('--model')+1]);assert model.stat().st_size==freeze['model_size'] and model.stat().st_mtime_ns==freeze['model_mtime_ns']
exe=w/'bin/candidate/rust-star';exe.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['cp','-c',str(old/'bin/candidate/rust-star'),str(exe)],check=True);assert sha(exe)==freeze['candidate_sha256'];exe.chmod(0o555)
oracle=w/'oracle-source/ds4-bench';oracle.chmod(0o555);build=read(w/'oracle-source/benchmark-build.json');assert sha(oracle)==build['executable_sha256']
commit=cmd('git','rev-parse','HEAD');assert hashlib.sha256(subprocess.check_output(['git','show',commit+':ds4_bench.c'])).hexdigest()==build['benchmark_sha256']
host=read(old/'inputs/host-before.json');host['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();host['power']={'configuration':cmd('pmset','-g','custom'),'thermal':cmd('pmset','-g','therm')};host['memory']={'swap':cmd('sysctl','vm.swapusage'),'pressure':cmd('memory_pressure','-Q')};host['load']={'uptime':cmd('uptime'),'material_background_state':'Desktop applications remain open; lightweight CPU, thermal and swap observations retained. No other model process observed, no builds during timed block.'};write(inputs/'host-before.json',host)
correctness=read(old/'inputs/correctness-manifest.json');correctness['oracle_executable_sha256']=sha(oracle);correctness['benchmark_driver_commit']=commit;correctness['benchmark_build_provenance']=build;correctness['repaired_driver_full_output_conformance']=read(w/'conformance.json');correctness['repaired_driver_repetitions']={rep:sha(w/f'conformance-{rep}/verification.json') for rep in ['a','b']};correctness['legacy_plan_rejected']=freeze['plan_sha256'];write(inputs/'correctness-manifest.json',correctness)
plan['host_manifest_sha256']=sha(inputs/'host-before.json');plan['correctness_manifest_sha256']=sha(inputs/'correctness-manifest.json')
for role,path in [('oracle',oracle),('candidate',exe)]:
 entry=plan[role];entry['working_directory']=str(path.parent);args=entry['adapter_command'];args[args.index('--executable')+1]=str(path)
 if role=='oracle':
  entry['identity']['executable_sha256']=sha(path);entry['identity']['build_configuration'].update(benchmark_driver_commit=commit,benchmark_driver_sha256=build['benchmark_sha256'],build_provenance_sha256=sha(w/'oracle-source/benchmark-build.json'),builder='rust-star/build_benchmark_oracle.py',source_overlay='ds4_bench.c only; all inference sources pinned to oracle-v3')
  entry['identity']['runtime_configuration'].update(sampling_in_interval=True,initial_selection_charged_to_prefill=True,decode_step='eval plus resulting-token argmax and commitment',csv_timing_precision_digits=9)
write(inputs/'plan.json',plan);freeze.update(plan_sha256=sha(inputs/'plan.json'),oracle_sha256=sha(oracle),benchmark_driver_commit=commit);write(inputs/'freeze.json',freeze)
print('Frozen sampling-inclusive plan:',freeze['plan_sha256'])
