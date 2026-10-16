from pathlib import Path
import datetime,hashlib,json,subprocess,shutil,os
source=Path(__file__).resolve().parent;root=source.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def cmd(*args):return subprocess.check_output(args,text=True).strip()
records=read(source/'synchronized-validation.json');assert len(records)==6 and all(r['exit']==0 for r in records)
commit=cmd('git','rev-parse','HEAD');build=read(source/'compressor-wait-build.json')
for name,digest in build['sha256'].items():
 if name.startswith('rust-star/runtime/src/'):
  assert sha(root/name)==digest
  assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name])).hexdigest()==digest, 'Commit validated sources first'
w=source.with_name('paired-32k-128-'+commit[:8]+'-synchronized');assert not w.exists();w.mkdir();inputs=w/'inputs';inputs.mkdir()
plan=read(source/'inputs/plan.json');freeze=read(source/'inputs/freeze.json')
args=plan['candidate']['adapter_command'];model=Path(args[args.index('--model')+1]);assert model.stat().st_size==freeze['model_size'] and model.stat().st_mtime_ns==freeze['model_mtime_ns']
candidate=w/'bin/candidate/rust-star';candidate.parent.mkdir(parents=True);subprocess.run(['cp','-c',str(source/'compressor-wait-rust-star'),str(candidate)],check=True);candidate.chmod(0o555)
oracle=w/'bin/oracle/ds4-bench';oracle.parent.mkdir();shutil.copyfile(source/'oracle-source/ds4-bench',oracle);oracle.chmod(0o555);shutil.copytree(source/'oracle-source/metal',oracle.parent/'metal');assert sha(oracle)==freeze['oracle_sha256']
checks={}
for name in ['synchronized-a','synchronized-b','synchronized-c','synchronized-d']:
 r=read(source/(name+'.json'));assert r['complete_32k_c0_claim'] and r['checked_decode_steps']==128
 checks[name]={'sha256':sha(source/(name+'.json')),'full_output_c0':True,'production_gpu_selection_checked':True}
correctness={'schema':'rust-star-correctness-manifest-v1','correctness_class':'C0','candidate_commit':commit,'candidate_executable_sha256':sha(candidate),'oracle_commit':plan['oracle']['identity']['source_commit'],'oracle_executable_sha256':sha(oracle),'context':32768,'gen_tokens':128,'fresh_candidate_full_output_runs':checks,'candidate_build':build,'shorter_controls':{name:sha(source/(name+'.json')) for name in ['synchronized-2k','synchronized-8k']},'rust_tests_passed':309,'python_tests_passed':88,'reference_build':read(source/'oracle-source/benchmark-build.json'),'reference_full_output_conformance':read(source/'conformance.json'),'prior_correctness_manifest_sha256':sha(source/'inputs/correctness-manifest.json'),'failed_predecessor_plan_sha256':freeze['plan_sha256'],'failure_policy':'Predecessor C0 failure retained, no retry; new source with synchronization repair validated before a fresh plan','internal_hc_accepted_through_tokens':24576,'later_internal_hc_claim':False}
write(inputs/'correctness-manifest.json',correctness)
host=read(source/'inputs/host-before.json');host['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();host['power']={'configuration':cmd('pmset','-g','custom'),'thermal':cmd('pmset','-g','therm')};host['memory']={'swap':cmd('sysctl','vm.swapusage'),'pressure':cmd('memory_pressure','-Q')};host['load']['uptime']=cmd('uptime');displays=json.loads(cmd('system_profiler','SPDisplaysDataType','-json'))['SPDisplaysDataType'];host['displays']=[{'chip':d.get('sppci_model'),'cores':d.get('sppci_cores'),'screens':[{'resolution':x.get('_spdisplays_resolution'),'online':x.get('spdisplays_online')} for x in d.get('spdisplays_ndrvs',[])]} for d in displays];write(inputs/'host-before.json',host)
for role,path in [('candidate',candidate),('oracle',oracle)]:
 e=plan[role];e['working_directory']=str(path.parent);a=e['adapter_command'];a[a.index('--executable')+1]=str(path);identity=e['identity'];identity['executable_sha256']=sha(path)
 if role=='candidate':
  identity['source_commit']=commit;identity['source_tree']=cmd('git','rev-parse',commit+'^{tree}');identity['build_configuration']['command']=['cargo','build','--manifest-path','rust-star/runtime/Cargo.toml','--release'];identity['build_configuration']['validated_build_sha256']=sha(source/'compressor-wait-build.json');identity['runtime_configuration']['compressor_dependency']='host-waited shared odd/even prefill projection-to-consumer boundary; both command buffers counted';identity['runtime_configuration']['failure_collection']='completed logit readback only after failed token comparison; successful steps keep GPU selection readback'
 else:identity['build_configuration']['command']=['python3','rust-star/build_benchmark_oracle.py','--output','$ORACLE_BUILD','--jobs','4']
plan['host_manifest_sha256']=sha(inputs/'host-before.json');plan['correctness_manifest_sha256']=sha(inputs/'correctness-manifest.json');write(inputs/'plan.json',plan)
freeze.update(plan_sha256=sha(inputs/'plan.json'),candidate_sha256=sha(candidate),candidate_commit=commit);write(inputs/'freeze.json',freeze)
for name in ['run-milestone.py','observe-load.py','audit.py']:
 shutil.copyfile(source.with_name('paired-32k-128-7406473')/name,w/name)
runner=w/'run-milestone.py';script=runner.read_text();marker=" (w/'inputs'/name).write_text"
script=script.replace(marker,""" if name.startswith('host-after'):
  record['power_configuration']=subprocess.check_output(['pmset','-g','custom'],text=True).strip()
  displays=json.loads(subprocess.check_output(['system_profiler','SPDisplaysDataType','-json'],text=True))['SPDisplaysDataType']
  record['displays']=[{'chip':d.get('sppci_model'),'cores':d.get('sppci_cores'),'screens':[{'resolution':x.get('_spdisplays_resolution'),'online':x.get('spdisplays_online')} for x in d.get('spdisplays_ndrvs',[])]} for d in displays]
"""+marker);runner.write_text(script)
write(source/'synchronized-plan-location.json',{'work':str(w),'plan_sha256':freeze['plan_sha256']});print(w,freeze['plan_sha256'])
