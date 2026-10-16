from pathlib import Path
import datetime,hashlib,io,json,os,subprocess,tarfile
root=Path.cwd(); w=root/'rust-star/.work/paired-32k-128-7406473';inputs=w/'inputs'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def cmd(*args):return subprocess.check_output(args,text=True).strip()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');prompt=root/'speed-bench/promessi_sposi.txt'
print('Hashing pinned model and prompt',flush=True)
assert sha(model)=='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0'
assert sha(prompt)=='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f'
oracle_commit='d35fb12d01d500b9cefcef24092c295687ceaf7e'
for source,dest,expected in [(root/'rust-star/.work/oracle-v3-8k-source/ds4-bench.accepted-oracle-v3',w/'bin/oracle/ds4-bench','8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3'),(root/'rust-star/.work/runtime-target/release/rust-star',w/'bin/candidate/rust-star','8d9b607331fa4dce37caeff0a8ef7d2b9c3cc8f9285eb0590f347cca4a6aa84d')]:
 subprocess.run(['cp','-c',str(source),str(dest)],check=True);assert sha(dest)==expected;dest.chmod(0o555)
archive=subprocess.check_output(['git','archive',oracle_commit,'metal'])
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
 for member in tar.getmembers():
  assert not member.name.startswith('/') and '..' not in Path(member.name).parts
 tar.extractall(w/'bin/oracle')
# Only explicitly enumerated, non-identifying host values are retained.
host={'schema':'rust-star-host-manifest-v1','captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hardware':{k:cmd('sysctl','-n',k) for k in ['hw.model','hw.memsize','hw.logicalcpu','hw.physicalcpu']},'software':{'macos':cmd('sw_vers'),'kernel':cmd('uname','-srm'),'xcode':cmd('xcodebuild','-version'),'clang':cmd('xcrun','clang','--version').splitlines()[0],'rustc':cmd('rustc','--version'),'cargo':cmd('cargo','--version')},'power':{'configuration':cmd('pmset','-g','custom'),'thermal':cmd('pmset','-g','therm')},'memory':{'swap':cmd('sysctl','vm.swapusage'),'pressure':cmd('memory_pressure','-Q')},'load':{'uptime':cmd('uptime'),'material_background_state':'Codex UI active; no other model process observed; one engine process at a time; no builds or other agent workloads during timed pairs'},'environment_allowlist':{},'environment_policy':'Remove inherited DS4_*, RUST_STAR_*, METAL_* and MTL_* overrides before runner; default inference configuration'}
displays=json.loads(cmd('system_profiler','SPDisplaysDataType','-json'))['SPDisplaysDataType'];host['displays']=[{'chip':d.get('sppci_model'),'cores':d.get('sppci_cores'),'screens':[{'resolution':x.get('_spdisplays_resolution'),'online':x.get('spdisplays_online')} for x in d.get('spdisplays_ndrvs',[])]} for d in displays]
write(inputs/'host-before.json',host)
old=root/'rust-star/.work/native-32k-v31';verification=json.loads((old/'final-verification.json').read_text())
for name,h in verification['source_sha256'].items():assert sha(root/name)==h
for name,h in verification['artifacts_sha256'].items():assert sha(old/name)==h
correctness={'schema':'rust-star-correctness-manifest-v1','correctness_class':'C0','candidate_commit':cmd('git','rev-parse','HEAD'),'candidate_executable_sha256':sha(w/'bin/candidate/rust-star'),'oracle_commit':oracle_commit,'oracle_executable_sha256':sha(w/'bin/oracle/ds4-bench'),'context':32768,'gen_tokens':128,'prior_cross_artifact_verification_sha256':sha(old/'final-verification.json'),'validation':verification,'source_delta_after_full_output_c0':'Explicit native first-token feedback (same validated token) plus CLI help/comment changes; no GPU arithmetic or inference schedule changes. Build13 fresh/retained engine and 2K/8K controls pass; final build14 CLI checks and 309 Rust tests pass.','oracle_runtime_metal_sha256':{str(p.relative_to(w/'bin/oracle')):sha(p) for p in sorted((w/'bin/oracle/metal').rglob('*')) if p.is_file()}}
write(inputs/'correctness-manifest.json',correctness)
plan=json.loads((root/'rust-star/.work/paired-2k-128-2c6f80e-r1/inputs/plan.json').read_text());plan['host_manifest_sha256']=sha(inputs/'host-before.json');plan['correctness_manifest_sha256']=sha(inputs/'correctness-manifest.json');plan['configuration']['contexts']=[32768]
for role,exe,commit in [('oracle',w/'bin/oracle/ds4-bench',oracle_commit),('candidate',w/'bin/candidate/rust-star',cmd('git','rev-parse','HEAD'))]:
 entry=plan[role];ident=entry['identity'];ident['source_commit']=commit;ident['source_tree']=cmd('git','rev-parse',commit+'^{tree}');ident['executable_sha256']=sha(exe);entry['working_directory']=str(exe.parent);entry['timeout_seconds']=3600
 args=['python3',str(root/('rust-star/measure_dwarfstar.py' if role=='oracle' else 'rust-star/measure_ruststar.py')),'--executable',str(exe),'--model',str(model)]
 if role=='oracle':args+=['--prompt',str(prompt)]
 args+=['--context','{context}','--gen-tokens','{gen_tokens}','--output','{output}','--timeout-seconds','3500'];entry['adapter_command']=args
 if role=='oracle':ident['runtime_configuration'].update(oracle_version='oracle-v3',prefill_chunk_tokens=4096,compressor_dependency='accepted host-waited projection-to-read boundary')
 else:ident['runtime_configuration'].update(prefill_context_tokens=32768,prefill_chunk_tokens=4096,exact_prefix_rebuild=True,decode_steps=128)
write(inputs/'plan.json',plan);write(inputs/'freeze.json',{'plan_sha256':sha(inputs/'plan.json'),'source_commit':cmd('git','rev-parse','HEAD'),'model_size':model.stat().st_size,'model_mtime_ns':model.stat().st_mtime_ns,'model_sha256':ident['model_sha256'],'prompt_sha256':sha(prompt),'candidate_sha256':sha(w/'bin/candidate/rust-star'),'oracle_sha256':sha(w/'bin/oracle/ds4-bench')})
print('Frozen plan',sha(inputs/'plan.json'),flush=True)
