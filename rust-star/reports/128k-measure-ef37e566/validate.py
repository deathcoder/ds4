from pathlib import Path
import subprocess,json,time,hashlib,os,re,datetime,sys
w=Path(__file__).resolve().parent;root=w.parents[2]
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
model='$MODEL_DIRECTORY/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';exe=w/'candidate-rust-star'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
while not (w/'candidate-build.json').exists():time.sleep(10)
build=read(w/'candidate-build.json');assert sha(exe)==build['executable_sha256']
for n,h in build['sources'].items():assert sha(root/n)==h,n
assert '317 passed; 0 failed' in (w/'tests-final.log').read_text()
ref=root/'rust-star/reports/128k-c0-0eb484ea/reference-131072';manifest=read(ref/'manifest.json')
for n,info in manifest['artifacts'].items():assert sha(ref/n)==info['sha256']
fixture=root/'rust-star/fixtures/measurement-131072-v5'
for n in ['prompt.u32le.bin','selected.u32le.bin']:assert (fixture/n).read_bytes()==(ref/n).read_bytes()
i=read(root/'rust-star/reports/128k-c0-0eb484ea/inputs.json');o=root/'rust-star/.work/oracle-v5-source'
assert sha(Path(model))==i['model_sha256'];assert sha(root/'speed-bench/promessi_sposi.txt')==i['prompt_sha256'];assert sha(o/'ds4-bench')==i['oracle_executable_sha256'];assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip()==i['oracle_source'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=o,text=True).strip()
(w/'preflight.json').write_text(json.dumps(dict(passed=True,inputs=i,reference_manifest_sha256=sha(ref/'manifest.json'),candidate=build['commit'],verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
checks=w/'checks';checks.mkdir();(checks/'audit-components.py').write_bytes((w/'audit-components.py').read_bytes());records=[]
queue=[('c0-128k',['long-prefill-reference-probe',model,str(ref/'prompt.u32le.bin'),str(ref/'selected.u32le.bin'),str(ref/'logits.f32le.bin')]),('retained-128k',['engine-retained-measure',model,'--context','131072','--gen-tokens','128']),('batch-32k-output',['long-prefill-chunk-probe',model,'--context','32768','--output-only']),('batch-2k',['prefill-layers012-attention-loop-probe',model]),('retained-32k',['engine-retained-measure',model,'--context','32768','--gen-tokens','128']),('batch-32k-profiled',['long-prefill-chunk-probe',model,'--context','32768','--output-only','--profile-json',str(checks/'components.json')])]
for name,args in queue:
 output=checks/(name+'.json');cmd=['/usr/bin/time','-l',str(exe),*args,*([str(output)] if name=='c0-128k' else ['--json',str(output)])];print('START',name,flush=True);start=time.time()
 with (checks/(name+'.log')).open('x') as log:p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 log=(checks/(name+'.log')).read_text();rss=re.search(r'(\d+)\s+maximum resident set size',log)
 records.append(dict(name=name,exit=p.returncode,seconds=time.time()-start,peak_rss_bytes=int(rss[1]) if rss else None,command=cmd,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));(w/'validation-progress.json').write_text(json.dumps(records,indent=2)+'\n')
 assert p.returncode==0,(name,p.returncode)
 x=read(output)
 if name=='c0-128k':
  prior=read(root/'rust-star/reports/128k-c0-0eb484ea/checks/native-131072-a.json');assert x==prior
  assert log.count('all 129280 logits exact, token ')==128
 elif name.startswith('retained-'):assert x['selection']['both_oracle_transcripts_match'] and not x['paired_protocol_eligible']
 elif name.startswith('batch-32k'):assert x['complete_32k_c0_claim'] and x['checked_decode_steps']==128 and x['logits_checksum']==2996662110144943749 and x['decode_logits_checksum']==1716816788919264169
 else:assert x['output_head']['full_logits_c0_bitwise_match']
 print('PASS',name,flush=True)
subprocess.run([sys.executable,str(checks/'audit-components.py')],check=True,stdout=subprocess.DEVNULL)
(w/'validation-acceptance.json').write_text(json.dumps(dict(passed=True,checks=records,candidate=build['commit'],full_128k_c0=True,retained_128k_transcripts=True,performance_claim=False),indent=2)+'\n');print('ALL GATES PASS',flush=True)
