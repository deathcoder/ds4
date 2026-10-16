from pathlib import Path
import subprocess,json,time,hashlib,os,re,datetime,sys
w=Path(__file__).resolve().parent;root=w.parents[2];checks=w/'checks'
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';exe=w/'candidate-rust-star'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
build=read(w/'candidate-build.json');assert sha(exe)==build['executable_sha256']
for n,h in build['sources'].items():assert sha(root/n)==h,n
assert '317 passed; 0 failed' in (w/'tests.log').read_text()
assert read(w/'preflight.json')['passed']
ref=root/'rust-star/reports/128k-c0-0eb484ea/reference-131072';manifest=read(ref/'manifest.json')
for n,info in manifest['artifacts'].items():assert sha(ref/n)==info['sha256']
prior_c0=read(root/'rust-star/reports/128k-c0-0eb484ea/checks/native-131072-a.json')
reference_args=[model,str(ref/'prompt.u32le.bin'),str(ref/'selected.u32le.bin'),str(ref/'logits.f32le.bin')]
queue=[
 ('batch-2k',['prefill-layers012-attention-loop-probe',model]),
 ('batch-8k',['long-prefill-continuation-bootstrap-probe',model]),
 ('batch-12k',['long-prefill-chunk-probe',model,'--context','12288']),
 ('batch-32k-output',['long-prefill-chunk-probe',model,'--context','32768','--output-only']),
 ('batch-32k-internal',['long-prefill-chunk-probe',model,'--context','32768']),
 ('batch-32k-profiled',['long-prefill-chunk-probe',model,'--context','32768','--output-only','--profile-json',str(checks/'components.json')]),
 ('retained-32k',['engine-retained-measure',model,'--context','32768','--gen-tokens','128']),
 ('c0-128k',['long-prefill-reference-probe',*reference_args]),
 ('c0-128k-profiled',['long-prefill-reference-probe',*reference_args]),
 ('retained-128k',['engine-retained-measure',model,'--context','131072','--gen-tokens','128']),
]
records=[];assert not (w/'validation-progress.json').exists()
for name,args in queue:
 output=checks/(name+'.json');cmd=['/usr/bin/time','-l',str(exe),*args,*([str(output)] if name.startswith('c0-') else ['--json',str(output)])]
 if name=='c0-128k-profiled':cmd+=['--profile-json',str(checks/'components-128k.json')]
 print('START',name,flush=True);start=time.time()
 with (checks/(name+'.log')).open('x') as log:p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 log=(checks/(name+'.log')).read_text();rss=re.search(r'(\d+)\s+maximum resident set size',log)
 records.append(dict(name=name,exit=p.returncode,seconds=time.time()-start,peak_rss_bytes=int(rss[1]) if rss else None,command=cmd,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));(w/'validation-progress.json').write_text(json.dumps(records,indent=2)+'\n')
 assert p.returncode==0,(name,p.returncode)
 x=read(output)
 if name.startswith('c0-'):
  assert x==prior_c0
  assert log.count('all 129280 logits exact, token ')==128
 elif name.startswith('retained-'):assert x['selection']['both_oracle_transcripts_match'] and not x['paired_protocol_eligible']
 elif name.startswith('batch-32k'):
  assert x['complete_32k_c0_claim'] and x['checked_decode_steps']==128 and x['logits_checksum']==2996662110144943749 and x['decode_logits_checksum']==1716816788919264169
  assert x['decode_tokens_checksum']==11977321378026793466 and x['selected_token']==85166
  assert log.count('all 129280 logits exact, token ')==128
 elif name=='batch-12k':assert x['frontier_tokens']==12288 and x['internal_hc_collection'] and x['checked_decode_steps']==0 and x['chunks']==read(root/'rust-star/reports/32k-memory-de6d2c11/correctness/batch-12k.json')['chunks']
 elif name=='batch-8k':assert x['complete_8k_transformer_claim'] and x['output_logits_c0_claim'] and x['second_consecutive_tile']['selector_fixture']=='dwarfstar-oracle-v5-prefill-indexer-selector-4128'
 else:assert x['output_head']['full_logits_c0_bitwise_match']
 if name=='batch-32k-profiled':subprocess.run([sys.executable,str(checks/'audit-components.py')],check=True,stdout=subprocess.DEVNULL)
 if name=='c0-128k-profiled':subprocess.run([sys.executable,str(checks/'audit-profile128.py'),str(checks/'components-128k.json'),str(output)],check=True,stdout=subprocess.DEVNULL)
 print('PASS',name,flush=True)
(w/'validation-acceptance.json').write_text(json.dumps(dict(passed=True,checks=records,candidate=build['commit'],full_128k_c0=True,profiled_128k_c0=True,retained_128k_transcripts=True,performance_claim=False),indent=2)+'\n')
print('ALL TEN GATES PASS',flush=True)
