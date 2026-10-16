from pathlib import Path
import subprocess,json,time,hashlib,os,re,datetime,sys,struct
w=Path(__file__).resolve().parent;root=w.parents[2];checks=w/'checks'
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';exe=w/'candidate-rust-star'
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fnv(raw):
 h=0xcbf29ce484222325
 for (v,) in struct.iter_unpack('<I',raw):h=((h^v)*0x100000001b3)&0xffffffffffffffff
 return h
assert read(w/'capture-acceptance.json')['passed']
build=read(w/'candidate-build.json');assert sha(exe)==build['executable_sha256']
for n,h in build['sources'].items():assert sha(root/n)==h,n
assert '318 passed; 0 failed' in (w/'tests.log').read_text()
assert read(w/'preflight.json')['passed']
refs={};reference_args={}
for n in [131072,196608,262144]:
 d=root/'rust-star/reports/128k-c0-0eb484ea/reference-131072' if n==131072 else w/f'reference-{n}'
 m=read(d/'manifest.json');assert m['repeat_processes']==2 and m['frontier']==n
 for name,info in m['artifacts'].items():assert sha(d/name)==info['sha256'] and (d/name).stat().st_size==info['bytes']
 logits=(d/'logits.f32le.bin').read_bytes();tokens=(d/'selected.u32le.bin').read_bytes();steps=0 if n==196608 else 128
 refs[n]={'frontier_tokens':n,'checked_logits':129280,'checked_decode_steps':steps,'prefill_logits_checksum':fnv(logits[:129280*4]),'decode_logits_checksum':fnv(logits[129280*4:]) if steps else 0,'decode_tokens_checksum':fnv(tokens[4:]) if steps else 0,'selected_token':struct.unpack('<I',tokens[:4])[0],'manifest_sha256':sha(d/'manifest.json')}
 reference_args[n]=[model,str(d/'prompt.u32le.bin'),str(d/'selected.u32le.bin'),str(d/'logits.f32le.bin')]
queue=[
 ('batch-2k',['prefill-layers012-attention-loop-probe',model],None,False),
 ('batch-8k',['long-prefill-continuation-bootstrap-probe',model],None,False),
 ('batch-12k',['long-prefill-chunk-probe',model,'--context','12288'],None,False),
 ('batch-32k-output',['long-prefill-chunk-probe',model,'--context','32768','--output-only'],None,False),
 ('batch-32k-internal',['long-prefill-chunk-probe',model,'--context','32768'],None,False),
 ('batch-32k-profiled',['long-prefill-chunk-probe',model,'--context','32768','--output-only','--profile-json',str(checks/'components.json')],None,False),
 ('c0-128k',['long-prefill-reference-probe',*reference_args[131072]],131072,False),
 ('c0-192k',['long-prefill-reference-probe',*reference_args[196608]],196608,False),
 ('c0-256k',['long-prefill-reference-probe',*reference_args[262144]],262144,False),
 ('retained-128k',['long-prefill-reference-probe',*reference_args[131072]],131072,True),
 ('retained-256k',['long-prefill-reference-probe',*reference_args[262144]],262144,True),
]
def check_reference(x,n):
 assert x['external_reference_match'] and not x['oracle_provenance_verified_by_runtime'] and not x['throughput_claim']
 for key,value in refs[n].items():
  if key!='manifest_sha256':assert x[key]==value,(n,key,x[key],value)
records=[];assert not (w/'validation-progress.json').exists()
for name,args,n,retained in queue:
 output=checks/(name+'.json');cmd=['/usr/bin/time','-l',str(exe),*args,*([str(output)] if n else ['--json',str(output)])]
 if retained:cmd+=['--retained']
 print('START',name,flush=True);start=time.time()
 with (checks/(name+'.log')).open('x') as log:p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 log=(checks/(name+'.log')).read_text();rss=re.search(r'(\d+)\s+maximum resident set size',log)
 records.append(dict(name=name,exit=p.returncode,seconds=time.time()-start,peak_rss_bytes=int(rss[1]) if rss else None,command=cmd,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));(w/'validation-progress.json').write_text(json.dumps(records,indent=2)+'\n')
 assert p.returncode==0,(name,p.returncode);assert rss
 x=read(output)
 if retained:
  assert x['schema']=='rust-star-retained-frontier-check-v1' and x['cycles']==2 and len(x['checks'])==2 and x['frontier_tokens']==n
  for key in ['same_process','same_model_mapping','same_metal_context','request_state_reset_between_cycles','model_warm_reused_on_retained_cycle','retained_model_residency_reused']:assert x[key] is True,key
  assert x['first_model_residency_reused'] is False and x['paired_protocol_eligible'] is False and x['throughput_claim'] is False
  for cycle in x['checks']:check_reference(cycle,n)
  assert x['checks'][0]==x['checks'][1]
  assert log.count('all 129280 logits exact, token ')==256
 elif n:
  check_reference(x,n);assert log.count('all 129280 logits exact, token ')==refs[n]['checked_decode_steps']
 elif name.startswith('batch-32k'):
  assert x['complete_32k_c0_claim'] and x['checked_decode_steps']==128 and x['logits_checksum']==2996662110144943749 and x['decode_logits_checksum']==1716816788919264169
  assert x['decode_tokens_checksum']==11977321378026793466 and x['selected_token']==85166
  assert log.count('all 129280 logits exact, token ')==128
  if name=='batch-32k-internal':
   prior=read(root/'rust-star/reports/32k-memory-de6d2c11/correctness/batch-32k-internal.json')
   for actual,expected in zip(x['chunks'],prior['chunks']):
    if actual['end']<=24576:assert actual==expected
 elif name=='batch-12k':assert x['frontier_tokens']==12288 and x['internal_hc_collection'] and x['checked_decode_steps']==0 and x['chunks']==read(root/'rust-star/reports/32k-memory-de6d2c11/correctness/batch-12k.json')['chunks']
 elif name=='batch-8k':
  assert x['sparse_transition']['score_kernel']=='rust_star_indexer_scores_tiled64'
  assert x['layer4_prefix_complete']['sparse_attention']['kernels'][0]=='rust_star_indexer_scores_tiled64'
  assert x['complete_8k_transformer_claim'] and x['output_logits_c0_claim'] and x['second_consecutive_tile']['selector_fixture']=='dwarfstar-oracle-v5-prefill-indexer-selector-4128'
 else:assert x['output_head']['full_logits_c0_bitwise_match']
 if name=='batch-32k-profiled':subprocess.run([sys.executable,str(checks/'audit-components.py')],check=True,stdout=subprocess.DEVNULL)
 print('PASS',name,'peak GiB',int(rss[1])/2**30,flush=True)
(w/'validation-acceptance.json').write_text(json.dumps(dict(passed=True,checks=records,candidate=build['commit'],references=refs,full_256k_c0=True,retained_full_c0_contexts=[131072,262144],performance_claim=False),indent=2)+'\n')
print('ALL ELEVEN GATES PASS',flush=True)
