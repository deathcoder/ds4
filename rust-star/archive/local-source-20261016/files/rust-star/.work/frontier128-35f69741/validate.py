from pathlib import Path
import subprocess,json,time,hashlib,os,struct,re,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';exe=w/'candidate-rust-star'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fnv(b):
 h=0xcbf29ce484222325
 for (v,) in struct.iter_unpack('<I',b):h=((h^v)*0x100000001b3)&0xffffffffffffffff
 return h
while not (w/'capture-acceptance.json').exists() or not (w/'candidate-build.json').exists():time.sleep(45)
assert read(w/'capture-acceptance.json')['passed'];build=read(w/'candidate-build.json');assert sha(exe)==build['executable_sha256']
for n,h in build['sources'].items():assert sha(root/n)==h
assert '314 passed; 0 failed' in (w/'tests-final.log').read_text()
checks=w/'checks';checks.mkdir();refs={}
for n in [65536,98304,131072]:
 d=w/f'reference-{n}';m=read(d/'manifest.json');assert m['repeat_processes']==2 and m['frontier']==n
 for name,info in m['artifacts'].items():assert sha(d/name)==info['sha256'] and (d/name).stat().st_size==info['bytes']
 refs[n]=dict(prefill=fnv((d/'logits.f32le.bin').read_bytes()[:129280*4]),decode=fnv((d/'logits.f32le.bin').read_bytes()[129280*4:]) if n==131072 else 0,tokens=fnv((d/'selected.u32le.bin').read_bytes()[4:]) if n==131072 else 0,selected=struct.unpack('<I',(d/'selected.u32le.bin').read_bytes()[:4])[0],manifest_sha256=sha(d/'manifest.json'))
queue=[('batch-32k-output',['long-prefill-chunk-probe',model,'--context','32768','--output-only'],None)]
for n,rep in [(65536,'a'),(98304,'a'),(131072,'a'),(131072,'b')]:
 d=w/f'reference-{n}';name=f'native-{n}-{rep}';queue.append((name,['long-prefill-reference-probe',model,str(d/'prompt.u32le.bin'),str(d/'selected.u32le.bin'),str(d/'logits.f32le.bin')],n))
queue.extend([('batch-12k',['long-prefill-chunk-probe',model,'--context','12288'],None),('batch-8k',['long-prefill-continuation-bootstrap-probe',model],None),('batch-2k',['prefill-layers012-attention-loop-probe',model],None),('batch-32k-internal',['long-prefill-chunk-probe',model,'--context','32768'],None),('batch-retained',['engine-retained-measure',model,'--context','32768','--gen-tokens','128'],None),('batch-32k-profiled',['long-prefill-chunk-probe',model,'--context','32768','--output-only','--profile-json',str(checks/'components.json')],None)])
records=[]
for name,args,n in queue:
 output=checks/(name+'.json');command=['/usr/bin/time','-l',str(exe),*args,*([str(output)] if n else ['--json',str(output)])];print('START',name,flush=True);start=time.time()
 with (checks/(name+'.log')).open('x') as log:p=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 text=(checks/(name+'.log')).read_text();rss=re.search(r'(\d+)\s+maximum resident set size',text)
 record=dict(name=name,exit=p.returncode,seconds=time.time()-start,peak_rss_bytes=int(rss[1]) if rss else None,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command);records.append(record);(w/'validation-progress.json').write_text(json.dumps(records,indent=2)+'\n')
 if p.returncode:raise RuntimeError(record)
 assert rss,'Missing process memory measurement'
 x=read(output)
 if n:
  assert x['frontier_tokens']==n and x['checked_logits']==129280 and x['external_reference_match'] and not x['oracle_provenance_verified_by_runtime'] and not x['throughput_claim']
  assert x['checked_decode_steps']==(128 if n==131072 else 0)
  for k,t in [('prefill_logits_checksum','prefill'),('decode_logits_checksum','decode'),('decode_tokens_checksum','tokens'),('selected_token','selected')]:assert x[k]==refs[n][t],(name,k)
  if n==131072:assert text.count('all 129280 logits exact, token ')==128
 elif name.startswith('batch-32k'):assert x['complete_32k_c0_claim'] and x['checked_decode_steps']==128 and x['logits_checksum']==2996662110144943749 and x['decode_logits_checksum']==1716816788919264169
 elif name=='batch-retained':assert x['selection']['both_oracle_transcripts_match'] and not x['paired_protocol_eligible']
 print('PASS',name,'peak GiB',int(rss[1])/2**30,flush=True)
(checks/'batch-validation.json').write_text(json.dumps([x for x in records if x['name'].startswith('batch-')],indent=2)+'\n')
(w/'validation-acceptance.json').write_text(json.dumps(dict(passed=True,checks=records,references=refs,full_128k_native_runs=2,source_commit=build['commit'],oracle_unchanged=True,performance_claim=False),indent=2)+'\n');print('ALL NATIVE CHECKS PASS',flush=True)
