"""Portable audit: pinned repeated oracle vectors and every native acceptance gate."""
from pathlib import Path
import json,hashlib,struct,math,subprocess,sys,re
r=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def fnv(b):
 h=0xcbf29ce484222325
 for (v,) in struct.iter_unpack('<I',b):h=((h^v)*0x100000001b3)&0xffffffffffffffff
 return h
m=read(r/'artifact-hashes.json')['artifacts'];actual={str(p.relative_to(r)) for p in r.rglob('*') if p.is_file() and p!=r/'artifact-hashes.json' and '__pycache__' not in p.parts};assert actual==set(m)
for name,x in m.items():assert sha(r/name)==x['sha256'] and (r/name).stat().st_size==x['bytes'],name
inputs=read(r/'inputs.json');assert inputs['oracle_source']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4' and inputs['oracle_executable_sha256']=='a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0'
assert inputs['model_sha256']=='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0'
build=read(r/'candidate-build.json')
for name,h in build['sources'].items():assert sha(r/'candidate-source'/name)==h
assert '314 passed; 0 failed' in (r/'tests-final.log').read_text()
assert read(r/'capture-acceptance.json')['passed'];assert len(read(r/'capture-progress.json'))==6
assert all(x['exit']==0 for x in read(r/'capture-progress.json'))
refs={}
for n in [65536,98304,131072]:
 d=r/f'reference-{n}';manifest=read(d/'manifest.json');assert manifest['frontier']==n and manifest['repeat_processes']==2
 assert manifest['oracle_source']==inputs['oracle_source'] and manifest['model_sha256']==inputs['model_sha256'] and manifest['prompt_text_sha256']==inputs['prompt_sha256']
 for name,x in manifest['artifacts'].items():assert sha(d/name)==x['sha256'] and (d/name).stat().st_size==x['bytes']
 assert (d/'prompt.u32le.bin').stat().st_size==n*4
 assert hashlib.sha256((d/'prompt.u32le.bin').read_bytes()[:32768*4]).hexdigest()=='874c479e2cde7b231dfa4a963d5f1fbe9dd9e60c2842f869e2a117d4860902a0'
 if n>65536:assert (d/'prompt.u32le.bin').read_bytes()[:65536*4]==(r/'reference-65536/prompt.u32le.bin').read_bytes()
 raw=(d/'logits.f32le.bin').read_bytes();selected=list(struct.unpack(f'<{len(raw)//(129280*4)}I',(d/'selected.u32le.bin').read_bytes()));assert len(raw)==(129 if n==131072 else 1)*129280*4
 hashes=[]
 for i,token in enumerate(selected):
  row=raw[i*129280*4:(i+1)*129280*4];values=struct.unpack('<129280f',row);assert all(map(math.isfinite,values));assert max(range(129280),key=values.__getitem__)==token!=inputs['eos_token'];hashes.append(hashlib.sha256(row).hexdigest())
 for rep in ['a','b']:
  o=r/f'oracle-{n}-{rep}';v=read(o/'verification.json');assert v['row_sha256']==hashes and v['selected_tokens']==selected and v['all_logits_sha256']==sha(d/'logits.f32le.bin')
  assert (o/'all-logits.f32le.bin').read_bytes()==raw and (o/'selected.u32le.bin').read_bytes()==(d/'selected.u32le.bin').read_bytes()
 refs[n]=(fnv(raw[:129280*4]),fnv(raw[129280*4:]) if n==131072 else 0,fnv((d/'selected.u32le.bin').read_bytes()[4:]) if n==131072 else 0,selected[0])
a=read(r/'validation-acceptance.json');assert a['passed'] and a['source_commit']==build['commit'] and a['full_128k_native_runs']==2 and not a['performance_claim']
assert len(a['checks'])==11 and all(x['exit']==0 and x['peak_rss_bytes']>0 for x in a['checks'])
for n,rep in [(65536,'a'),(98304,'a'),(131072,'a'),(131072,'b')]:
 name=f'native-{n}-{rep}';x=read(r/'checks'/(name+'.json'));assert x['external_reference_match'] and not x['oracle_provenance_verified_by_runtime'] and not x['throughput_claim'];assert x['frontier_tokens']==n and x['checked_logits']==129280 and x['checked_decode_steps']==(128 if n==131072 else 0)
 assert tuple(x[k] for k in ['prefill_logits_checksum','decode_logits_checksum','decode_tokens_checksum','selected_token'])==refs[n]
 log=(r/'checks'/(name+'.log')).read_text()
 assert [int(x) for x in re.findall(r'output-only gate completed native prefill through (\d+) tokens',log)]==list(range(8192,n+1,4096))
 if n==131072:assert log.count('all 129280 logits exact, token ')==128
for name in ['batch-32k-output','batch-32k-internal','batch-32k-profiled']:
 x=read(r/'checks'/(name+'.json'));assert x['complete_32k_c0_claim'] and x['checked_logits']==129280 and x['checked_decode_steps']==128
 assert (x['logits_checksum'],x['decode_logits_checksum'],x['decode_tokens_checksum'])==(2996662110144943749,1716816788919264169,11977321378026793466)
assert read(r/'checks/batch-retained.json')['selection']['both_oracle_transcripts_match']
assert read(r/'checks/batch-2k.json')['output_head']['full_logits_c0_bitwise_match']
assert read(r/'checks/batch-8k.json')['output_logits_c0_claim']
subprocess.run([sys.executable,str(r/'checks/audit-components.py')],check=True,stdout=subprocess.DEVNULL)
assert sha(r/'checks/component-audit.json')==m['checks/component-audit.json']['sha256']
assert read(r/'final-model-verification.json')['model_sha256']==inputs['model_sha256']
print(f'PASS: {len(actual)} artifacts; repeated 64K/96K/128K oracle, two complete native 128K runs, shorter C0 controls')
