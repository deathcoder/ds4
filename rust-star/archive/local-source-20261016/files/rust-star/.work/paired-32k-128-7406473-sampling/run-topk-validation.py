from pathlib import Path
import os,subprocess,time,json,datetime,struct,hashlib
w=Path(__file__).resolve().parent;root=w.parents[2];env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
manifest=w/'oracle-v4-acceptance/manifest.json'
while True:
 if manifest.exists():
  m=json.loads(manifest.read_text())
  if m['status']=='complete':break
  if m['status'] in ['failed','interrupted']:raise SystemExit('Oracle v4 acceptance failed; stop')
 time.sleep(10)
subprocess.run(['python3',str(root/'rust-star/verify_oracle_bundle.py'),str(manifest.parent),'--json',str(w/'oracle-v4-verification.json')],check=True)
expected={2048:root/'rust-star/fixtures/prefill-frontier-2048-v1/batch-prefill-logits.f32le.bin',32768:root/'rust-star/fixtures/prefill-frontier-32768-v3/batch-prefill-logits.f32le.bin'}
fixture=root/'rust-star/fixtures/prefill-frontier-32768-v3';fm=json.loads((fixture/'manifest.json').read_text());expected[32768]=next(fixture/t['path'] for t in fm['tensors'] if t.get('shape')==[129280])
checks=[]
for r in m['conformance']['runs']:
 p=manifest.parent/r['logits']['path'];values=json.loads(p.read_text())['logits'];actual=struct.pack('<129280f',*values);assert actual==expected[r['context']].read_bytes(),('immutable target mismatch',r['context'],r['repetition']);checks.append({'context':r['context'],'repetition':r['repetition'],'exact':True,'fp32_sha256':hashlib.sha256(actual).hexdigest()})
(w/'oracle-v4-target-equivalence.json').write_text(json.dumps({'passed':True,'runs':checks,'immutable_targets_changed':False},indent=2)+'\n');print('PASS v4 repeatability and unchanged 2K/32K target equivalence',flush=True)
with (w/'conformance-v4.log').open('w') as log:r=subprocess.run(['python3','-u',str(w/'conformance-v4.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
if r.returncode:raise SystemExit(r.returncode)
print('PASS benchmark oracle v4 all 129 full-logit rows twice',flush=True)
exe=w/'topk-barrier-rust-star';model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';records=[]
queue=[(f'topk-{rep}', ['long-prefill-chunk-probe',model,'--context','32768']+(['--output-only'] if rep in ['a','b'] else [])) for rep in ['a','b','c','d']]
queue += [('topk-2k',['prefill-layers012-attention-loop-probe',model]),('topk-8k',['long-prefill-continuation-bootstrap-probe',model])]
for name,args in queue:
 print('START',name,flush=True);start=time.time()
 with (w/(name+'.log')).open('w') as log:r=subprocess.run([str(exe),*args,'--json',str(w/(name+'.json'))],env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append({'name':name,'exit':r.returncode,'seconds':time.time()-start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});(w/'topk-validation.json').write_text(json.dumps(records,indent=2)+'\n');print('END',name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 if name in ['topk-a','topk-b','topk-c','topk-d']:assert json.loads((w/(name+'.json')).read_text())['complete_32k_c0_claim'] is True
print('ALL top-k repair correctness gates passed',flush=True)
