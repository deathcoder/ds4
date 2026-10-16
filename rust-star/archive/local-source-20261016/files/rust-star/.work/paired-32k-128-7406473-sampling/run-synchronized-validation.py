from pathlib import Path
import os,subprocess,time,json,datetime
w=Path(__file__).resolve().parent;env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
# Wait for the already running unsynchronized localization process.
while True:
 try:os.kill(78734,0)
 except ProcessLookupError:break
 time.sleep(5)
exe=w/'compressor-wait-rust-star';model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';records=[]
queue=[(f'synchronized-{rep}', ['long-prefill-chunk-probe',model,'--context','32768']+(['--output-only'] if rep in ['a','b'] else [])) for rep in ['a','b','c','d']]
queue += [('synchronized-2k',['prefill-layers012-attention-loop-probe',model]),('synchronized-8k',['long-prefill-continuation-bootstrap-probe',model])]
for name,args in queue:
 print('START',name,flush=True);start=time.time()
 with (w/(name+'.log')).open('w') as log:r=subprocess.run([str(exe),*args,'--json',str(w/(name+'.json'))],env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append({'name':name,'exit':r.returncode,'seconds':time.time()-start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});(w/'synchronized-validation.json').write_text(json.dumps(records,indent=2)+'\n');print('END',name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 if name in ['synchronized-a','synchronized-b','synchronized-c','synchronized-d']:assert json.loads((w/(name+'.json')).read_text())['complete_32k_c0_claim'] is True
print('ALL synchronized correctness gates passed',flush=True)
