from pathlib import Path
import os,subprocess,time,json,datetime
w=Path(__file__).resolve().parent;env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};exe=w/'batch-rust-star';model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
queue=[('batch-12k',['long-prefill-chunk-probe',model,'--context','12288']),('batch-8k',['long-prefill-continuation-bootstrap-probe',model]),('batch-2k',['prefill-layers012-attention-loop-probe',model]),('batch-32k-output',['long-prefill-chunk-probe',model,'--context','32768','--output-only']),('batch-32k-internal',['long-prefill-chunk-probe',model,'--context','32768']),('batch-retained',['engine-retained-measure',model,'--context','32768','--gen-tokens','128'])]
queue.append(('batch-32k-profiled',['long-prefill-chunk-probe',model,'--context','32768','--output-only','--profile-json',str(w/'components.json')]))
records=[];assert not (w/'batch-validation.json').exists(),'Never overwrite validation attempts'
for name,args in queue:
 print('START',name,flush=True);start=time.time()
 with (w/(name+'.log')).open('w') as log:r=subprocess.run([str(exe),*args,'--json',str(w/(name+'.json'))],env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append({'name':name,'exit':r.returncode,'seconds':time.time()-start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});(w/'batch-validation.json').write_text(json.dumps(records,indent=2)+'\n');print('END',name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 data=json.loads((w/(name+'.json')).read_text())
 if name.startswith('batch-32k'):assert data['complete_32k_c0_claim'] and data['checked_decode_steps']==128
 if name=='batch-retained':assert data['selection']['both_oracle_transcripts_match'] and not data['paired_protocol_eligible']
print('ALL batch correctness and lifecycle gates passed',flush=True)
