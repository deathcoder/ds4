from pathlib import Path
import json,subprocess,os,struct,hashlib,datetime,time
w=Path(__file__).resolve().parent;root=w.parents[2];src=w.with_name('oracle-v5-source');model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env['DS4_TEST_MODEL']=model
assert json.loads((w/'pristine-repeatability.json').read_text())['passed']
records=[]
def run(name,args,cwd):
 print('START',name,flush=True);t=time.time()
 with (w/(name+'.log')).open('w') as f:p=subprocess.run(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 records.append({'name':name,'exit':p.returncode,'seconds':time.time()-t,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});(w/'control-validation.json').write_text(json.dumps(records,indent=2)+'\n');print('END',name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
run('m1-kernels-corrected',[str(src/'ds4_test'),'--metal-kernels'],src)
run('pristine-logprob-vectors',[str(src/'ds4_test'),'--logprob-vectors'],src)
for ctx in [32768,8192]:
 d=w/f'selector-control-{ctx}';d.mkdir();exe=w/'v4-selector-control/ds4-bench';run(f'selector-control-{ctx}',[str(exe),'--metal','-m',model,'--prompt-file',str(root/'speed-bench/promessi_sposi.txt'),'--ctx-start',str(ctx),'--ctx-max',str(ctx),'--gen-tokens','0','--warm-weights','--dump-frontier-logits-dir',str(d),'--csv',str(d/'benchmark.csv')],exe.parent)
 if ctx==32768:
  a=json.loads((d/f'frontier_{ctx:06d}.logits.json').read_text());b=json.loads((w/'pristine-32768-1/frontier_032768.logits.json').read_text());aa=struct.pack('<129280f',*a['logits']);bb=struct.pack('<129280f',*b['logits']);r={'matches_upstream_v5':aa==bb,'differing_words':sum(aa[i:i+4]!=bb[i:i+4] for i in range(0,len(aa),4)),'control_sha256':hashlib.sha256(aa).hexdigest(),'upstream_sha256':hashlib.sha256(bb).hexdigest(),'scope':'V4 plus only removal of streaming topk branch vs pristine upstream at32K'};(w/'selector-attribution.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
