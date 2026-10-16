from pathlib import Path
import subprocess,json,hashlib,struct,os,time,datetime,math
w=Path(__file__).resolve().parent;root=w.parents[2];src=w.with_name('oracle-v5-source');model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env['DS4_TEST_MODEL']=str(model)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
records=[]
def run(name,args):
 print('START',name,flush=True);start=time.time()
 with (w/(name+'.log')).open('w') as f:r=subprocess.run(args,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT)
 records.append({'name':name,'exit':r.returncode,'seconds':time.time()-start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});save(w/'pristine-frontier-validation.json',records)
 if r.returncode:raise SystemExit(r.returncode)
 print('PASS',name,flush=True)

hashes={};comparisons=[]
for rep in range(1,5):
 for ctx in ([2048,32768] if rep%2 else [32768,2048]):
  name=f'pristine-{ctx}-{rep}';d=w/name;d.mkdir();run(name,[str(src/'ds4-bench'),'--metal','-m',str(model),'--prompt-file',str(root/'speed-bench/promessi_sposi.txt'),'--ctx-start',str(ctx),'--ctx-max',str(ctx),'--gen-tokens','0','--warm-weights','--dump-frontier-logits-dir',str(d),'--csv',str(d/'benchmark.csv')])
  p=d/f'frontier_{ctx:06d}.logits.json';j=json.loads(p.read_text());assert len(j['logits'])==129280 and all(math.isfinite(v) for v in j['logits']);bits=struct.pack('<129280f',*j['logits']);h=hashlib.sha256(bits).hexdigest()
  old=root/f'rust-star/.work/paired-32k-128-7406473-sampling/oracle-v4-acceptance/conformance/ctx_{ctx:07d}/run_01/logits/frontier_{ctx:06d}.logits.json';o=json.loads(old.read_text());oldbits=struct.pack('<129280f',*o['logits']);row={'context':ctx,'repetition':rep,'fp32_sha256':h,'argmax_id':j['argmax_id'],'v4_bit_exact':bits==oldbits,'v4_differing_words':sum(bits[i:i+4]!=oldbits[i:i+4] for i in range(0,len(bits),4))};comparisons.append(row);save(w/'pristine-comparison.json',comparisons);assert hashes.setdefault(ctx,h)==h,('fresh process drift',row);print(row,flush=True)
save(w/'pristine-repeatability.json',{'passed':True,'repetitions':4,'contexts':hashes,'unchanged_v4_targets':all(r['v4_bit_exact'] for r in comparisons)})
