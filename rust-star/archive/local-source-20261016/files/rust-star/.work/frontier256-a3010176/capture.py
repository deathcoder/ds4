from pathlib import Path
import subprocess,json,hashlib,struct,math,os,time,datetime,resource,re
w=Path(__file__).resolve().parent;root=w.parents[2];src=w.with_name('oracle-v5-source');exe=src/'ds4-bench';model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');prompt=root/'speed-bench/promessi_sposi.txt';env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src,text=True).strip()
assert sha(exe)=='a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0'
assert sha(model)=='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0'
assert sha(prompt)=='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f'
p=subprocess.run([str(src/'ds4'),'-m',str(model),'--raw-prompt','--prompt-file',str(prompt),'--dump-tokens'],cwd=src,env=env,capture_output=True,check=True)
tokens=json.loads(p.stdout.splitlines()[0]);assert len(tokens)>=262144
assert struct.pack('<32768I',*tokens[:32768])==(root/'rust-star/fixtures/prefill-frontier-32768-v5/token-ids.u32le.bin').read_bytes()
eos=json.loads((w.with_name('upstream-refresh-1190237b')/'eos-token-v5.json').read_text())['eos_id']
write(w/'inputs.json',dict(oracle_source='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4',oracle_executable_sha256=sha(exe),cli_sha256=sha(src/'ds4'),model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(prompt),available_prompt_tokens=len(tokens),eos_token=eos,captures=[[n,rep,128 if n==262144 else 0] for n in [196608,262144] for rep in ['a','b']],full_256k_claim_requires_two_identical_129_row_traces=True))
records=[]
for n in [196608,262144]:
 target=w/f'reference-{n}';target.mkdir();(target/'prompt.u32le.bin').write_bytes(struct.pack(f'<{n}I',*tokens[:n]));gen=128 if n==262144 else 0
 for rep in ['a','b']:
  d=w/f'oracle-{n}-{rep}';d.mkdir();dump=d/'logits';dump.mkdir();print('START',n,rep,flush=True)
  command=[str(exe),'--metal','-m',str(model),'--prompt-file',str(prompt),'--ctx-start',str(n),'--ctx-max',str(n),'--gen-tokens',str(gen),'--warm-weights','--dump-frontier-logits-dir',str(dump),'--csv',str(d/'benchmark.csv')]
  start=time.time()
  with (d/'console.log').open('x') as log:run=subprocess.run(['/usr/bin/time','-l',*command],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT)
  peak=re.search(r'(\d+)\s+maximum resident set size', (d/'console.log').read_text());assert peak
  rec=dict(process_peak_bytes=int(peak[1]),context=n,repetition=rep,exit=run.returncode,seconds=time.time()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command);records.append(rec);write(w/'capture-progress.json',records)
  if run.returncode:raise RuntimeError(rec)
  selected=[];hashes=[]
  with (d/'all-logits.f32le.bin').open('xb') as out:
   for step in range(gen+1):
    j=json.loads((dump/f'frontier_{n+step:06d}.logits.json').read_text());raw=struct.pack('<129280f',*j['logits']);values=struct.unpack('<129280f',raw);assert all(map(math.isfinite,values));token=max(range(129280),key=values.__getitem__);assert token==j['argmax_id'] and token!=eos;selected.append(token);hashes.append(hashlib.sha256(raw).hexdigest());out.write(raw)
  (d/'selected.u32le.bin').write_bytes(struct.pack(f'<{len(selected)}I',*selected));write(d/'verification.json',dict(rows=gen+1,words=(gen+1)*129280,row_sha256=hashes,selected_tokens=selected,all_logits_sha256=sha(d/'all-logits.f32le.bin'),paired_eligible=False))
  if rep=='a':
   (target/'logits.f32le.bin').write_bytes((d/'all-logits.f32le.bin').read_bytes());(target/'selected.u32le.bin').write_bytes((d/'selected.u32le.bin').read_bytes())
  else:
   assert (target/'logits.f32le.bin').read_bytes()==(d/'all-logits.f32le.bin').read_bytes(),'oracle logits not repeatable'
   assert (target/'selected.u32le.bin').read_bytes()==(d/'selected.u32le.bin').read_bytes()
  print('PASS',n,rep,flush=True)
 write(target/'manifest.json',dict(schema='rust-star-external-frontier-reference-v1',frontier=n,decode_steps=gen,vocabulary=129280,repeat_processes=2,oracle_source='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4',model_sha256='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0',prompt_text_sha256=sha(prompt),artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in target.iterdir() if p.is_file()}))
write(w/'capture-acceptance.json',dict(passed=True,frontiers=[196608,262144],fresh_processes=4,full_256k_rows=129,oracle_unchanged=True));print('ALL CAPTURES PASS',flush=True)
