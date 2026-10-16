from pathlib import Path
import hashlib,json,os,subprocess,shutil
root=Path.cwd();w=root/'rust-star/.work/native-32k-v31';src=root/'rust-star/.work/oracle-v3-8k-source';exe=src/'ds4-bench.accepted-oracle-v3';prompt=root/'speed-bench/promessi_sposi.txt';model=src/'model.gguf'
assert hashlib.sha256(exe.read_bytes()).hexdigest()=='8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3'
assert hashlib.sha256(prompt.read_bytes()).hexdigest()=='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for end in range(12288,32769,4096):
 prior=None
 for repeat in ['a','b']:
  d=w/f'oracle-{end}-{repeat}';d.mkdir();(d/'logits').mkdir();assert shutil.disk_usage(root).free>25*2**30
  env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_'))};env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(d/'oracle'),DS4_METAL_GRAPH_DUMP_POS=str(end-4096),DS4_METAL_GRAPH_DUMP_NAME='hc_ffn_post')
  cmd=[str(exe),'--metal','--model',str(model),'--prompt-file',str(prompt),'--ctx-start',str(end),'--ctx-max',str(end),'--gen-tokens','0','--prefill-chunk','4096','--dump-frontier-logits-dir',str(d/'logits')]
  print('capture',end,repeat,flush=True)
  with (d/'console.log').open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  hashes={}
  for p in sorted(d.glob('*.bin')):
   hashes[p.name]={'bytes':p.stat().st_size,'sha256':sha(p),'fnv1a_u32':subprocess.check_output([str(root/'rust-star/.work/layer13-continuation-v28/hash_words'),str(p)],text=True).split()[0]}
  assert len(hashes)==43,len(hashes)
  (d/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
  if prior is not None:assert prior==hashes,'independent capture mismatch'
  prior=hashes
  for p in d.glob('*.bin'):p.unlink()
  (d/'raw-compacted.json').write_text(json.dumps({'reason':'temporary diagnostic payloads hashed; preserve both independent hash inventories and logs','files':list(hashes)},indent=2)+'\n')
 print('verified all 43 layers at',end,flush=True)
