from pathlib import Path
import os,sys,subprocess,hashlib,json,shutil,struct
root=Path.cwd();work=root/'rust-star/.work/remaining-layers-v30/all-layers';work.mkdir(exist_ok=True)
source=root/'rust-star/.work/oracle-v3-8k-source';exe=source/'ds4-bench.accepted-oracle-v3';prompt=root/'speed-bench/promessi_sposi.txt';model=source/'model.gguf'
assert hashlib.sha256(exe.read_bytes()).hexdigest()=='8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3'
assert hashlib.sha256(prompt.read_bytes()).hexdigest()=='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f'
assert model.resolve()==Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf')
phase=sys.argv[1];assert phase in ['prefix','continuation'];pos=0 if phase=='prefix' else 4096
names='hc_attn_pre,attn_norm,Qcur,KVcur,KVcompress,attn_state_kv,attn_state_score,indexer_KVcompress,indexer_state_kv,indexer_state_score,kqv_back,hc_attn_post,hc_ffn_post'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for repeat in ['a','b']:
 dest=work/f'{phase}-{repeat}'
 if (dest/'hashes.json').exists():continue
 assert not dest.exists(),f'Inspect unfinished capture {dest}'
 if shutil.disk_usage(root).free<90*2**30:raise RuntimeError('Need 90 GiB available for whole-chunk capture')
 dest.mkdir();(dest/'logits').mkdir()
 env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_'))}
 env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(dest/'oracle'),DS4_METAL_GRAPH_DUMP_POS=str(pos),DS4_METAL_GRAPH_DUMP_NAME=names)
 command=[str(exe),'--metal','--model',str(model),'--prompt-file',str(prompt),'--ctx-start','8192','--ctx-max','8192','--gen-tokens','0','--prefill-chunk','4096','--dump-frontier-logits-dir',str(dest/'logits')]
 print('Capturing all layers',phase,repeat,flush=True)
 with (dest/'console.log').open('w') as log:subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 hashes={}
 for f in sorted(dest.glob('*.bin')):
  fnv=subprocess.check_output([str(root/'rust-star/.work/layer13-continuation-v28/hash_words'),str(f)],text=True).split()[0]
  hashes[f.name]={'bytes':f.stat().st_size,'sha256':sha(f),'fnv1a_u32':fnv}
 (dest/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print('Hashed',phase,repeat,len(hashes),'payloads',flush=True)
 if repeat=='a':
  # These are temporary generated captures. Persist exact hashes before freeing
  # first-run payloads; keep the second full capture for each verified pair.
  for f in dest.glob('*.bin'):f.unlink()
  (dest/'payloads-removed.json').write_text(json.dumps({'reason':'first-run hashes persisted; second full capture retained','files':list(hashes)},indent=2)+'\n')
a=json.loads((work/f'{phase}-a/hashes.json').read_text());b=json.loads((work/f'{phase}-b/hashes.json').read_text());assert a.keys()==b.keys()
diff=[name for name in a if a[name]!=b[name]]
(work/f'{phase}-comparison.json').write_text(json.dumps({'mismatches':diff,'payloads':len(a)},indent=2)+'\n')
print('Repeat mismatches',phase,len(diff),flush=True)
# Known isolated captures are independent controls for instrumentation effects.
known=[]
for layer in range(15,43):
 f=root/f'rust-star/fixtures/prefill-layer{layer}-continuation-8192-v3/checksums.json'
 if not f.exists():continue
 for row in json.loads(f.read_text())['payloads']:
  if row['file'].endswith(f'_pos{pos}.bin') and row['sha256']!=b[row['file']]['sha256']:known.append(row['file'])
(work/f'{phase}-isolated-controls.json').write_text(json.dumps({'mismatches':known},indent=2)+'\n')
print('Isolated control mismatches',phase,len(known),flush=True)
if diff or known:raise SystemExit(1)
if '--compact-after-verification' in sys.argv:
 dest=work/f'{phase}-b'
 files=list(dest.glob('*.bin'))
 for f in files:f.unlink()
 (dest/'payloads-removed.json').write_text(json.dumps({'reason':'both fresh captures verified exactly against each other and isolated controls; compact hashes retained','files':list(b)},indent=2)+'\n')
print('ACCEPTABLE repeated whole-chunk evidence',phase,flush=True)
