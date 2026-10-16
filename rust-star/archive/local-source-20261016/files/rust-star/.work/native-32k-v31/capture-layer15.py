from pathlib import Path
import hashlib,json,os,subprocess
root=Path.cwd();w=root/'rust-star/.work/native-32k-v31';src=root/'rust-star/.work/oracle-v3-8k-source';exe=src/'ds4-bench.accepted-oracle-v3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for repeat in ['c','d']:
 d=w/f'oracle-28672-layer15-{repeat}';d.mkdir();(d/'logits').mkdir()
 env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_'))};env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(d/'oracle'),DS4_METAL_GRAPH_DUMP_POS='24576',DS4_METAL_GRAPH_DUMP_LAYER='15')
 cmd=[str(exe),'--metal','--model',str(src/'model.gguf'),'--prompt-file',str(root/'speed-bench/promessi_sposi.txt'),'--ctx-start','28672','--ctx-max','28672','--gen-tokens','0','--prefill-chunk','4096','--dump-frontier-logits-dir',str(d/'logits')]
 print('capture layer15',repeat,flush=True)
 with (d/'console.log').open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 hashes={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(d.glob('*.bin'))}
 (d/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print('captured',len(hashes),'payloads',flush=True)
a=json.loads((w/'oracle-28672-layer15-c/hashes.json').read_text());b=json.loads((w/'oracle-28672-layer15-d/hashes.json').read_text());print('differences',[k for k in a if a[k]!=b[k]],flush=True)
