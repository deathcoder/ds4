from pathlib import Path
import os,sys,subprocess,hashlib,json,shutil
root=Path.cwd();work=root/'rust-star/.work/remaining-layers-v30';source=root/'rust-star/.work/oracle-v3-8k-source';exe=source/'ds4-bench.accepted-oracle-v3';prompt=root/'speed-bench/promessi_sposi.txt'
assert hashlib.sha256(exe.read_bytes()).hexdigest()=='8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3'
assert hashlib.sha256(prompt.read_bytes()).hexdigest()=='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f'
model=source/'model.gguf';assert model.exists()
assert model.resolve()==Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()=='d35fb12d01d500b9cefcef24092c295687ceaf7e'
for layer in range(int(sys.argv[1]),int(sys.argv[2])+1):
 folder=work/f'layer{layer}';folder.mkdir(exist_ok=True)
 if (folder/'oracle-evidence.json').exists():
  print('Already verified',layer,flush=True);continue
 names='hc_attn_pre,attn_norm,Qcur,KVcur,KVcompress,attn_state_kv,attn_state_score,kqv_back,hc_attn_post,hc_ffn_post'
 if layer%2==0:names+=',indexer_KVcompress,indexer_state_kv,indexer_state_score'
 for phase,pos in [('prefix',0),('continuation',4096)]:
  for repeat in ['a','b']:
   dest=folder/f'{phase}-{repeat}'
   if (dest/'completed').exists():continue
   assert not dest.exists(),f'Unfinished capture needs inspection: {dest}'
   if shutil.disk_usage(root).free<12*2**30:raise RuntimeError('Insufficient disk headroom before next capture')
   dest.mkdir();(dest/'logits').mkdir()
   env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_'))}
   env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(dest/'oracle'),DS4_METAL_GRAPH_DUMP_LAYER=str(layer),DS4_METAL_GRAPH_DUMP_POS=str(pos),DS4_METAL_GRAPH_DUMP_NAME=names)
   command=[str(exe),'--metal','--model',str(model),'--prompt-file',str(prompt),'--ctx-start','8192','--ctx-max','8192','--gen-tokens','0','--prefill-chunk','4096','--dump-frontier-logits-dir',str(dest/'logits')]
   print('Capture',layer,phase,repeat,flush=True)
   with (dest/'console.log').open('w') as f:subprocess.run(command,cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   (dest/'completed').write_text('exit 0\n')
 subprocess.run([sys.executable,str(work/'verify-layer.py'),str(layer),'--evidence-only'],check=True)
 # Preserve one complete raw capture per position and all repeat hashes/logs.
 deleted=[]
 for phase in ['prefix','continuation']:
  for f in (folder/f'{phase}-b').glob('*.bin'):
   deleted.append(dict(file=str(f.relative_to(folder)),bytes=f.stat().st_size));f.unlink()
 (folder/'duplicate-payload-cleanup.json').write_text(json.dumps(deleted,indent=2)+'\n')
 print('VERIFIED',layer,'duplicate payload bytes reclaimed',sum(x['bytes'] for x in deleted),flush=True)
