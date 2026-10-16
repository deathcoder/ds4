from pathlib import Path
import json,hashlib,subprocess,os
w=Path(__file__).resolve().parent;root=w.parents[2];src=w.with_name('oracle-v5-source');exe=src/'ds4-bench';hasher=root/'rust-star/.work/layer13-continuation-v28/hash_words';model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';records=[]
for rep in ['a','b']:
 d=w/f'all-selectors-v5-{rep}';d.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(d/'oracle'),DS4_METAL_GRAPH_DUMP_POS='4096',DS4_METAL_GRAPH_DUMP_NAME='indexer_scores,indexer_topk',DS4_METAL_GRAPH_DUMP_LAYER='all')
 cmd=[str(exe),'--metal','-m',model,'--prompt-file',str(root/'speed-bench/promessi_sposi.txt'),'--ctx-start','8192','--ctx-max','8192','--gen-tokens','0','--prefill-chunk','4096','--csv',str(d/'benchmark.csv')];print('START all selectors',rep,flush=True)
 with (d/'console.log').open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 hashes={}
 for layer in range(2,43,2):
  hashes[str(layer)]={}
  for name,ext,size in [('indexer_scores','bin',4096*2048*4),('indexer_topk','i32',4096*512*4)]:
   p=d/f'oracle_{name}-{layer}_pos4096.{ext}';b=p.read_bytes();assert len(b)==size;h=subprocess.check_output([str(hasher),str(p)],text=True).split()[0];hashes[str(layer)][name]={'sha256':hashlib.sha256(b).hexdigest(),'fnv1a_u32':h,'bytes':len(b)}
 assert int(hashes['4']['indexer_topk']['fnv1a_u32'],16)==14229657549071909978
 (d/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n');records.append(hashes);print('PASS all selectors',rep,flush=True)
assert records[0]==records[1]
(w/'all-selectors-v5-acceptance.json').write_text(json.dumps({'passed':True,'source_commit':'ccbd16d36895ef9fe8edf7f793014988ff1ab7f4','executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'fresh_repetitions':2,'prefill_tokens':8192,'chunk_start':4096,'rows':4096,'top_k':512,'compressed_rows':2048,'layers':list(range(2,43,2)),'hashes':records[0]},indent=2)+'\n')
