import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd(); out=root/'rust-star/.work/remaining-layers-v30/layer16'
out.mkdir(exist_ok=True)
source=root/'rust-star/.work/oracle-v3-8k-source'; exe=source/'ds4-bench.accepted-oracle-v3'
manifest=json.loads((root/'rust-star/fixtures/prefill-frontier-8192-v3/manifest.json').read_text())
assert hashlib.sha256(exe.read_bytes()).hexdigest()==manifest['oracle']['capture_executable_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()=='d35fb12d01d500b9cefcef24092c295687ceaf7e'
prompt=root/'speed-bench/promessi_sposi.txt'
assert hashlib.sha256(prompt.read_bytes()).hexdigest()=='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f'
model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf')
for repeat,position in [('prefix-a',0),('prefix-b',0),('continuation-a',4096),('continuation-b',4096)]:
    dest=out/repeat; dest.mkdir(exist_ok=False); (dest/'logits').mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_'))}
    env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(dest/'oracle'),DS4_METAL_GRAPH_DUMP_LAYER='16',DS4_METAL_GRAPH_DUMP_POS=str(position),DS4_METAL_GRAPH_DUMP_NAME='hc_attn_pre,attn_norm,Qcur,KVcur,KVcompress,attn_state_kv,attn_state_score,indexer_KVcompress,indexer_state_kv,indexer_state_score,kqv_back,hc_attn_post,hc_ffn_post')
    cmd=[str(exe),'--metal','--model',str(model),'--prompt-file',str(prompt),'--ctx-start','8192','--ctx-max','8192','--gen-tokens','0','--prefill-chunk','4096','--dump-frontier-logits-dir',str(dest/'logits')]
    print('Starting fresh oracle capture',repeat,flush=True)
    with (dest/'console.log').open('w') as log: subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Finished oracle capture',repeat,flush=True)
