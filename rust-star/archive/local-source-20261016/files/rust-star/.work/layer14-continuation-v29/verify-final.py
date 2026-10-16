import hashlib,json,subprocess,struct,re
from pathlib import Path
root=Path.cwd();out=root/'rust-star/.work/layer14-continuation-v29'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
rows=[]
for pos,runs in [(0,('a','b')),(4096,('c','d'))]:
 files=sorted((out/runs[0]).glob(f'*_pos{pos}.bin'));assert len(files)==13
 for f in files:
  other=out/runs[1]/f.name;digest=sha(f);assert digest==sha(other),f.name
  value=subprocess.check_output([str(root/'rust-star/.work/layer13-continuation-v28/hash_words'),str(f)],text=True).split()[0]
  rows.append(dict(file=f.name,bytes=f.stat().st_size,sha256=digest,fnv1a_u32=value,repetitions=list(runs)))
result=dict(schema='rust-star-layer14-continuation-oracle-v1',oracle_commit='d35fb12d01d500b9cefcef24092c295687ceaf7e',oracle_executable_sha256='8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3',model_sha256='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0',prompt_sha256='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f',context_tokens=8192,prefill_chunk=4096,gen_tokens=0,layer=14,fresh_process_repetitions_per_boundary=2,full_8k_logits_c0_claim=False,capture_note='Prefix uses matching a/b captures. Continuation uses c/d with DUMP_POS=4096; a/b continuation drifted and is rejected.',payloads=rows)
(out/'oracle-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
lookup={x['file']:x['fnv1a_u32'] for x in rows}
s=Path('rust-star/runtime/src/metal.rs').read_text()
for pos,section in [(0,'PREFIX'),(4096,'CONTINUATION')]:
 for kind,names in [('BOUNDARY',['hc_attn_pre','attn_norm','Qcur','KVcur','KVcompress','attn_state_kv','attn_state_score','indexer_KVcompress','indexer_state_kv','indexer_state_score']),('TAIL',['kqv_back','hc_attn_post','hc_ffn_post'])]:
  values=[lookup[f'oracle_{name}-14_pos{pos}.bin'] for name in names]
  target=f'PREFILL_LAYER14_{section}_{kind}_CHECKSUMS'
  replacement=f'const {target}: [u64; {len(values)}] = [\n'+''.join('    0x'+ '_'.join(v[i:i+4] for i in range(0,16,4))+',\n' for v in values)+'];'
  s,n=re.subn(r'const '+target+r': \[u64; \d+\] = \[.*?\];',replacement,s,flags=re.S);assert n==1
Path('rust-star/runtime/src/metal.rs').write_text(s)
fixture=root/'rust-star/fixtures/prefill-layer14-continuation-8192-v3';fixture.mkdir(exist_ok=True)
(fixture/'checksums.json').write_text(json.dumps(result,indent=2)+'\n')
print('26 tensor pairs exact; installed independently verified layer-14 targets.')
