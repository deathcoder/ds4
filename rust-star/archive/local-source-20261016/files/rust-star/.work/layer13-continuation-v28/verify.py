import hashlib,json,subprocess
from pathlib import Path
root=Path.cwd(); out=root/'rust-star/.work/layer13-continuation-v28'; a=out/'a2'; b=out/'b2'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
files=sorted(a.glob('*.bin')); assert len(files)==20,len(files)
rows=[]
for f in files:
    other=b/f.name
    assert f.stat().st_size==other.stat().st_size
    digest=sha(f); assert digest==sha(other),f.name
    values=subprocess.check_output([str(out/'hash_words'),str(f)],text=True).split()
    rows.append(dict(file=f.name,bytes=f.stat().st_size,sha256=digest,fnv1a_u32=values[0]))
import struct
baseline=root/'rust-star/.work/oracle-v3-8k-capture/run_01/logits/frontier_008192.logits.json'
def bits(p):
    return [struct.pack('<f',v) for v in json.loads(p.read_text())['logits']]
ref=bits(baseline); observed=[]
for folder in (a,b):
    logits=folder/'logits/frontier_008192.logits.json'; actual=bits(logits)
    observed.append(dict(run=folder.name,sha256=sha(logits),mismatching_logits=sum(x!=y for x,y in zip(ref,actual)),argmax=json.loads(logits.read_text())['argmax_id']))
result=dict(schema='rust-star-layer13-continuation-oracle-v1',oracle_commit='d35fb12d01d500b9cefcef24092c295687ceaf7e',oracle_executable_sha256='8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3',model_sha256='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0',prompt_sha256='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f',context_tokens=8192,prefill_chunk=4096,gen_tokens=0,layer=13,fresh_process_repetitions=2,full_8k_logits_c0_claim=False,instrumented_logits=observed,payloads=rows)
(out/'oracle-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
lookup={x['file']:x['fnv1a_u32'] for x in rows}
for pos in (0,4096):
    for name in ('hc_attn_pre','attn_norm','Qcur','KVcur','KVcompress','kqv_back','hc_attn_post','hc_ffn_post'):
        print(pos,name,lookup[f'oracle_{name}-13_pos{pos}.bin'])
print('All 20 payload pairs match; instrumented full logits do not establish C0.')
