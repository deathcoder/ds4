from pathlib import Path
import hashlib,json,subprocess,sys
root=Path.cwd();layer=int(sys.argv[1]);folder=root/f'rust-star/.work/remaining-layers-v30/layer{layer}'
rows=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for phase,pos in [('prefix',0),('continuation',4096)]:
 files=sorted((folder/(phase+'-a')).glob('*.bin'));assert len(files)==(10 if layer%2 else 13),len(files)
 for a in files:
  b=folder/(phase+'-b')/a.name;sa,sb=sha(a),sha(b)
  assert sa==sb,(layer,phase,a.name,sa,sb)
  fnv=subprocess.check_output([str(root/'rust-star/.work/layer13-continuation-v28/hash_words'),str(a)],text=True).split()[0]
  rows.append(dict(file=a.name,bytes=a.stat().st_size,sha256=sa,fnv1a_u32=fnv,runs=[phase+'-a',phase+'-b']))
record=dict(schema='rust-star-remaining-layer-oracle-v1',layer=layer,oracle_commit='d35fb12d01d500b9cefcef24092c295687ceaf7e',oracle_executable_sha256='8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3',model_sha256='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0',prompt_sha256='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f',context_tokens=8192,prefill_chunk=4096,gen_tokens=0,fresh_process_repetitions_per_boundary=2,dump_positions=[0,4096],full_8k_logits_c0_claim=False,payloads=rows)
(folder/'oracle-evidence.json').write_text(json.dumps(record,indent=2)+'\n')
fixture=root/f'rust-star/fixtures/prefill-layer{layer}-continuation-8192-v3';fixture.mkdir(exist_ok=True);(fixture/'checksums.json').write_text(json.dumps(record,indent=2)+'\n')
# Rebuild only the contiguous verified target list.
if '--evidence-only' in sys.argv:
 print('Verified layer',layer,'payload pairs',len(rows),flush=True)
 raise SystemExit(0)
lines=[]
for n in range(15,43):
 f=root/f'rust-star/fixtures/prefill-layer{n}-continuation-8192-v3/checksums.json'
 if not f.exists():break
 x=json.loads(f.read_text());lookup={r['file']:r['fnv1a_u32'] for r in x['payloads']}
 names=['hc_attn_pre','attn_norm','Qcur','KVcur','KVcompress']
 if n%2==0:names+=['attn_state_kv','attn_state_score','indexer_KVcompress','indexer_state_kv','indexer_state_score']
 def vals(names,pos):return ', '.join('0x'+lookup[f'oracle_{name}-{n}_pos{pos}.bin'] for name in names)
 tails=['kqv_back','hc_attn_post','hc_ffn_post']
 lines.append(f'    ({n}, &[{vals(names,0)}], &[{vals(names,4096)}], [{vals(tails,0)}], [{vals(tails,4096)}]),')
(root/'rust-star/runtime/src/remaining_prefill_checksums.rs').write_text("type RemainingPrefillChecksums = (u32, &'static [u64], &'static [u64], [u64; 3], [u64; 3]);\nconst REMAINING_PREFILL_CHECKSUMS: &[RemainingPrefillChecksums] = &[\n"+'\n'.join(lines)+'\n];\n')
print('Verified layer',layer,'payload pairs',len(rows),flush=True)
