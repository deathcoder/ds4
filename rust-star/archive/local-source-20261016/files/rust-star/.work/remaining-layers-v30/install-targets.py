from pathlib import Path
import json
root=Path.cwd()
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
print('Installed verified remaining layers:', len(lines))
