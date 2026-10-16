from pathlib import Path
import json,hashlib,shutil,math
w=Path(__file__).resolve().parent;root=w.parents[2];accepted=json.loads((w/'hc-v5-acceptance.json').read_text());assert accepted['passed'] and accepted['source_commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4' and accepted['executable_sha256']=='a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0';dest=root/'rust-star/fixtures/long-prefill-chunks-v5';assert not dest.exists();dest.mkdir();rows=[];changes={}
for end in accepted['contexts']:
 for rep in ['a','b']:
  d=w/f'hc-v5-{end}-{rep}';record=next(r for r in accepted['records'] if r['frontier']==end and r['repeat']==rep);assert hashlib.sha256((d/'hashes.json').read_bytes()).hexdigest()==record['hashes_sha256']
  j=json.loads((d/'logits'/f'frontier_{end:06d}.logits.json').read_text());assert j['frontier_tokens']==end and j['vocab']==129280 and j['backend']=='metal' and j['quant_bits']==2 and all(map(math.isfinite,j['logits']))
 a=json.loads((w/f'hc-v5-{end}-a/hashes.json').read_text());b=json.loads((w/f'hc-v5-{end}-b/hashes.json').read_text());assert a==b and len(a)==43
 old=json.loads((root/f'rust-star/fixtures/long-prefill-chunks-v31/frontier-{end}-a.json').read_text());changes[str(end)]=[layer for layer in range(43) if a[f'oracle_hc_ffn_post-{layer}_pos{end-4096}.bin']!=old[f'oracle_hc_ffn_post-{layer}_pos{end-4096}.bin']]
 if end==12288:assert not changes[str(end)],'12K direct v5 differs from Rust historical control'
 rows.append('    (\n        '+str(end)+',\n        [\n'+''.join('            0x'+a[f'oracle_hc_ffn_post-{layer}_pos{end-4096}.bin']['fnv1a_u32']+',\n' for layer in range(43))+'        ],\n    ),\n')
 for rep in ['a','b']:shutil.copyfile(w/f'hc-v5-{end}-{rep}/hashes.json',dest/f'frontier-{end}-{rep}.json')
accepted['historical_changed_layers']=changes;accepted['configuration']={'backend':'metal','prefill_chunk':4096,'gen_tokens':0,'prompt':'speed-bench/promessi_sposi.txt','dump_name':'hc_ffn_post','dump_position':'frontier-4096','all_layers':True,'performance_enabled':False};accepted['payload_retention']='Full tensors compacted after SHA256/FNV inventories; both independent inventories and full frontier logits retained in private capture evidence.';(dest/'capture-provenance.json').write_text(json.dumps(accepted,indent=2)+'\n')
p=root/'rust-star/runtime/src/long_prefill_checksums.rs';s=p.read_text();start=s.index('const LONG_PREFILL_CHUNK_CHECKSUMS:');end=s.index('];',start)+2;s='// Independently repeated oracle-v5 HC outputs through24K; historical v3 inventories remain immutable.\n'+s[start:];end=s.index('];')+2;s=s[:s.index('const LONG_PREFILL_CHUNK_CHECKSUMS:')]+'const LONG_PREFILL_CHUNK_CHECKSUMS: &[(u32, [u64; 43])] = &[\n'+''.join(rows)+'];'+s[end:];p.write_text(s)
(w/'hc-v5-changes.json').write_text(json.dumps(changes,indent=2)+'\n');print(changes)
