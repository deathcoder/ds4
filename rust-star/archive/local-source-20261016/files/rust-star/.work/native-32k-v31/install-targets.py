from pathlib import Path
import json
w=Path('rust-star/.work/native-32k-v31');lines=['// Independently repeated oracle-v3 HC outputs for each later chunk.','const LONG_PREFILL_CHUNK_CHECKSUMS: &[(u32, [u64; 43])] = &['];count=0
for end in range(12288,32769,4096):
 a=w/f'oracle-{end}-a/hashes.json';b=w/f'oracle-{end}-b/hashes.json'
 if not a.exists() or not b.exists():break
 x=json.loads(a.read_text())
 if x!=json.loads(b.read_text()):
  print('unaccepted non-repeating frontier',end);break
 values=[x[f'oracle_hc_ffn_post-{layer}_pos{end-4096}.bin']['fnv1a_u32'] for layer in range(43)]
 lines.append(f'    ({end}, ['+', '.join('0x'+v for v in values)+']),');count+=1
lines.append('];')
old=Path('rust-star/runtime/src/long_prefill_checksums.rs').read_text()
if '// Repeated complete 8K' in old: lines.append(old[old.index('// Repeated complete 8K'):])
Path('rust-star/runtime/src/long_prefill_checksums.rs').write_text('\n'.join(lines)+'\n');print('installed checkpoints',count)
