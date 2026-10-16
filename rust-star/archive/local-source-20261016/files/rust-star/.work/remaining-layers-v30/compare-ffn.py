from pathlib import Path
import numpy as np,json,hashlib
w=Path('rust-star/.work/remaining-layers-v30');out={}
for p in sorted((w/'rust26-ffn').glob('*.bin')):
 o=w/'oracle26-ffn'/('oracle_'+p.stem+'-26_pos0.bin')
 if not o.exists():continue
 a=np.fromfile(p,dtype='<u4');b=np.fromfile(o,dtype='<u4');assert a.shape==b.shape,(p,a.shape,b.shape)
 ix=np.flatnonzero(a!=b);width=256 if 'logits' in p.stem or 'probs' in p.stem else 6 if 'weights' in p.stem else 16384 if p.stem=='hc_ffn_post' else 4096
 out[p.stem]={'mismatches':len(ix),'rows':np.unique(ix//width).tolist(),'first_indices':ix[:10].tolist(),'actual_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'oracle_sha256':hashlib.sha256(o.read_bytes()).hexdigest()}
 print(p.stem,len(ix),'rows',out[p.stem]['rows'][:10])
p=w/'rust26-ffn/ffn_moe_selected.bin'
if p.exists():
 a=np.fromfile(p,dtype='<i4').reshape(-1,6);b=np.load(w/'layer26-bitonic-selected.npy');rows=np.flatnonzero(np.any(a!=b,axis=1));print('router rows',rows.tolist());out['selected_vs_oracle_bitonic']={'rows':rows.tolist(),'actual':a[rows].tolist(),'expected':b[rows].tolist()}
(w/'layer26-ffn-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
