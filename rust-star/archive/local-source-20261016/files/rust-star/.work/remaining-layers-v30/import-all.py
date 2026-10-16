from pathlib import Path
import json,struct,hashlib
root=Path.cwd();work=root/'rust-star/.work/remaining-layers-v30/all-layers';captures={}
for phase in ['prefix','continuation']:
 comparison=json.loads((work/f'{phase}-comparison.json').read_text())
 assert comparison['payloads']==487 and not comparison['mismatches']
 assert not json.loads((work/f'{phase}-isolated-controls.json').read_text())['mismatches']
 a=json.loads((work/f'{phase}-a/hashes.json').read_text());b=json.loads((work/f'{phase}-b/hashes.json').read_text());assert a==b
 captures[phase]=a
model=json.loads((root/'rust-star/.work/remaining-layers-v30/model-verification.json').read_text());assert model['matches_pinned_model']
ref=json.loads((root/'rust-star/.work/oracle-v3-8k-capture/run_01/logits/frontier_008192.logits.json').read_text())['logits'];logits=[]
for phase in ['prefix','continuation']:
 for repeat in ['a','b']:
  f=work/f'{phase}-{repeat}/logits/frontier_008192.logits.json';x=json.loads(f.read_text());assert len(x['logits'])==len(ref)
  logits.append({'run':f'{phase}-{repeat}','sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'mismatching_logits':sum(struct.pack('<f',a)!=struct.pack('<f',b) for a,b in zip(x['logits'],ref)),'argmax':x['argmax_id']})
for layer in range(15,43):
 rows=[]
 for phase,pos in [('prefix',0),('continuation',4096)]:
  selected={name:row for name,row in captures[phase].items() if name.endswith(f'-{layer}_pos{pos}.bin')}
  assert len(selected)==(10 if layer%2 else 13)
  for name,row in sorted(selected.items()):rows.append(dict(file=name,**row,runs=[f'all-layers/{phase}-a',f'all-layers/{phase}-b']))
 fixture=root/f'rust-star/fixtures/prefill-layer{layer}-continuation-8192-v3';fixture.mkdir(exist_ok=True);dest=fixture/'checksums.json'
 if dest.exists():
  old=json.loads(dest.read_text());known={r['file']:r['sha256'] for r in old['payloads']}
  assert known=={r['file']:r['sha256'] for r in rows},layer
  print('Isolated targets preserved',layer);continue
 record=dict(schema='rust-star-remaining-layer-oracle-v1',layer=layer,oracle_commit='d35fb12d01d500b9cefcef24092c295687ceaf7e',oracle_executable_sha256='8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3',model_sha256=model['sha256'],prompt_sha256='f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f',context_tokens=8192,prefill_chunk=4096,gen_tokens=0,fresh_process_repetitions_per_boundary=2,dump_scope='all layers, one chunk position per run',dump_positions=[0,4096],full_8k_logits_c0_claim=False,capture_date='2026-09-06',raw_payload_policy='Temporary raw payloads compacted; both independent hashes, sizes, logs, and isolated-control comparisons retained.',instrumented_logits=logits,payloads=rows)
 dest.write_text(json.dumps(record,indent=2)+'\n');print('Imported layer',layer)
(work/'instrumented-logits.json').write_text(json.dumps(logits,indent=2)+'\n')
