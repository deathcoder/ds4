from pathlib import Path
import hashlib,json
w=Path(__file__).resolve().parent;plan=json.loads((w.parent/'plan.json').read_text())['edge_tests'];read=lambda n:json.loads((w/n).read_text())
assert read('status.json')['exit_code']==0
rows=[json.loads(l) for l in (w/'probe.jsonl').read_text().splitlines()]
h=rows.pop(0);assert h==dict(schema='rust-star-compressed-skip-edges-v1',device='Apple M1 Ultra',rows=16,heads=2,width=512,patterns=8,key_cases=15,pairs_per_case=2)
words=2*16*2*512;compared=0;types=[0,0,0];zero_cases=0
for ki,keys in enumerate(plan['key_cases']):
 nblocks=(keys+63)//64
 for pattern in range(8):
  def live(row,col):
   b=col//64
   return [False,True,b==0,b==nblocks-1,b%4==0 or b==nblocks-1,b%8==7 or b==nblocks-1,(b%9==0 or b%9==8 or b==nblocks-1) and (col+row)%3!=1,col==0 or col==keys-1][pattern]
  counts=[0,0,0]
  for qb in range(2):
   for kb in range(nblocks):
    if (kb+1)*64>keys:kind=1
    else:
     visible=sum(live(row,col) for row in range(qb*8,qb*8+8) for col in range(kb*64,kb*64+64));kind=2 if visible==512 else (1 if visible else 0)
    counts[kind]+=1
  r=rows.pop(0);assert (r['event'],r['key_index'],r['keys'],r['pattern'],r['padded'],r['mask_blocks'],r['all_masked_zero'],r['checked_words'])==('case',ki,keys,pattern,keys%64!=0,counts,pattern==0,words)
  assert all(r[k] for k in ['cpu_gpu_preparation_exact','exact','guards_exact','inputs_unchanged'])
  if pattern==0:
   h=0xcbf29ce484222325
   for _ in range(words//2):h=(h*0x100000001b3)&((1<<64)-1)
   assert r['output_fnv1a_u32']==h,'All-masked output must be positive zero in every word'
   zero_cases+=1
  compared+=words;types=[a+b for a,b in zip(types,counts)]
assert all(types) and rows==[dict(event='complete',passed=True,cases=120,compared_fp32_words=compared)]
result=dict(passed=True,cases=120,compared_fp32_words=compared,all_masked_positive_zero_cases=zero_cases,aggregate_block_types=types,raw_sha256=hashlib.sha256((w/'probe.jsonl').read_bytes()).hexdigest())
if (w/'audit.json').exists():assert read('audit.json')==result
else:(w/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
