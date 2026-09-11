from pathlib import Path
import json,math,hashlib,statistics
w=Path(__file__).resolve().parent;read=lambda n:json.loads((w/n).read_text())
rows=[json.loads(s) for s in (w/'probe.jsonl').read_text().splitlines()];plan=read('plan.json')
assert read('status.json')['exit_code']==0
header=rows.pop(0);assert header['schema']=='rust-star-compressed-grid-probe-v1' and header['device']=='Apple M1 Ultra'
assert (header['rows'],header['heads'],header['width'],header['threads_per_group'],header['threadgroup_bytes'],header['warmups_per_variant'],header['pairs_per_timed_case'])==(4096,64,512,256,28672,2,7)
def block_counts(start,keys):
 counts=[0,0,0]
 for qb in range(512):
  for kb in range((keys+63)//64):
   lo,hi=64*kb,64*kb+63
   if hi>=keys:flag=1
   else:
    visible=0
    for r in range(8*qb,8*qb+8):
     visible+=max(0,min(hi,4223,r+128)-max(lo,r+1)+1)
     visible+=max(0,min(hi,keys-1,4224+(start+r+1)//128-1)-max(lo,4224)+1)
    flag=2 if visible==512 else (1 if visible else 0)
   counts[flag]+=1
 return counts
words=4096*64*512;comparison_words=0;timed=0;shapes=[]
for case,shape in enumerate(plan['correctness_cases']):
 record=rows.pop(0);assert record['event']=='shape' and record['case']==case
 assert record['chunk_start']==shape['chunk_start'] and record['key_rows']==shape['key_rows'] and record['padded']==shape['padded']
 assert record['cpu_gpu_preparation_exact'] and record['output_words']==words
 assert record['mask_blocks']==block_counts(shape['chunk_start'],shape['key_rows'])
 checksum=None
 for warmup in range(2):
  r=rows.pop(0);assert (r['event'],r['case'],r['repetition'],r['order'])==('warmup',case,warmup+1,'AB' if warmup%2==0 else 'BA')
  assert r['exact'] and r['checked_words']==words
  if checksum is not None:assert r['fnv1a_u32']==checksum
  checksum=r['fnv1a_u32'];comparison_words+=words
 stats=None
 if shape['chunk_start'] in plan['timed_chunk_starts']:
  raw=[]
  for pair in range(7):
   r=rows.pop(0);assert (r['event'],r['case'],r['pair'],r['order'])==('pair',case,pair+1,'AB' if pair%2==0 else 'BA')
   assert r['exact'] and r['checked_words']==words and r['fnv1a_u32']==checksum
   a,b=r['baseline_gpu_ms'],r['candidate_gpu_ms'];assert math.isfinite(a) and math.isfinite(b) and a>0 and b>0
   assert abs(r['ratio']-b/a)<1e-8
   raw.append(r);comparison_words+=words;timed+=1
  ratios=[r['candidate_gpu_ms']/r['baseline_gpu_ms'] for r in raw];median=statistics.median(ratios);faster=sum(v<1 for v in ratios)
  stats=dict(pairs=7,median_paired_ratio=median,mad_paired_ratio=statistics.median(abs(v-median) for v in ratios),faster_pairs=faster,minimum_ratio=min(ratios),maximum_ratio=max(ratios),baseline_median_ms=statistics.median(r['baseline_gpu_ms'] for r in raw),candidate_median_ms=statistics.median(r['candidate_gpu_ms'] for r in raw),passes=median<1 and faster>=5,raw_pairs=raw)
 r=rows.pop(0);assert (r['event'],r['case'])==('shape_complete',case) and r['inputs_unchanged'] and r['guards_exact']
 shapes.append(dict(case=case,**shape,output_fnv1a_u32=checksum,mask_blocks=record['mask_blocks'],statistics=stats))
assert rows==[dict(event='complete',passed=True,cases=7,timed_pairs=21,compared_fp32_words=4697620480)]
assert comparison_words==4697620480 and timed==21
screen=all(s['statistics']['passes'] for s in shapes if s['statistics'])
result=dict(passed=True,all_exactness_checks_pass=True,full_fp32_words_compared=comparison_words,shapes=shapes,screening_passed=screen,runtime_optimization_accepted=False,engine_speedup_claim=False,raw_sha256=hashlib.sha256((w/'probe.jsonl').read_bytes()).hexdigest())
if (w/'audit.json').exists():assert read('audit.json')==result,'Stored probe audit differs'
else:(w/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,fp32_words_compared=comparison_words,screening_passed=screen,statistics=[{k:v for k,v in s.items() if k!='statistics'}|dict(statistics={k:v for k,v in s['statistics'].items() if k!='raw_pairs'}) for s in shapes if s['statistics']]),indent=2))
