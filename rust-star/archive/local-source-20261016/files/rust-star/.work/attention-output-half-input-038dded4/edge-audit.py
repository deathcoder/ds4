from pathlib import Path
import json
w=Path(__file__).resolve().parent;p=json.loads((w/'plan.json').read_text());d=w/'edges';rows=[json.loads(l) for l in (d/'probe.jsonl').read_text().splitlines()]
assert json.loads((d/'status.json').read_text())['exit_code']==0
assert rows[0]['schema']==p['schema'] and rows[0]['mode']=='edges'
assert rows[0]['argument_sizes']==[48,104,88,152]
assert rows[-1]['event']=='complete' and rows[-1]['passed'] and rows[-1]['cases']==16 and rows[-1]['timed_pairs']==0
seen=0;total=0;checksums={};i=1
for c in range(16):
 shape=rows[i];i+=1;expected_rows=p['edge_rows'][c//2];assert shape['event']=='shape' and shape['case']==c and shape['rows']==expected_rows and shape['precision_pattern']==bool(c%2)
 words=expected_rows*(8192+4096+16384);assert shape['output_words']==words
 checksums[c]=None
 for rep in range(2):
  r=rows[i];i+=1;assert r['event']=='warmup' and r['case']==c and r['repetition']==rep+1 and r['order']==['AB','BA'][rep]
  assert r['exact'] and r['cpu_half_conversion_exact'] and r['checked_words']==words
  assert len(r['fnv1a_u32'])==3
  if rep:assert checksums[c]==r['fnv1a_u32']
  checksums[c]=r['fnv1a_u32'];total+=words
 end=rows[i];i+=1;assert end==dict(event='shape_complete',case=c,inputs_unchanged=True,guards_exact=True,cpu_gpu_map_exact=True)
assert i==len(rows)-1 and rows[-1]['compared_fp32_words']==total
r=dict(passed=True,cases=16,full_fp32_words_compared=total,cpu_half_conversions_exact=True,cpu_gpu_maps_exact=True,checksums=checksums)
if (w/'edge-audit.json').exists():assert json.loads((w/'edge-audit.json').read_text())==json.loads(json.dumps(r))
else:(w/'edge-audit.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
