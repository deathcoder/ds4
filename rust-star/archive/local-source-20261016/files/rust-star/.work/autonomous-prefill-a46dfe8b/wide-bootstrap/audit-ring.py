"""Independent modulo-write reference for the fixed 128-row raw snapshot."""
from pathlib import Path
import json
w=Path(__file__).resolve().parent
cases=[]
for rows in [32,64,4096]:
 for start in range(0 if rows<4096 else 4096,32768-rows+1,rows):
  values=[(start+i)*65537+17 for i in range(rows)]
  reference=[-1]*128
  for i,v in enumerate(values):reference[(start+i)%128]=v
  count=min(rows,128);source=rows-count;target=(start+source)%128
  assert source>=0 and source+count<=rows and target+count<=128
  candidate=[-1]*128;candidate[target:target+count]=values[source:source+count]
  assert reference==candidate
  if rows<4096:
   old=[-1]*128;old[start%128:start%128+rows]=values;assert old==candidate
  cases.append(dict(rows=rows,start=start,source_row=source,target_row=target,copied_rows=count))
assert 4096>128 # The old full-width copy would overrun the physical snapshot.
r=dict(passed=True,cases=len(cases),short_cases=sum(x['rows']<4096 for x in cases),wide_cases=sum(x['rows']==4096 for x in cases),scope='Indexing, bounds and short-row sentinel preservation; full-model gates validate GPU values and inference.',records=cases)
p=w/'ring-audit.json'
if p.exists():assert json.loads(p.read_text())==r
else:p.write_text(json.dumps(r,indent=2)+'\n')
print('Passed',len(cases),'raw snapshot indexing/bounds cases')
