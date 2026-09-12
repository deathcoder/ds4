from pathlib import Path
import json,statistics as st,hashlib
w=Path(__file__).resolve().parent;out=[]
for session in [0,1]:
 rows=[json.loads(s) for s in (w/f'aa-only/session-{session}/probe.jsonl').read_text().splitlines()]
 for case in range(3):
  cv=[];ranges=[]
  for x in rows:
   if x.get('event')!='pair' or x['case']!=case:continue
   for role in x['timing']:
    times=[c['gpu_ms'] for c in role['commands'] if not c['conditioning']];assert len(times)==3;cv.append(st.pstdev(times)/st.mean(times));ranges.append(max(times)/min(times))
  out.append(dict(session=session,case=case,slots=len(cv),median_triplet_cv=st.median(cv),maximum_triplet_cv=max(cv),maximum_triplet_duration_ratio=max(ranges)))
r=dict(interpretation='Exploratory diagnosis of identical-kernel timing variability; no exclusions, changed gates or candidate performance inference.',statistics=out,next_hypothesis='Prospectively test longer conditioning and a longer median-based sampling window, retaining the same neutral/candidate thresholds. Do not reclassify this failed screen.')
p=w/'variance-diagnosis.json'
if p.exists():assert json.loads(p.read_text())==r
else:p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
