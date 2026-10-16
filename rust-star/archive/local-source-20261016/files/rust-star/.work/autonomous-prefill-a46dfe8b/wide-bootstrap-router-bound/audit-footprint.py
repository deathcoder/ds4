from pathlib import Path
import json,re,statistics,hashlib
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());med=statistics.median
assert read(w/'engine/execution-complete.json')['passed'];plan=read(w/'engine-plan.json');progress=read(w/'engine/progress.json');assert len(progress)==12
manifest=read(w/'artifacts.json')['files'] if (w/'artifacts.json').exists() else None
def sha(p):
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if manifest:
  e=manifest[str(p.relative_to(w))];assert h==e['stored_sha256'];return e['original_sha256']
 return h
def stats(v):
 m=med(v);return dict(count=len(v),median=m,mad=med(abs(x-m) for x in v),minimum=min(v),maximum=max(v),values=v)
raw={};logs={}
for row in progress:
 p=w/'engine'/row['name']/'run.log';matches=re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$',p.read_text(),re.M);assert len(matches)==1 and int(matches[0])>0;raw[row['name']]=int(matches[0]);logs[str(p.relative_to(w))]=sha(p)
pairs=[dict(pair=i,order=o,baseline_bytes=raw[f'pair-{i}-A'],candidate_bytes=raw[f'pair-{i}-B'],ratio=raw[f'pair-{i}-B']/raw[f'pair-{i}-A']) for i,o in enumerate(plan['orders'],1)]
r=dict(passed=True,metric='peak memory footprint reported by macOS /usr/bin/time -l',interpretation='Supplemental descriptive metric, distinct from maximum resident set size. No new or changed retention threshold.',summary=dict(baseline=stats([p['baseline_bytes'] for p in pairs]),candidate=stats([p['candidate_bytes'] for p in pairs]),paired_ratio=stats([p['ratio'] for p in pairs]),orders={o:stats([p['ratio'] for p in pairs if p['order']==o]) for o in ['AB','BA']}),pairs=pairs,warmups={n:raw[n] for n in ['warmup-A','warmup-B']},raw_files=logs)
p=w/'memory-footprint-audit.json'
if p.exists():assert read(p)==r
else:p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(passed=True,baseline_median=r['summary']['baseline']['median'],candidate_median=r['summary']['candidate']['median'],paired_ratio_median=r['summary']['paired_ratio']['median']),indent=2))
