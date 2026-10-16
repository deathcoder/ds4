from pathlib import Path
import hashlib,json,shutil
w=Path(__file__).resolve().parent;root=w.parents[2];target=root/'rust-star/reports/32k-baseline-7406473'
def read(p):return json.loads(p.read_text())
assert read(w/'independent-audit.json')['passed']
assert not target.exists(), 'Never overwrite evidence'
target.mkdir(parents=True)
shutil.copytree(w/'results',target/'results')
for name in ['independent-audit.json','profile32k.json','host-observations.jsonl','load-observations.jsonl','conformance.json']:
 shutil.copyfile(w/name,target/name)
(target/'inputs').mkdir()
for p in (w/'inputs').glob('*.json'):
 if p.name != 'plan.json':shutil.copyfile(p,target/'inputs'/p.name)
for rep in ['a','b']:
 shutil.copyfile(w/f'conformance-{rep}/verification.json',target/f'conformance-{rep}.json')
shutil.copyfile(w/'oracle-source/benchmark-build.json',target/'benchmark-build.json')
old=w.with_name('paired-32k-128-7406473')
shutil.copyfile(old/'rejected-plan.json',target/'rejected-plan.json')
plan=read(w/'inputs/plan.json')
# A portable template is distinct from the frozen private plan; the original
# hash remains recorded. Never alter hashed raw or measurement evidence.
for role in ['oracle','candidate']:
 e=plan[role];e['working_directory']='$ORACLE_BUILD' if role=='oracle' else '$CANDIDATE_BUILD'
 args=e['adapter_command']
 for i,v in enumerate(args):
  if v.startswith(str(root)):args[i]='$REPO/'+v[len(str(root))+1:]
  elif v.startswith('/Users/'):args[i]='$MODEL' if i and args[i-1]=='--model' else '$LOCAL_PATH'
(target/'inputs/plan-template.json').write_text(json.dumps(plan,indent=2)+'\n')
private=[]
for p in target.rglob('*'):
 if p.is_file() and '/Users/' in p.read_text():private.append(str(p.relative_to(target)))
assert not private, private
print(target)
