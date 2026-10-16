from pathlib import Path
import hashlib,json,shutil
source=Path(__file__).resolve().parent;root=source.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
w=Path(read(source/'topk-plan-location.json')['work'])
assert read(w/'independent-audit.json')['passed']
target=root/'rust-star/reports/32k-baseline-cbb078bb'
assert not target.exists(),'Never overwrite evidence'
target.mkdir(parents=True)
shutil.copytree(w/'results',target/'results')
for name in ['independent-audit.json','profile32k.json','host-observations.jsonl','load-observations.jsonl']:
 shutil.copyfile(w/name,target/name)
(target/'inputs').mkdir()
for p in (w/'inputs').glob('*.json'):
 if p.name!='plan.json':shutil.copyfile(p,target/'inputs'/p.name)
plan=read(w/'inputs/plan.json')
for role in ['oracle','candidate']:
 e=plan[role];e['working_directory']='$ORACLE_BUILD' if role=='oracle' else '$CANDIDATE_BUILD'
 args=e['adapter_command']
 for i,v in enumerate(args):
  if v.startswith(str(root)):args[i]='$REPO/'+v[len(str(root))+1:]
  elif v.startswith('/Users/'):args[i]='$MODEL' if i and args[i-1]=='--model' else '$LOCAL_PATH'
write(target/'inputs/plan-template.json',plan)
proof=target/'correctness';proof.mkdir()
for name in ['oracle-v4-verification.json','oracle-v4-target-equivalence.json','conformance-v4.json','topk-validation.json','topk-barrier-build.json','rust-tests-topk.log','python-tests-oracle-v4.log','topk-regression.log']+[f'topk-{n}.json' for n in ['a','b','c','d','2k','8k']]:
 shutil.copyfile(source/name,proof/name)
for rep in ['a','b']:shutil.copyfile(source/f'conformance-v4-{rep}/verification.json',proof/f'conformance-v4-{rep}.json')
shutil.copytree(source/'oracle-v4-acceptance',proof/'oracle-v4-acceptance')
shutil.copyfile(source/'oracle-v4-benchmark/benchmark-build.json',proof/'benchmark-build.json')
rejected=target/'rejected';rejected.mkdir()
old=source.with_name('paired-32k-128-7406473')
shutil.copyfile(old/'rejected-plan.json',rejected/'timer-plan.json')
shutil.copytree(source/'results',rejected/'candidate-c0-results')
for name in ['checked-top1-a.log','synchronized-c.log','synchronized-validation.json','topk-original-skew.log','topk-fixed-skew.log']:
 shutil.copyfile(source/name,rejected/name)
private=[]
for p in target.rglob('*'):
 if p.is_file():
  try:
   if '/Users/' in p.read_text():private.append(str(p.relative_to(target)))
  except UnicodeDecodeError:pass
assert not private,private
write(target/'artifact-hashes.json',{'schema':'rust-star-report-artifacts-v1','original_private_plan_sha256':sha(w/'inputs/plan.json'),'note':'plan-template is a portable rendering, not the original frozen plan; all other copied evidence retains original bytes','artifacts':{str(p.relative_to(target)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(target.rglob('*')) if p.is_file()}})
print(target)
