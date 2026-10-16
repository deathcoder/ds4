from pathlib import Path
import hashlib,json,shutil
source=Path(__file__).resolve().parent;root=source.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
w=Path(read(source/'paired-location.json')['work']);assert read(w/'independent-audit.json')['passed'];assert read(w/'profile32k-analysis.json')['passed']
commit=read(source/'batch-build.json')['commit'];target=root/'rust-star/reports'/('32k-prefill-'+commit[:8]);assert not target.exists();target.mkdir(parents=True)
shutil.copytree(w/'results',target/'results')
for name in ['independent-audit.json','profile32k.json','profile32k-analysis.json','host-observations.jsonl','load-observations.jsonl','interruption.json','interrupted-state.json']:
 shutil.copyfile(w/name,target/name)
(target/'inputs').mkdir()
for p in (w/'inputs').glob('*.json'):
 if p.name!='plan.json':shutil.copyfile(p,target/'inputs'/p.name)
plan=read(w/'inputs/plan.json')
for role in ['candidate','oracle']:
 e=plan[role];e['working_directory']='$CANDIDATE_BUILD' if role=='candidate' else '$ORACLE_BUILD';args=e['adapter_command']
 for i,v in enumerate(args):
  if v.startswith(str(root)):args[i]='$REPO/'+v[len(str(root))+1:]
  elif v.startswith('/Users/'):args[i]='$MODEL' if i and args[i-1]=='--model' else '$LOCAL_PATH'
write(target/'inputs/plan-template.json',plan)
proof=target/'correctness';proof.mkdir()
for name in ['batch-build.json','batch-validation.json','batch-12k.json','batch-32k-output.json','batch-32k-internal.json','batch-2k.json','batch-8k.json','batch-retained.json','profile-before-build.json','profile-before-process.json','profile-before.json','profile-before-analysis.json','python-tests.log']:
 shutil.copyfile(source/name,proof/name)
original=source/'batch-tests-fixed.log';portable=proof/'rust-tests-portable.log';portable.write_text(original.read_text().replace(str(root),'$REPO').rstrip()+'\n')
baseline=root/'rust-star/reports/32k-baseline-cbb078bb';shutil.copyfile(baseline/'results/paired-summary.json',target/'baseline-summary.json');shutil.copyfile(baseline/'independent-audit.json',target/'baseline-audit.json');shutil.copyfile(baseline/'inputs/correctness-manifest.json',proof/'accepted-reference-correctness.json')
write(target/'portability.json',{'schema':'rust-star-report-portability-v1','derived_files':{'inputs/plan-template.json':{'original_sha256':sha(w/'inputs/plan.json'),'transformation':'local paths replaced with placeholders'},'correctness/rust-tests-portable.log':{'original_sha256':sha(original),'transformation':'repository prefix replaced with $REPO; trailing blank lines normalized'}},'reference_acceptance':'../32k-baseline-cbb078bb/correctness/oracle-v4-acceptance/'})
shutil.copyfile(baseline/'verify.py',target/'verify.py')
v=target/'verify.py';code=v.read_text();code=code.replace("assert len(raw['pairs']) == 5", "pairs = [p for p in raw['pairs'] if p['valid']]\ninvalid = [p for p in raw['pairs'] if not p['valid']]\nassert len(pairs) == 5 and len(invalid) == 1");code=code.replace("for pair in raw['pairs']", "for pair in pairs");code=code.replace("assert all(pair['valid'] and pair['attempt'] == 1 for pair in pairs)", "assert [p['attempt'] for p in pairs] == [1, 1, 1, 1, 2]");v.write_text(code)
with (target/'verify.py').open('a') as f:f.write((source/'verify-extension.py').read_text())
for p in target.rglob('*'):
 if p.is_file():
  try:assert '/Users/' not in p.read_text(),str(p)
  except UnicodeDecodeError:pass
write(source/'report-location.json',{'path':str(target)});print(target)
