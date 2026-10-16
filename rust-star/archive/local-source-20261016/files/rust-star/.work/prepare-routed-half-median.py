from pathlib import Path
import json,hashlib,subprocess,datetime,shutil,ast
root=Path.cwd();prior=root/'rust-star/.work/routed-half-prefill-c7adfb22';w=root/'rust-star/.work/routed-half-median-abe52d80';read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();assert read(prior/'milestone-completion.json')['passed'];w.mkdir(exist_ok=False)
plan=read(prior/'plan.json');plan['schema']='rust-star-routed-half-median-protocol-v2';plan['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();plan['probe_schema']='rust-star-routed-half-median-same-arena-v2';plan['conditioning_dispatches']=8;plan['samples_per_warmed_slot']=12;plan['protocols']={k:dict(warm=8,samples=12) for k in plan['protocols']};plan['slot_estimator']='median';plan['preparation_only_until']='Prior failed neutral screen is completely reported and closed; no overlapping GPU work.';plan['protocol_change']=dict(prior_commit=read(prior/'milestone-completion.json')['local_commit'],prior_manifest_sha256=read(prior/'milestone-completion.json')['manifest_sha256'],reason='Identical-kernel mean-of-three slots contain occasional large duration spikes and fail five of six neutral cells.',changes=['8 conditioning chains instead of 4','12 measured chains instead of 3','Median slot estimate; preserve every raw command and arithmetic mean as descriptive data'],thresholds_unchanged=True,no_reclassification_of_prior_data=True);(w/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');edits={};identities={}
def change(n,ops):
 original=(prior/n).read_text();s=original;records=[]
 for a,b in ops:
  count=s.count(a);assert count>0,(n,a);records.append(dict(before=a,after=b,count=count));s=s.replace(a,b)
 inverse=s
 for e in reversed(records):assert inverse.count(e['after'])==e['count'],(n,e['after']);inverse=inverse.replace(e['after'],e['before'])
 assert inverse==original;(w/n).write_text(s);edits[n]=records;identities[n]=dict(before_sha256=sha(original.encode()),after_sha256=sha(s.encode()),inverse_exact=True)
change('probe.m',[
 ('static NSDictionary*slot(', 'static int compare_double(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y); }\nstatic NSDictionary*slot('),
 ('int warm=4,samples=3;', 'int warm=8,samples=12;'),
 ('total=0;NSMutableArray*raw=', 'total=0;double measured[12];NSMutableArray*raw='),
 ('if(i>=warm)total+=ms;', 'if(i>=warm){total+=ms;measured[i-warm]=ms;}'),
 (' return @{@"candidate_variant"', ' qsort(measured,12,sizeof(double),compare_double);double estimate=(measured[5]+measured[6])/2.0;\n return @{@"candidate_variant"'),
 ('@"mean_gpu_ms"', '@"median_gpu_ms"'),
 ('@"median_gpu_ms":@(total/samples)', '@"median_gpu_ms":@(estimate),@"arithmetic_mean_gpu_ms":@(total/samples)'),
 ('rust-star-routed-half-same-arena-v1','rust-star-routed-half-median-same-arena-v2'),
 ('@"conditioning_dispatches":@4,@"samples_per_warmed_slot":@3', '@"conditioning_dispatches":@8,@"samples_per_warmed_slot":@12,@"slot_estimator":@"median"')])
change('audit.py',[
 ('conditioning_dispatches=4,samples_per_warmed_slot=3,', "conditioning_dispatches=8,samples_per_warmed_slot=12,slot_estimator='median',"),
 ("expected=sum(c['gpu_ms'] for c in commands[cfg['warm']:])/cfg['samples']", "expected=med(c['gpu_ms'] for c in commands[cfg['warm']:])\n      assert abs(sum(c['gpu_ms'] for c in commands[cfg['warm']:])/cfg['samples']-r['arithmetic_mean_gpu_ms'])<1e-9"),
 ("'mean_gpu_ms'","'median_gpu_ms'")])
change('audit-edges.py', [('conditioning_dispatches=4,samples_per_warmed_slot=3,',"conditioning_dispatches=8,samples_per_warmed_slot=12,slot_estimator='median',")])
change('run.py',[
 ("prior=root/'rust-star/.work/autonomous-prefill-a46dfe8b/wide-bootstrap-router-bound'", "prior=root/'rust-star/.work/routed-half-prefill-c7adfb22'"),
 ("assert read(prior/'completion-pins.json')['passed'] and read(prior/'engine/audit.json')['execution_passed']", "assert read(prior/'execution-complete.json')['passed'] and not read(prior/'execution-complete.json')['selected']\nassert not (prior/'timing').exists()"),
 ("'plan.json','prepare.py','prepare-audits.py'", "'plan.json','prepare.py','prepare-audits.py','prepare-protocol.py','protocol-source-edits.json','protocol-derivation.json'")])
(w/'protocol-source-edits.json').write_text(json.dumps(edits,indent=2)+'\n');(w/'protocol-derivation.json').write_text(json.dumps(dict(prior=str(prior.relative_to(root)),files=identities),indent=2)+'\n')
for n in ['prepare.py','integration-notes.md','scratch-lifetime-observation.json']:shutil.copyfile(prior/n,w/n)
shutil.copyfile(Path(__file__),w/'prepare-protocol.py')
(w/'prepare-audits.py').write_text('''from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;root=w.parents[2];prior=root/'rust-star/.work/routed-half-prefill-c7adfb22'
edits=json.loads((w/'protocol-source-edits.json').read_text());proof=json.loads((w/'protocol-derivation.json').read_text())
for n in ['audit.py','audit-edges.py']:
 s=(prior/n).read_text();assert hashlib.sha256(s.encode()).hexdigest()==proof['files'][n]['before_sha256']
 for e in edits[n]:assert s.count(e['before'])==e['count'];s=s.replace(e['before'],e['after'])
 assert s==(w/n).read_text() and hashlib.sha256(s.encode()).hexdigest()==proof['files'][n]['after_sha256']
(w/'audit-derivation.json').write_text(json.dumps({n:proof['files'][n] for n in ['audit.py','audit-edges.py']},indent=2)+'\\n')
''')
for n in ['audit.py','audit-edges.py','run.py','prepare-audits.py']:ast.parse((w/n).read_text())
subprocess.run(['python3',str(w/'prepare.py')],check=True);subprocess.run(['python3',str(w/'prepare-audits.py')],check=True)
assert (w/'candidate.metal').read_bytes()==(prior/'candidate.metal').read_bytes();assert (w/'combined.metal').read_bytes()==(prior/'combined.metal').read_bytes();assert plan['neutral_screen']==read(prior/'plan.json')['neutral_screen'] and plan['candidate_selection']==read(prior/'plan.json')['candidate_selection'];print('Prepared new fixed median protocol:',w)
