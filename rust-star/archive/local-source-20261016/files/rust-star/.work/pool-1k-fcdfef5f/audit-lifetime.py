from pathlib import Path
import json,re
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());p=read(w/'plan.json');log=(w/'candidate-lifetime.log').read_text();rows=[json.loads(x[12:]) for x in log.splitlines() if x.startswith('RS_LIFETIME ')];checkouts=[json.loads(x[len('RS_POOL_CHECKOUT '):]) for x in log.splitlines() if x.startswith('RS_POOL_CHECKOUT ')];owners=[tuple(map(int,x.split()[1:])) for x in log.splitlines() if x.startswith('RS_POOL_OWNER ')];assert len(rows)==119 and len(checkouts)==2940 and len(owners)==28;assert 'RS_LIFETIME_ERROR' not in log
names=p['pool_contract']['labels'];caps={x['frontier']:x['one_set_bytes'] for x in p['lifetime_reference']};results=[];peak=0;maxunfinished=0
for frontier in range(8192,32769,4096):
 frows=[x for x in rows if x['frontier']==frontier]; assert len(frows)==17
 expected_cap=caps[frontier]
 if isinstance(expected_cap,dict):expected_cap=expected_cap['scratch_bytes']
 identity=None;owner=None;firstlabels=None
 for i,pos in enumerate(range(frontier-4096,frontier,1024)):
  rs=[x for x in frows if x['position']==pos];assert [x['phase'] for x in rs]==(['entry','registered','submitted','waited','after_pool'] if i==3 else ['entry','registered','submitted','after_pool'])
  cs=[x for x in checkouts if x['frontier']==frontier and x['position']==pos];assert len(cs)==105 and sorted(x['label'] for x in cs)==names
  ids={x['label']:x['identity'] for x in cs};assert len(set(ids.values()))==105
  assert len(set(x['owner'] for x in cs))==1
  if i==0:identity=ids;owner=cs[0]['owner'];firstlabels={x['label']:x['bytes'] for x in cs};assert rs[0]['scratch_bytes']==0
  else:assert identity==ids and owner==cs[0]['owner'];assert rs[0]['scratch_bytes']==expected_cap
  assert all(x['prior_status']==(-1 if i==0 else 4) for x in cs)
  assert all(x['copy']==(x['label'] in ['token_buffer','position_buffer','compressed_position_buffer']) for x in cs)
  assert {x['label']:x['bytes'] for x in cs}==firstlabels
  assert sum(x['bytes'] for x in cs)==expected_cap<=2493173233
  assert cs[-1]['pool_count']==105 and cs[-1]['pool_bytes']==expected_cap
  assert all(x['pool_count']<=105 and x['pool_bytes']<=expected_cap for x in cs)
  assert (pos,frontier,int(i!=3)) in owners
  for row in rs:
   assert row['rows']==1024 and row['vm_result']==0
   statuses=row['command_status_counts'];assert len(statuses)==7 and sum(statuses)==row['commands'] and statuses[5:]==[0,0]
   unfinished=sum(statuses[:4]);assert unfinished<=1;maxunfinished=max(maxunfinished,unfinished)
   peak=max(peak,row['scratch_bytes']);assert row['scratch_bytes']<=expected_cap
   for label,v in row['labels'].items():
    if not v['persistent']:assert label in names and v['count']==1
   if row['phase'] in ['registered','submitted','waited']:
    assert row['scratch_bytes']==expected_cap
    assert {k:v['bytes'] for k,v in row['labels'].items() if not v['persistent']}==firstlabels
  # Owner removal is the production contract at final cleanup; this records
  # weak liveness separately without claiming immediate Metal destruction.
 results.append(dict(frontier=frontier,bytes=expected_cap,labels=105,tiles=4,final_weak_scratch_bytes=frows[-1]['scratch_bytes']))
assert read(w/'lifetime-validation-acceptance.json')['passed'];assert read(w/'validation-acceptance.json')['passed'];assert read(w/'profile-audit.json')['passed'] if (w/'profile-audit.json').exists() else True
out=dict(criteria_pass=True,chunks=results,tiles=28,snapshots=len(rows),checkouts=len(checkouts),owner_cleanup_events=len(owners),max_unfinished_commands=maxunfinished,max_sampled_scratch_bytes=peak,full_fp32_words=16677120)
(w/'diagnostic-selection.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
