from pathlib import Path
import hashlib,json,math,statistics
w=Path(__file__).resolve().parent
read=lambda p:json.loads((w/p).read_text())
sha=lambda p:hashlib.sha256((w/p).read_bytes()).hexdigest()
plan=read('plan.json');parent=read('parent-shape-audit.json');words=4096*64*512
assert sha('parent-shape-audit.json')==plan['parent_aa_audit_sha256']
med=statistics.median
results=[];total=0
for session,order in enumerate(plan['shape_orders']):
 assert read(f'session-{session}/status.json')['exit_code']==0
 rows=[json.loads(l) for l in (w/f'session-{session}/probe.jsonl').read_text().splitlines()]
 h=rows.pop(0)
 assert h==dict(schema='rust-star-compressed-skip-runs-v1',session=session,device='Apple M1 Ultra',rows=4096,heads=64,width=512,pairs_per_protocol_per_shape=12,conditioning_dispatches=4,samples_per_warmed_slot=3,aa_uses_same_pipeline_object=True)
 compared=0
 for case in order:
  old=parent['shapes'][case];start=4096*(case+1);keys=4224+(start+4096)//128
  s=rows.pop(0);assert (s['event'],s['case'],s['chunk_start'],s['key_rows'],s['padded'])==('shape',case,start,keys,keys%64!=0)
  assert s['cpu_gpu_preparation_exact'] and s['mask_blocks']==old['mask_blocks'] and s['output_words']==words
  checksum=old['output_fnv1a_u32']
  for warm in range(2):
   t=rows.pop(0);assert t==dict(event='warmup',case=case,repetition=warm+1,order='AB' if warm==0 else 'BA',exact=True,checked_words=words,fnv1a_u32=checksum);compared+=words
  timings={k:[] for k in plan['protocols']}
  if case in plan['timed_cases']:
   for pair in range(12):
    for slot in range(2):
     t=rows.pop(0);protocol='candidate' if (pair+session+slot)%2 else 'aa-control'
     assert (t['event'],t['case'],t['pair'],t['protocol'],t['order'],t['a_buffer'])==('pair',case,pair+1,protocol,plan['pair_orders'][pair],plan['a_buffers'][pair])
     assert t['exact'] and t['checked_words']==words and t['fnv1a_u32']==checksum;compared+=words
     cfg=plan['protocols'][protocol];a,b=t['timing'];assert len(t['timing'])==2
     for role,r in enumerate(t['timing']):
      assert r['candidate_variant']==(protocol=='candidate' and role==1)
      commands=r['commands'];assert len(commands)==cfg['warm']+cfg['samples']
      assert math.isfinite(r['host_begin']) and r['host_end']>r['host_begin']
      last=0
      for i,c in enumerate(commands):
       assert c['conditioning']==(i<cfg['warm'])
       assert all(math.isfinite(c[k]) for k in ['start','end','gpu_ms']) and last<=c['start']<c['end']
       assert abs((c['end']-c['start'])*1000-c['gpu_ms'])<1e-6
       last=c['end']
      expected=sum(c['gpu_ms'] for c in commands[cfg['warm']:])/cfg['samples']
      assert abs(expected-r['mean_gpu_ms'])<1e-9
      assert r['mean_gpu_ms']==t['a_gpu_ms' if role==0 else 'b_gpu_ms']
     first,second=(a,b) if t['order']=='AB' else (b,a)
     assert first['host_end']<=second['host_begin'] and first['commands'][-1]['end']<=second['commands'][0]['start']
     assert abs(t['b_gpu_ms']/t['a_gpu_ms']-t['ratio'])<1e-12
     timings[protocol].append(t)
  s=rows.pop(0);assert s==dict(event='shape_complete',case=case,inputs_unchanged=True,guards_exact=True)
  for protocol,pairs in timings.items():
   if not pairs:continue
   fo=[t['a_gpu_ms']/t['b_gpu_ms'] if t['order']=='AB' else t['b_gpu_ms']/t['a_gpu_ms'] for t in pairs]
   ph=[t['a_gpu_ms']/t['b_gpu_ms'] if t['a_buffer']=='a' else t['b_gpu_ms']/t['a_gpu_ms'] for t in pairs]
   ab=[t['ratio'] for t in pairs if t['order']=='AB'];ba=[t['ratio'] for t in pairs if t['order']=='BA']
   assert len(ab)==len(ba)==6
   ratio=med(fo);mad=med(abs(v-ratio) for v in fo)
   stats=dict(session=session,case=case,chunk_start=start,protocol=protocol,pairs=12,first_second_median=ratio,first_second_mad=mad,ab_role_ratio_median=med(ab),ba_role_ratio_median=med(ba),physical_a_b_median=med(ph),pairs_within_3_percent=sum(abs(v-1)<=.03 for v in fo),minimum_first_second=min(fo),maximum_first_second=max(fo),role_ratio_median=med(t['ratio'] for t in pairs),a_median_gpu_ms=med(t['a_gpu_ms'] for t in pairs),b_median_gpu_ms=med(t['b_gpu_ms'] for t in pairs))
   rule=plan['null_screen']['per_shape_per_session']
   gates=dict(first_second=abs(ratio-1)<=rule['median_first_over_second_max_abs_deviation'],ab_ba=max(abs(med(ab)-1),abs(med(ba)-1))<=rule['ab_and_ba_role_ratio_medians_max_abs_deviation'],physical_buffer=abs(med(ph)-1)<=rule['physical_a_over_b_median_max_abs_deviation'],mad=mad<=rule['first_over_second_ratio_mad_max'],pair_dispersion=stats['pairs_within_3_percent']>=rule['minimum_pairs_within_3_percent'])
   speed=plan['candidate_screen']['per_shape_per_session']
   speed_gates=dict(ab_median=med(ab)<speed['ab_and_ba_median_ratio_strictly_below'],ba_median=med(ba)<speed['ab_and_ba_median_ratio_strictly_below'],ab_faster=sum(v<1 for v in ab)>=speed['minimum_faster_pairs_per_order'],ba_faster=sum(v<1 for v in ba)>=speed['minimum_faster_pairs_per_order'])
   results.append(dict(**stats,gates=gates if protocol=='aa-control' else speed_gates,null_screen_pass=all(gates.values()) if protocol=='aa-control' else None,candidate_screen_pass=all(speed_gates.values()) if protocol=='candidate' else None,ab_faster_pairs=sum(v<1 for v in ab),ba_faster_pairs=sum(v<1 for v in ba)))
 assert rows==[dict(event='complete',passed=True,cases=7,timed_pairs=72,compared_fp32_words=86*words)]
 assert compared==86*words;total+=compared
null_pass=all(r['null_screen_pass'] for r in results if r['protocol']=='aa-control')
candidate_pass=all(r['candidate_screen_pass'] for r in results if r['protocol']=='candidate')
result=dict(passed=True,correctness_passed=True,fp32_words_compared=total,sessions=2,total_pairs=144,contemporary_null_screen_pass=null_pass,candidate_speed_criteria_pass=candidate_pass,candidate_selected_for_later_c0=null_pass and candidate_pass,statistics=results,runtime_changed=False,engine_speedup_claim=False,files={p:sha(p) for p in ['plan.json','probe.m','parent-shape-audit.json','session-0/probe.jsonl','session-1/probe.jsonl']})
if (w/'audit.json').exists():assert read('audit.json')==result,'Stored audit differs'
else:(w/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
