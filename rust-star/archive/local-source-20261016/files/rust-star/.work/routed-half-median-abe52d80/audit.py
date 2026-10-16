from pathlib import Path
import hashlib,json,math,statistics,sys
base=Path(__file__).resolve().parent
mode=sys.argv[1];assert mode in ['aa-only','timing']
w=base/mode
aaOnly=mode=='aa-only'
read=lambda p:json.loads((w/p).read_text())
sha=lambda p:hashlib.sha256((w/p).read_bytes()).hexdigest()
plan=json.loads((base/'plan.json').read_text());words=4096*6*2048
checksums={};input_hashes={};diagnostics=[]
med=statistics.median
results=[];total=0;candidate_ratios=[]
for session,order in enumerate(plan['shape_orders']):
 assert read(f'session-{session}/status.json')['exit_code']==0
 rows=[json.loads(l) for l in (w/f'session-{session}/probe.jsonl').read_text().splitlines()]
 h=rows.pop(0)
 assert h==dict(schema=plan['probe_schema'],session=session,device='Apple M1 Ultra',mode=mode,same_arena_roles=True,conditioning_dispatches=8,samples_per_warmed_slot=12,slot_estimator='median',aa_uses_same_pipeline_objects=True,argument_sizes=[48,104,48])
 compared=0
 for case in order:
  shape=rows.pop(0);assert (shape['event'],shape['case'],shape['rows'],shape['seed'],shape['pattern'])==('shape',case,4096,plan['case_seeds'][case],case)
  assert shape['output_fp16_words']==words and shape['half_scratch_bytes']==plan['shape']['extra_half_bytes']
  if session:assert shape['input_fnv1a_u32']==input_hashes[case]
  else:input_hashes[case]=shape['input_fnv1a_u32']
  checksum=checksums.get(case)
  for warm in range(2):
   t=rows.pop(0)
   if checksum is None:checksum=t['fnv1a_u16'];assert isinstance(checksum,int) and 0<=checksum<2**64
   assert t==dict(event='warmup',case=case,repetition=warm+1,order='AB' if warm==0 else 'BA',exact=True,cpu_half_conversion_exact=True,checked_fp16_words=words,fnv1a_u16=checksum);compared+=words
  checksums[case]=checksum
  timings={k:[] for k in (['aa-control'] if aaOnly else plan['protocols'])}
  if case in plan['timed_cases']:
   for pair in range(12):
    for slot in range(1 if aaOnly else 2):
     t=rows.pop(0);protocol='candidate' if not aaOnly and (pair+session+slot)%2 else 'aa-control'
     assert (t['event'],t['case'],t['pair'],t['protocol'],t['order'],t['active_buffer'])==('pair',case,pair+1,protocol,plan['pair_orders'][pair],plan['a_buffers'][pair])
     assert t['per_role_verified'] and t['reference_buffer']!=t['active_buffer']
     assert t['exact'] and t['checked_fp16_words']==2*words and t['fnv1a_u16']==checksum;compared+=2*words
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
      expected=med(c['gpu_ms'] for c in commands[cfg['warm']:])
      assert abs(sum(c['gpu_ms'] for c in commands[cfg['warm']:])/cfg['samples']-r['arithmetic_mean_gpu_ms'])<1e-9
      assert abs(expected-r['median_gpu_ms'])<1e-9
      assert r['median_gpu_ms']==t['a_gpu_ms' if role==0 else 'b_gpu_ms']
     first,second=(a,b) if t['order']=='AB' else (b,a)
     assert first['host_end']<=second['host_begin'] and first['commands'][-1]['end']<=second['commands'][0]['start']
     assert abs(t['b_gpu_ms']/t['a_gpu_ms']-t['ratio'])<1e-12
     timings[protocol].append(t)
     if protocol=='candidate':candidate_ratios.append(t['ratio'])
  s=rows.pop(0);assert s==dict(event='shape_complete',case=case,inputs_unchanged=True,guards_exact=True,cpu_gpu_map_exact=True)
  for protocol,pairs in timings.items():
   if not pairs:continue
   fo=[t['a_gpu_ms']/t['b_gpu_ms'] if t['order']=='AB' else t['b_gpu_ms']/t['a_gpu_ms'] for t in pairs]
   arena_medians={ar:med(t['ratio'] for t in pairs if t['active_buffer']==ar) for ar in ['a','b']}
   ab=[t['ratio'] for t in pairs if t['order']=='AB'];ba=[t['ratio'] for t in pairs if t['order']=='BA']
   assert len(ab)==len(ba)==6
   ratio=med(fo);mad=med(abs(v-ratio) for v in fo)
   stats=dict(session=session,case=case,rows=4096,protocol=protocol,pairs=12,first_second_median=ratio,first_second_mad=mad,ab_role_ratio_median=med(ab),ba_role_ratio_median=med(ba),active_arena_role_medians=arena_medians,pairs_within_3_percent=sum(abs(v-1)<=.03 for v in fo),minimum_first_second=min(fo),maximum_first_second=max(fo),role_ratio_median=med(t['ratio'] for t in pairs),a_median_gpu_ms=med(t['a_gpu_ms'] for t in pairs),b_median_gpu_ms=med(t['b_gpu_ms'] for t in pairs))
   rule=plan['neutral_screen']['per_shape_per_session']
   gates=dict(first_second=abs(ratio-1)<=rule['median_first_over_second_max_abs_deviation'],ab_ba=max(abs(med(ab)-1),abs(med(ba)-1))<=rule['ab_and_ba_role_ratio_medians_max_abs_deviation'],active_arena=max(abs(v-1) for v in arena_medians.values())<=rule['active_arena_subgroup_role_ratio_max_abs_deviation'],mad=mad<=rule['first_over_second_ratio_mad_max'],pair_dispersion=stats['pairs_within_3_percent']>=rule['minimum_pairs_within_3_percent'])
   speed=plan['candidate_selection']
   speed_gates=dict(ab_median=med(ab)<speed['every_case_session_order_median_ratio_below'],ba_median=med(ba)<speed['every_case_session_order_median_ratio_below'])
   results.append(dict(**stats,gates=gates if protocol=='aa-control' else speed_gates,null_screen_pass=all(gates.values()) if protocol=='aa-control' else None,candidate_screen_pass=all(speed_gates.values()) if protocol=='candidate' else None,ab_faster_pairs=sum(v<1 for v in ab),ba_faster_pairs=sum(v<1 for v in ba)))
 assert rows==[dict(event='complete',passed=True,cases=3,timed_pairs=36 if aaOnly else 72,compared_fp16_words=(78 if aaOnly else 150)*words)]
 assert compared==(78 if aaOnly else 150)*words;total+=compared
null_pass=all(r['null_screen_pass'] for r in results if r['protocol']=='aa-control')
candidate_pass=not aaOnly and all(r['candidate_screen_pass'] for r in results if r['protocol']=='candidate') and med(candidate_ratios)<plan['candidate_selection']['overall_median_ratio_below']
if not aaOnly:
 prior=json.loads((base/'aa-only/audit.json').read_text());assert prior['contemporary_null_screen_pass'];assert prior['output_checksums']=={str(k):v for k,v in checksums.items()}
result=dict(passed=True,correctness_passed=True,overall_candidate_median=med(candidate_ratios) if candidate_ratios else None,fp16_words_compared=total,sessions=2,total_pairs=72 if aaOnly else 144,contemporary_null_screen_pass=null_pass,candidate_speed_criteria_pass=candidate_pass,candidate_selected_for_later_c0=null_pass and candidate_pass,statistics=results,diagnostics=diagnostics,output_checksums={str(k):v for k,v in checksums.items()},runtime_changed=False,engine_speedup_claim=False,files={p:sha(p) for p in ['session-0/probe.jsonl','session-1/probe.jsonl']},protocol_inputs={p:hashlib.sha256((base/p).read_bytes()).hexdigest() for p in ['plan.json','probe.m','edge-audit.json']})
if (w/'audit.json').exists():assert read('audit.json')==result,'Stored audit differs'
else:(w/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
