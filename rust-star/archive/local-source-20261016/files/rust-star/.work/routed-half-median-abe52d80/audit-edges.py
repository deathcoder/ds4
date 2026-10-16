from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;plan=json.loads((w/'plan.json').read_text());p=w/'edges/probe.jsonl';records=[json.loads(l) for l in p.read_text().splitlines()];header=records.pop(0)
assert header==dict(schema=plan['probe_schema'],device='Apple M1 Ultra',mode='edges',session=0,same_arena_roles=True,aa_uses_same_pipeline_objects=True,conditioning_dispatches=8,samples_per_warmed_slot=12,slot_estimator='median',argument_sizes=[48,104,48])
assert json.loads((w/'edges/status.json').read_text())['exit_code']==0
compared=0;cases=[]
for case in range(16):
 shape=records.pop(0);rows=plan['edge_rows'][case//2];words=rows*6*2048
 assert (shape['event'],shape['case'],shape['rows'],shape['seed'],shape['pattern'])==('shape',case,rows,0x5417311+case,case%2)
 assert shape['output_fp16_words']==words and shape['half_scratch_bytes']==rows*4096*2
 checksum=None
 for warm in range(2):
  r=records.pop(0)
  if checksum is None:checksum=r['fnv1a_u16'];assert isinstance(checksum,int) and 0<=checksum<2**64
  assert r==dict(event='warmup',case=case,repetition=warm+1,order='AB' if warm==0 else 'BA',exact=True,cpu_half_conversion_exact=True,checked_fp16_words=words,fnv1a_u16=checksum);compared+=words
 assert records.pop(0)==dict(event='shape_complete',case=case,inputs_unchanged=True,guards_exact=True,cpu_gpu_map_exact=True)
 cases.append(dict(case=case,rows=rows,pattern=case%2,input_fnv1a_u32=shape['input_fnv1a_u32'],output_fnv1a_u16=checksum))
assert records==[dict(event='complete',passed=True,cases=16,timed_pairs=0,compared_fp16_words=compared)]
r=dict(passed=True,cases=cases,fp16_words_compared=compared,all_guards_cpu_maps_and_half_conversions_exact=True,files={n:hashlib.sha256((w/n).read_bytes()).hexdigest() for n in ['edges/probe.jsonl','edges/status.json','plan.json','probe.m','candidate.metal','combined.metal']});out=w/'edge-audit.json'
if out.exists():assert json.loads(out.read_text())==r
else:out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(passed=True,cases=16,fp16_words_compared=compared),indent=2))
