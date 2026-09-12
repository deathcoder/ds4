from pathlib import Path
import json,hashlib,ast
w=Path(__file__).resolve().parent;root=w.parents[2];old=root/'rust-star/.work/attention-output-milestone-08ebd1aa/audit.py';s=old.read_text();source=s
replacements=[("words=4096*(8192+4096+16384)","words=4096*6*2048"),("schema=plan['schema']","schema=plan['probe_schema']"),("argument_sizes=[48,104,88,152]","argument_sizes=[48,104,48]"),("shape['precision_pattern']","shape['pattern']"),("plan['case_seeds'][case],False","plan['case_seeds'][case],case"),("shape['output_words']","shape['output_fp16_words']"),("shape['extra_half_scratch_bytes']==plan['extra_fp16_scratch_bytes']","shape['half_scratch_bytes']==plan['shape']['extra_half_bytes']"),("checksum=plan['reference_checksums'][str(case)]","checksum=checksums.get(case)"),("checksum=t['fnv1a_u32'];assert len(checksum)==3","checksum=t['fnv1a_u16'];assert isinstance(checksum,int) and 0<=checksum<2**64"),("fnv1a_u32","fnv1a_u16"),("checked_words","checked_fp16_words"),("compared_fp32_words","compared_fp16_words"),("fp32_words_compared","fp16_words_compared"),("plan['null_screen']","plan['neutral_screen']")]
# Avoid changing the independent input digest field to a 16-bit label.
for a,b in replacements:
 assert a in s,a;s=s.replace(a,b)
s=s.replace('input_fnv1a_u16','input_fnv1a_u32')
a=s.index("   speed=plan['candidate_screen']");b=s.index('   results.append(',a)
s=s[:a]+"   speed=plan['candidate_selection']\n   speed_gates=dict(ab_median=med(ab)<speed['every_case_session_order_median_ratio_below'],ba_median=med(ba)<speed['every_case_session_order_median_ratio_below'])\n"+s[b:]
s=s.replace("results=[];total=0","results=[];total=0;candidate_ratios=[]")
s=s.replace("     timings[protocol].append(t)","     timings[protocol].append(t)\n     if protocol=='candidate':candidate_ratios.append(t['ratio'])")
s=s.replace("candidate_pass=not aaOnly and all(r['candidate_screen_pass'] for r in results if r['protocol']=='candidate')", "candidate_pass=not aaOnly and all(r['candidate_screen_pass'] for r in results if r['protocol']=='candidate') and med(candidate_ratios)<plan['candidate_selection']['overall_median_ratio_below']\nif not aaOnly:\n prior=json.loads((base/'aa-only/audit.json').read_text());assert prior['contemporary_null_screen_pass'];assert prior['output_checksums']=={str(k):v for k,v in checksums.items()}")
s=s.replace("result=dict(passed=True,correctness_passed=True,", "result=dict(passed=True,correctness_passed=True,overall_candidate_median=med(candidate_ratios) if candidate_ratios else None,")
ast.parse(s);(w/'audit.py').write_text(s)
(w/'audit-derivation.json').write_text(json.dumps(dict(original_path=str(old.relative_to(root)),original_sha256=hashlib.sha256(source.encode()).hexdigest(),derived_sha256=hashlib.sha256(s.encode()).hexdigest(),description='Retain raw command/role/order/arena/neutral checks; adapt shape and all output words to routed FP16, bind stable cross-session/cross-mode reference checksums, and use the new prospectively declared candidate gates.'),indent=2)+'\n')
p=json.loads((w/'plan.json').read_text());p.update(probe_schema='rust-star-routed-half-same-arena-v1',shape_orders=[[0,1,2],[2,1,0]],case_seeds=[0x5417311+i for i in range(3)],a_buffers=['a' if (i//2)%2==0 else 'b' for i in range(12)],protocols={'aa-control':{'warm':4,'samples':3},'candidate':{'warm':4,'samples':3}},timed_cases=[0,1,2],edge_sessions=[0]);(w/'plan.json').write_text(json.dumps(p,indent=2)+'\n')
print('Prepared strict derived audit; no data generated')
