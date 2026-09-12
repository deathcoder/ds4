from pathlib import Path
import json,statistics,hashlib
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());b=read(w.parent/'profile-audit.json');c=read(w/'profile-audit.json');plan=read(w/'plan.json')['diagnostic_selection']
assert b['passed'] and c['passed'] and read(w/'validation-acceptance.json')['passed']
target=plan['target'];bt=b['stages'][target]['gpu_ms'];ct=c['stages'][target]['gpu_ms'];bc=sum(b['stages'][n]['gpu_ms'] for n in plan['controls']);cc=sum(c['stages'][n]['gpu_ms'] for n in plan['controls']);raw=ct/bt;normalized=(ct/cc)/(bt/bc)
r=dict(criteria_pass=raw<plan['raw_aggregate_ratio_below'] and normalized<plan['control_normalized_aggregate_ratio_below'],baseline_target_ms=bt,candidate_target_ms=ct,raw_aggregate_ratio=raw,control_normalized_aggregate_ratio=normalized,baseline_controls_ms=bc,candidate_controls_ms=cc,controls=plan['controls'],baseline_profile_audit_sha256=hashlib.sha256((w.parent/'profile-audit.json').read_bytes()).hexdigest(),candidate_profile_audit_sha256=hashlib.sha256((w/'profile-audit.json').read_bytes()).hexdigest(),engine_speedup_claim=False)
p=w/'diagnostic-selection.json'
if p.exists():assert read(p)==r
else:p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
