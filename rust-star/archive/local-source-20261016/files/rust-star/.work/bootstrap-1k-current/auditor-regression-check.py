from pathlib import Path
import json,hashlib,tempfile,shutil,subprocess,sys
r=Path.cwd();w=r/'rust-star/.work/bootstrap-1k-current';old=r/'rust-star/.work/autonomous-prefill-a46dfe8b/wide-bootstrap-router-bound';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as d:
 root=Path(d);p=root/'rust-star/.work/replay';p.mkdir(parents=True)
 for n in ['ruststar_measurement_lib.py','dwarfstar_measurement_lib.py','artifact_lib.py']:shutil.copyfile(r/'rust-star'/n,root/'rust-star'/n)
 shutil.copytree(old/'engine',p/'engine');(p/'engine/audit.json').unlink()
 for n in ['candidate-build.json','validation-acceptance.json']:shutil.copyfile(old/n,p/n)
 plan=read(old/'engine-plan.json');plan['retention_criteria']['median_peak_memory_footprint_ratio_at_most']=1.15;(p/'engine-plan.json').write_text(json.dumps(plan))
 shutil.copyfile(w/'engine-audit.py',p/'engine-audit.py');run=subprocess.run([sys.executable,str(p/'engine-audit.py')],capture_output=True,text=True);assert run.returncode==0,run.stderr
 result=read(p/'engine/audit.json');prior=read(old/'engine/audit.json')
 for n,v in prior['gates'].items():assert result['gates'][n]==v
 for n,v in prior['summary'].items():assert result['summary'][n]==v
 assert not result['gates']['peak_memory_footprint_median'];assert not result['candidate_retained_as_engine_improvement'];ratio=result['summary']['peak_memory_footprint_bytes']['paired_ratio']['median'];assert 1.52<ratio<1.53
 out=dict(passed=True,purpose='Auditor regression check on preserved rejected4K data only; no new inference or relabeling of the prior result.',auditor_sha256=sha(w/'engine-audit.py'),prior_audit_sha256=sha(old/'engine/audit.json'),all_prior_metrics_and_gates_unchanged=True,new_footprint_gate_rejects_known_regression=True,observed_old_footprint_ratio=ratio)
 (w/'auditor-regression-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
