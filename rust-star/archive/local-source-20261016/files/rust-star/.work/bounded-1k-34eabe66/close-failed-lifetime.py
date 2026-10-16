from pathlib import Path
import json,hashlib,datetime,subprocess
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan=read(w/'plan.json');build=read(w/'candidate-build.json');a=read(w/'diagnostic-failure-analysis.json')
assert a['structure_passed'] and not a['criteria_pass'] and read(w/'input-after.json')['passed']
assert read(w/'validation-acceptance.json')['passed'] and read(w/'lifetime-validation-acceptance.json')['passed']
assert not (w/'engine').exists() and not (w/'diagnostic-selection.json').exists()
assert all(sha(r/n)==h for n,h in build['sources'].items())
assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in {'candidate-rust-star','diagnostic-rust-star','baseline-rust-star','rust-star','ds4','ds4-bench'}]
for n in ['metal.rs','long_prefill.rs','metal_shim.m']:(r/'rust-star/runtime/src'/n).write_bytes((w/('baseline-'+n)).read_bytes())
sources={n:sha(r/n) for n in plan['source_reference']['sources']};assert sources==plan['source_reference']['sources'];default=sha(r/'rust-star/.work/runtime-target/release/rust-star');assert default==sha(w/'prior-default-rust-star')=='2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4';utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (w/'final-runtime-state.json').open('x') as f:json.dump(dict(passed=True,utc=utc,sources=sources,default_executable_sha256=default,prior_default_executable_sha256=default,candidate_retained=False,gpu_runs_complete=True),f,indent=2)
with (w/'decision.json').open('x') as f:json.dump(dict(closed=True,utc=utc,retained=False,reason='Failed the predeclared scratch-lifetime prerequisite despite full C0 and at-most-one unfinished command. Engine study did not start; restore accepted runtime without relaxing gates.',engine_runs=0,full_fp32_words=50031360,inference_repeats=0,stop_after_milestone=True),f,indent=2)
print('Rejected at lifetime prerequisite; 42 accepted sources restored; default unchanged; no engine runs.')
