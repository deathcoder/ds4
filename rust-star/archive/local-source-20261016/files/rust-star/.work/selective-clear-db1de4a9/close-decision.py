from pathlib import Path
import json,hashlib,subprocess,datetime,os
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','diagnostic-rust-star','baseline-rust-star','ds4','ds4-bench','ds4-server'}]
assert read(w/'input-after.json')['passed'];b=read(w/'candidate-build.json');plan=read(w/'plan.json');assert all(sha(r/n)==h for n,h in b['sources'].items())
engine=w/'engine/audit.json';retained=False;failed=[];runs=0
if engine.exists():
 a=read(engine);assert a['execution_passed'] and a['raw_runs']==12 and a['timed_pairs']==5;assert read(w/'passive-phase-audit.json')['passed'];assert read(w/'diagnostic-selection.json')['criteria_pass'];retained=a['candidate_retained_as_engine_improvement'];assert retained==all(a['gates'].values());failed=[n for n,v in a['gates'].items() if not v];runs=a['raw_runs'];reason='Every unchanged engine and memory gate passes.' if retained else 'Failed unchanged engine gates: '+', '.join(failed)
else:
 assert (w/'lifetime-failure.json').exists() or (w/'failure.json').exists() or (w/'engine/failure.json').exists()
 failed=['validation_or_protocol_prerequisite'];reason='Prerequisite or execution protocol failed; preserve the recorded failure and restore accepted runtime.'
 if (w/'engine/progress.json').exists():runs=len(read(w/'engine/progress.json'))
default=r/'rust-star/.work/runtime-target/release/rust-star';old=sha(default);assert old=='2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4';assert sha(w/'prior-default-rust-star')==old
if retained:
 tmp=default.with_name('rust-star.new-selective-clear');assert not tmp.exists();subprocess.run(['cp','-c',str(w/'candidate-rust-star'),str(tmp)],check=True);assert sha(tmp)==b['executable_sha256'];tmp.chmod(0o555);os.replace(tmp,default);expected=b['sources']
else:
 for n in ['metal.rs','long_prefill.rs','metal_shim.m']:(r/'rust-star/runtime/src'/n).write_bytes((w/('baseline-'+n)).read_bytes())
 expected=plan['source_reference']['sources']
sources={n:sha(r/n) for n in expected};assert sources==expected
state=dict(passed=True,utc=utc(),sources=sources,default_executable_sha256=sha(default),prior_default_executable_sha256=old,candidate_retained=retained)
with (w/'final-runtime-state.json').open('x') as f:f.write(json.dumps(state,indent=2)+'\n')
with (w/'decision.json').open('x') as f:f.write(json.dumps(dict(closed=True,utc=utc(),retained=retained,reason=reason,failed_gates=failed,engine_runs=runs),indent=2)+'\n')
print(json.dumps(dict(retained=retained,failed_gates=failed,source_files_verified=len(sources),default_executable_sha256=state['default_executable_sha256']),indent=2))
