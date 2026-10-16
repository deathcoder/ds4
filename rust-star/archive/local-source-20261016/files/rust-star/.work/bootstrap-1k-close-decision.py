from pathlib import Path
import json,hashlib,subprocess,datetime,os
r=Path.cwd();w=r/'rust-star/.work/bootstrap-1k-current';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
assert read(w/'input-after.json')['passed'];a=read(w/'engine/audit.json');assert a['execution_passed'] and a['raw_runs']==12 and a['timed_pairs']==5;assert read(w/'passive-phase-audit.json')['passed'];b=read(w/'candidate-build.json');plan=read(w/'plan.json');assert all(sha(r/n)==h for n,h in b['sources'].items());retained=a['candidate_retained_as_engine_improvement'];assert retained==all(a['gates'].values());default=r/'rust-star/.work/runtime-target/release/rust-star';old_default=sha(default);assert old_default=='2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4'
if retained:
 tmp=default.with_name('rust-star.new-1k');assert not tmp.exists();subprocess.run(['cp','-c',str(w/'candidate-rust-star'),str(tmp)],check=True);assert sha(tmp)==b['executable_sha256'];tmp.chmod(0o555);os.replace(tmp,default);expected=b['sources']
else:
 for n in read(w/'source-edits.json'):(r/n).write_bytes((w/('baseline-'+Path(n).name)).read_bytes())
 expected=plan['source_reference']['sources']
sources={n:sha(r/n) for n in expected};assert sources==expected
state=dict(passed=True,utc=utc(),sources=sources,default_executable_sha256=sha(default),prior_default_executable_sha256=old_default,candidate_retained=retained)
with (w/'final-runtime-state.json').open('x') as f:f.write(json.dumps(state,indent=2)+'\n')
failed=[n for n,p in a['gates'].items() if not p];reason='Every predeclared engine and memory gate passes.' if retained else 'Failed predeclared gates: '+', '.join(failed)+'. Correctness passes, but the candidate does not qualify for retention.'
with (w/'decision.json').open('x') as f:f.write(json.dumps(dict(closed=True,utc=utc(),retained=retained,reason=reason,failed_gates=failed),indent=2)+'\n')
print(json.dumps(dict(retained=retained,failed_gates=failed,source_files_verified=len(sources),default_executable_sha256=state['default_executable_sha256']),indent=2))
