from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[3];read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert '320 passed; 0 failed' in (w/'tests.log').read_text()
assert 'Finished `release`' in (w/'build.log').read_text()
subprocess.run(['cargo','fmt','--check','--manifest-path',str(root/'rust-star/runtime/Cargo.toml')],cwd=root,stdout=(w/'fmt.log').open('x'),stderr=subprocess.STDOUT,check=True)
base=read(w.parent/'plan.json')['accepted_pins']['sources'];ops=read(w/'source-edits.json');sources={n:sha(root/n) for n in base}
for n,h in base.items():
 if n not in ops:assert sources[n]==h,n
for n,edits in ops.items():
 s=(root/n).read_text()
 for e in reversed(edits):assert s.count(e['after'])==1;(None);s=s.replace(e['after'],e['before'])
 assert hashlib.sha256(s.encode()).hexdigest()==base[n],n
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
for n,h in sources.items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=root)).hexdigest()==h,n
old=root/'rust-star/.work/runtime-target/release/rust-star';exe=w/'candidate-rust-star';assert not exe.exists();subprocess.run(['/bin/cp','-c',str(old),str(exe)],check=True);exe.chmod(0o555);assert sha(exe)==sha(old)
(w/'implementation.patch').write_text(subprocess.check_output(['git','diff','4ced6ee8',commit,'--','rust-star/runtime'],cwd=root,text=True))
files={n:sha(w/n) for n in ['router-repair-audit.json','repair-source.py','router-probe.m','router-probe.json','plan.json','engine-plan.json','engine-runner.py','engine-audit.py','select-diagnostic.py','source-edits.json','source-selection.json','prepare-source.py','run-validation.py','audit-profile.py','audit-ring.py','ring-audit.json','tests.log','build.log','fmt.log','implementation.patch']}
r=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=commit,executable_sha256=sha(exe),sources=sources,files=files,baseline_executable_sha256=read(w.parent/'plan.json')['executable_sha256'],baseline_profile_audit_sha256=sha(w.parent/'profile-audit.json'),engine_plan_sha256=sha(w/'engine-plan.json'),complete_three_file_restoration=True,unchanged_other_source_count=len(base)-len(ops))
baseline=w/'baseline-rust-star';assert not baseline.exists();subprocess.run(['/bin/cp','-c',str(w.parent/'current-rust-star'),str(baseline)],check=True);baseline.chmod(0o555);assert sha(baseline)==r['baseline_executable_sha256']
(w/'candidate-build.json').write_text(json.dumps(r,indent=2)+'\n');print('Frozen candidate',commit,'with exact source-scope/build evidence')
