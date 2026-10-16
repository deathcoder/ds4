from pathlib import Path
import subprocess,json,hashlib,datetime
w=Path(__file__).resolve().parent;root=w.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['git','diff','HEAD','--','rust-star/runtime'],cwd=root)
assert '318 passed; 0 failed' in (w/'tests.log').read_text()
assert 'Ran 94 tests' in (w/'python-tests.log').read_text() and (w/'python-tests.log').read_text().rstrip().endswith('OK')
assert 'Finished `release` profile' in (w/'build.log').read_text()
assert json.loads((w/'probe-verification.json').read_text())['passed']
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
exe=w/'candidate-rust-star';assert not exe.exists();subprocess.run(['cp','-c',str(root/'rust-star/.work/runtime-target/release/rust-star'),str(exe)],check=True);exe.chmod(0o555)
prior=json.loads((root/'rust-star/reports/128k-score64-b6b0f125/candidate-build.json').read_text())
files=subprocess.check_output(['git','ls-files','rust-star/runtime'],cwd=root,text=True).splitlines()+[n for n in prior['sources'] if n.startswith('metal/')]
sources={n:sha(root/n) for n in files}
for n,h in sources.items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=root)).hexdigest()==h,n
result=dict(regression_verification_sha256=sha(w/'probe-verification.json'),commit=commit,tree=subprocess.check_output(['git','rev-parse',commit+'^{tree}'],cwd=root,text=True).strip(),executable_sha256=sha(exe),sources=sources,milestone_plan_sha256=sha(w/'milestone-plan.json'),tests_log_sha256=sha(w/'tests.log'),build_log_sha256=sha(w/'build.log'),python_tests_log_sha256=sha(w/'python-tests.log'),preflight_sha256=sha(w/'preflight.json'),frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(w/'candidate-build.json').write_text(json.dumps(result,indent=2)+'\n');print('FROZEN',commit,result['executable_sha256'])
help_run=subprocess.run([str(exe),'long-prefill-reference-probe','--help'],capture_output=True,text=True,check=True);assert '--retained' in help_run.stdout and '256K' in help_run.stdout
bad_run=subprocess.run([str(exe),'long-prefill-reference-probe','model','tokens','selected','logits','report','--retained','--profile-json','profile'],capture_output=True,text=True);assert bad_run.returncode and 'expected model' in bad_run.stderr
(w/'cli-checks.json').write_text(json.dumps({'passed':True,'help':help_run.stdout,'mutually_exclusive_options_exit':bad_run.returncode,'mutually_exclusive_options_error':bad_run.stderr,'executable_sha256':sha(exe)},indent=2)+'\n')
