from pathlib import Path
import hashlib,json,subprocess,datetime,traceback
w=Path(__file__).resolve().parent;r=w.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text());utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,x):(w/n).write_text(json.dumps(x,indent=2)+'\n')
try:
 assert not [x for x in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(x.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','diagnostic-rust-star','baseline-rust-star','probe','mutation-proof','ds4','ds4-bench','ds4-server'}]
 p=read(w/'plan.json');assert all(sha(r/n)==h for n,h in p['source_reference']['sources'].items())
 files={n:sha(w/n) for n in ['plan.json','mutation-proof.m','mutation-proof.metal','inventory.json','run-mutation-proof.py']};save('mutation-proof-freeze.json',dict(utc=utc(),files=files))
 with (w/'mutation-build.log').open('x') as log:result=subprocess.run(['clang','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(w/'mutation-proof.m'),'-o',str(w/'mutation-proof')],stdout=log,stderr=subprocess.STDOUT)
 save('mutation-build-status.json',dict(exit_code=result.returncode));assert result.returncode==0
 with (w/'mutation-proof.log').open('x') as log:result=subprocess.run([str(w/'mutation-proof'),'proof',str(w/'inventory.json'),str(w/'mutation-proof.metal')],stdout=log,stderr=subprocess.STDOUT)
 save('mutation-proof-status.json',dict(exit_code=result.returncode,executable_sha256=sha(w/'mutation-proof')));assert result.returncode==0
 lines=(w/'mutation-proof.log').read_text().splitlines();rows=[json.loads(x[7:]) for x in lines if x.startswith('RESULT ')]
 assert rows==[dict(passed=True,cases=24,checked_bytes=2147549248,guard_bytes=24576,shortened_copy_negative_controls=12,missing_mutation_negative_controls=12)]
 assert all(sha(w/n)==h for n,h in files.items());assert all(sha(r/n)==h for n,h in p['source_reference']['sources'].items())
 save('mutation-proof-acceptance.json',dict(passed=True,utc=utc(),result=rows[0],files=files,runtime_unchanged=True,previous_diagnostic_verdict_unchanged=True));print(json.dumps(rows[0],indent=2))
except BaseException:
 save('mutation-proof-failure.json',dict(utc=utc(),traceback=traceback.format_exc()));raise
