from pathlib import Path
import contextlib,datetime,hashlib,io,json,runpy
w=Path(__file__).resolve().parent;root=w.parents[2];report=root/'rust-star/reports/compressed-skip-runs-fbbea533';parent=root/'rust-star/reports/compressed-grid-warm-c21ebfda'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(w/'audit.json')['passed']
for script in ['audit.py','cpu_trace.py','edges/audit.py']:
 with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(w/script),run_name='__main__')
freeze=read(w/'freeze.json');plan=read(w/'plan.json')
for n,h in freeze['files'].items():assert sha(w/n)==h,n
for n,h in plan['runtime_sources'].items():assert sha(root/n)==h,n
completion=dict(passed=True,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime_commit=plan['runtime_commit'],runtime_sources=plan['runtime_sources'],freeze_sha256=sha(w/'freeze.json'),edge_freeze_sha256=sha(w/'edge-freeze.json'),probe_executable_sha256=sha(w/'probe'),edge_executable_sha256=sha(w/'edges/edge-probe'),runtime_changed=False)
(w/'completion-pins.json').write_text(json.dumps(completion,indent=2)+'\n')
report.mkdir();redactions={}
def copy(src,name):
 dst=report/name;dst.parent.mkdir(parents=True,exist_ok=True)
 raw=src.read_bytes();stored=raw.replace(str(root).encode(),b'<repo>');dst.write_bytes(stored)
 if stored!=raw:redactions[name]=dict(original_sha256=hashlib.sha256(raw).hexdigest(),stored_sha256=sha(dst),replacement='Absolute checkout path replaced with <repo>.')
for n in ['plan.json','probe.m','arguments.h','combined.metal','source-audit.json','baseline-kernel.metal','candidate-kernel.metal','prepare.py','cpu_trace.py','cpu-trace.json','edge-freeze.json','toolchain.txt','hardware.json','parent-shape-audit.json','build.log','run.py','audit.py','audit.json','freeze.json','completion-pins.json','verify.py']:
 copy(w/n,n)
for n in ['probe.m','arguments.h','build.log','run.py','run-freeze.json','audit.py','audit.json','probe.jsonl','run.log','status.json']:copy(w/'edges'/n,'edges/'+n)
for session in [0,1]:
 for n in ['probe.jsonl','run.log','status.json','preflight.json','final-pins.json','host-observations.jsonl']:copy(w/f'session-{session}'/n,f'session-{session}/{n}')
for src,n in [('probe.m','parent/probe.m'),('audit.json','parent/predecessor-audit.json'),('parent/protocol-audit.json','parent/protocol-audit.json')]:copy(parent/src,n)
for p in (parent/'source-inputs').rglob('*'):
 if p.is_file():copy(p,'source-inputs/'+str(p.relative_to(parent/'source-inputs')))
(report/'redactions.json').write_text(json.dumps(redactions,indent=2)+'\n');print(report)
