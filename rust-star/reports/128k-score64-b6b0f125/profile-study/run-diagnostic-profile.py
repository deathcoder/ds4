from pathlib import Path
import subprocess,os,json,datetime,hashlib,time
w=Path(__file__).resolve().parent;root=w.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((w/'diagnostic-build.json').read_text());exe=w/'candidate-rust-star'
assert sha(exe)==build['executable_sha256']
assert json.loads((w/'preflight.json').read_text())['passed']
for n,h in build['sources'].items():assert sha(root/n)==h,n
ref=root/'rust-star/reports/128k-c0-0eb484ea/reference-131072'
model='$MODEL_DIRECTORY/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
args=['/usr/bin/time','-l',str(exe),'long-prefill-reference-probe',model,str(ref/'prompt.u32le.bin'),str(ref/'selected.u32le.bin'),str(ref/'logits.f32le.bin'),str(w/'diagnostic-c0.json'),'--profile-json',str(w/'diagnostic-profile.json')]
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (w/'diagnostic-profile.log').open('x') as log:r=subprocess.run(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
(w/'diagnostic-profile-status.json').write_text(json.dumps({'exit':r.returncode,'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':args,'build':build['commit']},indent=2)+'\n')
assert r.returncode==0
assert json.loads((w/'diagnostic-c0.json').read_text())==json.loads((root/'rust-star/reports/128k-c0-0eb484ea/checks/native-131072-a.json').read_text())
subprocess.run(['python3',str(w/'audit-profile.py'),str(w/'diagnostic-profile.json')],check=True)
print('PROFILE AND FULL C0 AUDIT PASS',flush=True)
