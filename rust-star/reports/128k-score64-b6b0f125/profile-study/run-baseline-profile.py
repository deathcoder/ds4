from pathlib import Path
import subprocess,os,json,datetime
w=Path(__file__).resolve().parent
model='$MODEL_DIRECTORY/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
args=['/usr/bin/time','-l',str(w/'baseline-rust-star'),'engine-profile',model,'--context','131072','--gen-tokens','128','--json',str(w/'baseline-profile.json')]
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (w/'baseline-profile.log').open('x') as log:r=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
(w/'baseline-profile-status.json').write_text(json.dumps({'exit':r.returncode,'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
assert r.returncode==0
print('BASELINE PROFILE EXECUTION PASS; component audit still required',flush=True)
