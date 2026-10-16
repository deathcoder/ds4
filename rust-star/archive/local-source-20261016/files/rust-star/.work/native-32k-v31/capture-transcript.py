from pathlib import Path
import hashlib,json,os,subprocess,struct
root=Path.cwd();w=root/'rust-star/.work/native-32k-v31';src=w/'oracle-transcript-source';exe=src/'capture-32k-transcript';fixture=root/'rust-star/fixtures/prefill-frontier-32768-v3';model=root/'rust-star/.work/oracle-v3-8k-source/model.gguf'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
for repeat in ['a','b']:
 d=w/f'oracle-transcript-{repeat}';d.mkdir();steps=d/'steps';steps.mkdir()
 env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_'))};env['RUST_STAR_ORACLE_STEP_LOGITS_DIR']=str(steps)
 cmd=[str(exe),str(model),str(fixture/'token-ids.u32le.bin'),'32895','-',str(d/'input-tokens.u32le.bin'),str(d/'first.f32le.bin'),str(d/'final.f32le.bin'),str(d/'prefill.f32le.bin')]
 print('capture transcript',repeat,flush=True)
 with (d/'console.log').open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 rows=sorted(steps.glob('*.bin'));assert len(rows)==128
 assert all(p.stat().st_size==129280*4 for p in rows)
 (d/'all-logits.f32le.bin').write_bytes(b''.join(p.read_bytes() for p in rows))
 r={'prefill_matches_pinned':sha(d/'prefill.f32le.bin')==sha(fixture/'batch-prefill-logits.f32le.bin'),'files':{p.name:sha(p) for p in d.glob('*.bin')},'steps':{p.name:sha(p) for p in rows},'producer_executable_sha256':sha(exe),'helper_sha256':sha(src/'capture_32k_transcript.c')}
 (d/'verification.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r);print('prefill matches pinned',r['prefill_matches_pinned'],flush=True)
print('independent repeats',results[0]==results[1],flush=True)
