from pathlib import Path
import hashlib,json,os,struct,subprocess,time
w=Path(__file__).resolve().parent;root=w.parents[2];exe=w/'oracle-source/ds4-bench';model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';prompt=root/'speed-bench/promessi_sposi.txt';env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
fixture=root/'rust-star/fixtures';prefill=(fixture/'prefill-frontier-32768-v3/logits.f32le.bin')
manifest=json.loads((fixture/'prefill-frontier-32768-v3/manifest.json').read_text());prefill=next(fixture/'prefill-frontier-32768-v3'/t['path'] for t in manifest['tensors'] if t.get('shape')==[129280])
expected=prefill.read_bytes();trace=(fixture/'prefill-decode-32768-v3/decode-logits.f32le.bin').read_bytes();tokens=struct.unpack('<129I',(fixture/'prefill-decode-32768-v3/token-ids.u32le.bin').read_bytes());results=[]
for rep in ['a','b']:
 d=w/f'conformance-{rep}';d.mkdir();dump=d/'logits';dump.mkdir();print('START',rep,flush=True)
 command=[str(exe),'--metal','-m',model,'--prompt-file',str(prompt),'--ctx-start','32768','--ctx-max','32768','--gen-tokens','128','--warm-weights','--dump-frontier-logits-dir',str(dump),'--csv',str(d/'benchmark.csv')]
 with (d/'console.log').open('w') as f:r=subprocess.run(command,cwd=exe.parent,env=env,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise SystemExit(r.returncode)
 hashes={}
 for step in range(129):
  p=dump/f'frontier_{32768+step:06d}.logits.json';j=json.loads(p.read_text());actual=struct.pack('<129280f',*j['logits']);reference=expected if step==0 else trace[(step-1)*129280*4:step*129280*4]
  assert actual==reference,(rep,step,next(i for i in range(129280) if actual[i*4:i*4+4]!=reference[i*4:i*4+4]))
  assert j['argmax_id']==tokens[step],(step,j['argmax_id'],tokens[step]);hashes[str(step)]={'logits_sha256':hashlib.sha256(actual).hexdigest(),'selected_token':j['argmax_id']}
 (d/'verification.json').write_text(json.dumps({'full_fp32_match':True,'rows':129,'fp32_words':129*129280,'hashes':hashes},indent=2)+'\n');results.append(hashes)
 # Compact only after exact comparison; the immutable binary fixtures remain.
 for p in dump.iterdir():p.unlink()
 dump.rmdir();print('PASS',rep,'all 129 full-logit rows',flush=True)
assert results[0]==results[1]
(w/'conformance.json').write_text(json.dumps({'passed':True,'fresh_processes':2,'full_fp32_rows_per_run':129,'prefill_tokens':32768,'decode_steps':128,'selected_tokens_exact':True,'diagnostic_timing_ineligible':True,'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()},indent=2)+'\n')
