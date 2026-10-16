from pathlib import Path
import hashlib,json,os,struct,subprocess,datetime,math
w=Path(__file__).resolve().parent;root=w.parents[2];src=w.with_name('oracle-v5-source');exe=src/'ds4-bench';model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';prompt=root/'speed-bench/promessi_sposi.txt';env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
def sha(b):return hashlib.sha256(b).hexdigest()
a=json.loads((w/'oracle-v5-acceptance/manifest.json').read_text());assert a['status']=='complete' and a['conformance']['status']=='passed'
# CLI token dump loads vocabulary metadata only, and exercises the exact pinned tokenizer.
p=subprocess.run([str(src/'ds4'),'-m',model,'--raw-prompt','--prompt-file',str(prompt),'--dump-tokens'],cwd=src,env=env,capture_output=True,check=True)
raw=json.loads(p.stdout.splitlines()[0])[:32768];packed=struct.pack('<32768I',*raw);assert packed==(root/'rust-star/fixtures/prefill-frontier-32768-v3/token-ids.u32le.bin').read_bytes();(w/'v5-prompt-tokens.u32le.bin').write_bytes(packed);(w/'tokenization-v5.json').write_text(json.dumps({'source_commit':'ccbd16d36895ef9fe8edf7f793014988ff1ab7f4','cli_sha256':sha((src/'ds4').read_bytes()),'prompt_sha256':sha(prompt.read_bytes()),'token_sha256':sha(packed),'matches_historical_tokens':True},indent=2)+'\n')
eos_run=subprocess.run([str(src/'ds4'),'-m',model,'--raw-prompt','-p','<｜end▁of▁sentence｜>','--dump-tokens'],cwd=src,env=env,capture_output=True,check=True)
eos_ids=json.loads(eos_run.stdout.splitlines()[0]);assert len(eos_ids)==1;eos=eos_ids[0]
(w/'eos-token-v5.json').write_text(json.dumps({'token_text':'<｜end▁of▁sentence｜>','eos_id':eos})+'\n')
d=w/'oracle-v5-8192';d.mkdir()
command=[str(exe),'--metal','-m',model,'--prompt-file',str(prompt),'--ctx-start','8192','--ctx-max','8192','--gen-tokens','0','--warm-weights','--dump-frontier-logits-dir',str(d),'--csv',str(d/'benchmark.csv')]
with (d/'console.log').open('w') as f:subprocess.run(command,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
j=json.loads((d/'frontier_008192.logits.json').read_text());b=struct.pack('<129280f',*j['logits']);assert b==(root/'rust-star/fixtures/prefill-frontier-8192-v3/batch-prefill-logits.f32le.bin').read_bytes();(d/'verification.json').write_text(json.dumps({'passed':True,'full_fp32_words':129280,'matches_historical_fixture':True,'sha256':sha(b)})+'\n')
results=[]
for rep in ['a','b']:
 d=w/f'conformance-v5-{rep}';d.mkdir();dump=d/'logits';dump.mkdir();print('START',rep,flush=True)
 command=[str(exe),'--metal','-m',model,'--prompt-file',str(prompt),'--ctx-start','32768','--ctx-max','32768','--gen-tokens','128','--warm-weights','--dump-frontier-logits-dir',str(dump),'--csv',str(d/'benchmark.csv')]
 with (d/'console.log').open('w') as f:r=subprocess.run(command,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,r.returncode
 hashes={};tokens=[]
 with (d/'all-logits.f32le.bin').open('wb') as out:
  for step in range(129):
   p=dump/f'frontier_{32768+step:06d}.logits.json';j=json.loads(p.read_text());actual=struct.pack('<129280f',*j['logits']);values=struct.unpack('<129280f',actual);assert all(map(math.isfinite,values));selected=max(range(129280),key=values.__getitem__);assert selected==j['argmax_id'];assert selected!=eos,'Global argmax selects EOS: benchmark policy differs'
   out.write(actual);tokens.append(selected);hashes[str(step)]={'logits_sha256':sha(actual),'selected_token':selected}
 assert hashes['0']['logits_sha256']=='fb827b1c1ebc44933c35cd7d6e724c279dfb90035c9dbd209c49e893b3d4753b'
 (d/'token-ids.u32le.bin').write_bytes(struct.pack('<129I',*tokens));(d/'verification.json').write_text(json.dumps({'rows':129,'fp32_words':129*129280,'hashes':hashes,'command':command,'diagnostic_timing_ineligible':True},indent=2)+'\n');results.append(hashes)
 if rep=='b':assert (d/'all-logits.f32le.bin').read_bytes()==(w/'conformance-v5-a/all-logits.f32le.bin').read_bytes()
 print('PASS',rep,'all129 rows captured',flush=True)
assert results[0]==results[1]
(w/'conformance-v5.json').write_text(json.dumps({'passed':True,'fresh_processes':2,'full_fp32_rows_per_run':129,'prefill_tokens':32768,'decode_steps':128,'selected_tokens_exact':True,'diagnostic_timing_ineligible':True,'executable_sha256':sha(exe.read_bytes()),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
