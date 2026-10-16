from pathlib import Path
import hashlib,json,os,subprocess,shutil
root=Path.cwd();w=root/'rust-star/.work/native-32k-v31';src=w/'oracle-transcript-source';exe=src/'capture-deferred';fixture=root/'rust-star/fixtures/prefill-frontier-32768-v3';model=root/'rust-star/.work/oracle-v3-8k-source/model.gguf'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for end in [32768,28672]:
 tokens=w/f'tokens-{end}.u32le.bin';tokens.write_bytes((fixture/'token-ids.u32le.bin').read_bytes()[:end*4]);prior=None;prior_logits=None
 for repeat in ['a','b']:
  d=w/f'deferred-{end}-{repeat}';d.mkdir();assert shutil.disk_usage(root).free>25*2**30
  env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_'))};env.update(RUST_STAR_ORACLE_DEFER_DUMPS='1',DS4_METAL_GRAPH_DUMP_PREFIX=str(d/'oracle'),DS4_METAL_GRAPH_DUMP_POS=str(end-4096),DS4_METAL_GRAPH_DUMP_NAME='hc_ffn_post')
  cmd=[str(exe),str(model),str(tokens),str(end),'-',str(d/'input.u32le.bin'),str(d/'first.f32le.bin'),str(d/'final.f32le.bin'),str(d/'prefill.f32le.bin')]
  print('capture deferred',end,repeat,flush=True)
  with (d/'console.log').open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  hashes={p.name:{'bytes':p.stat().st_size,'sha256':sha(p),'fnv1a_u32':subprocess.check_output([str(root/'rust-star/.work/layer13-continuation-v28/hash_words'),str(p)],text=True).split()[0]} for p in sorted(d.glob('oracle*.bin'))}
  assert len(hashes)==43,len(hashes);(d/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
  current_logits=sha(d/'prefill.f32le.bin')
  if end==32768:assert current_logits==sha(fixture/'batch-prefill-logits.f32le.bin'),'deferred capture changed pinned 32K logits'
  if prior is not None:assert prior==hashes and prior_logits==current_logits,'deferred independent repeat differs'
  prior=hashes;prior_logits=current_logits
  for p in d.glob('oracle*.bin'):p.unlink()
  (d/'raw-compacted.json').write_text(json.dumps({'reason':'hashed deferred GPU snapshots; independent comparison and pinned full logits validated when applicable','files':list(hashes)},indent=2)+'\n')
 print('verified deferred',end,flush=True)
