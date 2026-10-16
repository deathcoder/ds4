from pathlib import Path
import hashlib,json,os,subprocess,shutil
root=Path.cwd();w=root/'rust-star/.work/native-32k-v31';src=w/'oracle-transcript-source';exe=src/'capture-deferred';fixture=root/'rust-star/fixtures/prefill-frontier-32768-v3';model=root/'rust-star/.work/oracle-v3-8k-source/model.gguf'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
pinned=sha(fixture/'batch-prefill-logits.f32le.bin')
for end in [32768,28672]:
 accepted=[]
 if end==32768:accepted=[w/'deferred-32768-a']
 for attempt in range(1,7):
  if len(accepted)==2:break
  d=w/f'anchored-{end}-{attempt}';d.mkdir();assert shutil.disk_usage(root).free>25*2**30
  env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_'))};env.update(RUST_STAR_ORACLE_DEFER_DUMPS='1',DS4_METAL_GRAPH_DUMP_PREFIX=str(d/'oracle'),DS4_METAL_GRAPH_DUMP_POS=str(end-4096),DS4_METAL_GRAPH_DUMP_NAME='hc_ffn_post')
  cmd=[str(exe),str(model),str(fixture/'token-ids.u32le.bin'),'32768','-',str(d/'input.u32le.bin'),str(d/'first.f32le.bin'),str(d/'final.f32le.bin'),str(d/'prefill.f32le.bin')]
  print('capture pinned-terminal',end,attempt,flush=True)
  with (d/'console.log').open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  hashes={p.name:{'bytes':p.stat().st_size,'sha256':sha(p),'fnv1a_u32':subprocess.check_output([str(root/'rust-star/.work/layer13-continuation-v28/hash_words'),str(p)],text=True).split()[0]} for p in sorted(d.glob('oracle*.bin'))}
  assert len(hashes)==43; (d/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
  valid=sha(d/'prefill.f32le.bin')==pinned
  (d/'terminal-anchor.json').write_text(json.dumps({'full_32k_logits_match':valid,'actual_sha256':sha(d/'prefill.f32le.bin'),'expected_sha256':pinned},indent=2)+'\n')
  if valid:
   if accepted:assert hashes==json.loads((accepted[0]/'hashes.json').read_text()),'full-logit-matching oracle snapshots disagree'
   accepted.append(d)
  print('terminal C0',valid,'accepted repetitions',len(accepted),flush=True)
  for p in d.glob('oracle*.bin'):p.unlink()
  (d/'raw-compacted.json').write_text(json.dumps({'reason':'hash inventories retained; acceptance requires immutable full terminal logits and repeated all-layer hashes','accepted':valid},indent=2)+'\n')
 assert len(accepted)==2,('no stable anchored pair',end)
 (w/f'anchored-{end}-accepted.json').write_text(json.dumps({'captures':[p.name for p in accepted],'terminal_logits_sha256':pinned},indent=2)+'\n')
 print('accepted anchored frontier',end,flush=True)
