from pathlib import Path
import hashlib,json,os,subprocess,shutil,struct,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];src=w.with_name('oracle-v5-source');exe=src/'ds4-bench';prompt=root/'speed-bench/promessi_sposi.txt';model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');hasher=root/'rust-star/.work/layer13-continuation-v28/hash_words';records=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert sha(exe)=='a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0'
for end in [16384,20480,24576,12288]:
 prior=None;prior_logits=None
 for rep in ['a','b']:
  d=w/f'hc-v5-{end}-{rep}';d.mkdir();(d/'logits').mkdir();assert shutil.disk_usage(root).free>25*2**30
  env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(d/'oracle'),DS4_METAL_GRAPH_DUMP_POS=str(end-4096),DS4_METAL_GRAPH_DUMP_NAME='hc_ffn_post')
  command=[str(exe),'--metal','-m',str(model),'--prompt-file',str(prompt),'--ctx-start',str(end),'--ctx-max',str(end),'--gen-tokens','0','--prefill-chunk','4096','--dump-frontier-logits-dir',str(d/'logits'),'--csv',str(d/'benchmark.csv')]
  print('START HC',end,rep,flush=True)
  with (d/'console.log').open('w') as f:subprocess.run(command,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  files=sorted(d.glob('*.bin'));assert len(files)==43,len(files);hashes={}
  for p in files:
   parts=subprocess.check_output([str(hasher),str(p)],text=True).split();assert int(parts[1])==268435456==p.stat().st_size;hashes[p.name]={'bytes':p.stat().st_size,'sha256':sha(p),'fnv1a_u32':parts[0]}
  (d/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n');j=json.loads((d/'logits'/f'frontier_{end:06d}.logits.json').read_text());logits=struct.pack('<129280f',*j['logits'])
  if prior is not None:assert prior==hashes and prior_logits==logits,'Independent v5 internal capture mismatch'
  if end==16384:assert hashes['oracle_hc_ffn_post-34_pos12288.bin']['fnv1a_u32']=='3e1a2b79f656939a','V5 checkpoint does not match Rust; investigate before replacing targets'
  prior=hashes;prior_logits=logits
  records.append({'frontier':end,'repeat':rep,'full_hc_layers':43,'full_hc_bytes_per_layer':268435456,'hashes_sha256':sha(d/'hashes.json'),'logits_sha256':hashlib.sha256(logits).hexdigest(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});(w/'hc-v5-progress.json').write_text(json.dumps(records,indent=2)+'\n')
  # Successful full payloads are compacted only after their complete SHA/FNV inventory is durable.
  for p in files:p.unlink()
  (d/'raw-compacted.json').write_text(json.dumps({'reason':'Full payloads hashed; repeat inventories and full output logits retained','files':list(hashes)},indent=2)+'\n');print('PASS HC',end,rep,flush=True)
(w/'hc-v5-acceptance.json').write_text(json.dumps({'passed':True,'source_commit':'ccbd16d36895ef9fe8edf7f793014988ff1ab7f4','executable_sha256':sha(exe),'hash_tool_sha256':sha(hasher),'hash_tool_source_sha256':sha(hasher.with_suffix('.c')),'contexts':[12288,16384,20480,24576],'fresh_repetitions_per_context':2,'layers_per_context':43,'scope':'complete4096-row hc_ffn_post tensors; full frontier outputs also repeated','records':records},indent=2)+'\n')
