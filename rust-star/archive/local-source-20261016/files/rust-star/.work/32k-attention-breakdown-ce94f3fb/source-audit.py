from pathlib import Path
import hashlib,json,re,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
path='rust-star/runtime/src/metal_shim.m'
base=subprocess.check_output(['git','show','ce94f3fb:'+path],cwd=root,text=True);candidate=(root/path).read_text()
markers=[('indexed','static int rust_star_metal_run_prefill_layer2_indexed_tile(', '\nint rust_star_metal_run_prefill_layer2_continuation_sparse_tile('),('compressed','int rust_star_metal_run_prefill_layer3_continuation_batch_attention(', '\nint rust_star_metal_begin_prefill_layer5_continuation(')]
extract=lambda s,a,b:s[s.index(a):s.index(b,s.index(a))]
normalized=lambda s:[l.strip() for l in s.splitlines() if l.strip()]
results={}
for kind,start,end in markers:
 b=extract(base,start,end);c=extract(candidate,start,end)
 pattern=r'id<MTLCounterSampleBuffer> attention_samples =.*?id<MTLComputeCommandEncoder> encoder = stage_compute_encoder\(command, attention_samples, 0u\);'
 restored,n=re.subn(pattern,'id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];',c,flags=re.S);assert n==1
 restored,n=re.subn(r'^[ ]*if \(attention_samples\) \{\n.*?^[ ]*\}\n','',restored,flags=re.S|re.M);assert n==(9 if kind=='indexed' else 4),(kind,n)
 assert normalized(restored)==normalized(b),kind
 (w/(kind+'-baseline.m')).write_text(b);(w/(kind+'-diagnostic.m')).write_text(c)
 results[kind]=dict(passed=True,baseline_sha256=sha(b.encode()),diagnostic_sha256=sha(c.encode()),profile_blocks_removed=n,normal_dispatches_bindings_arithmetic_exact=True)
changed=subprocess.check_output(['git','diff','--name-only','ce94f3fb','--','rust-star/runtime','metal'],cwd=root,text=True).splitlines()
assert set(changed)=={'rust-star/runtime/src/metal.rs','rust-star/runtime/src/metal_shim.h','rust-star/runtime/src/metal_shim.m','rust-star/runtime/src/long_prefill.rs'},changed
prior=json.loads((root/'rust-star/reports/32k-route-order-45520435/candidate-build.json').read_text())['sources']
kernels={p:sha((root/p).read_bytes()) for p in prior if p.endswith('.metal')}
assert all(prior[p]==h for p,h in kernels.items())
result=dict(passed=True,scope='Only diagnostic host instrumentation; normal encoder sequence restores exactly after stripping profile guards. All Metal sources unchanged.',changed_runtime_files=changed,paths=results,kernels=kernels)
(w/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Source audit passes; 2 attention paths and all kernel hashes unchanged')
