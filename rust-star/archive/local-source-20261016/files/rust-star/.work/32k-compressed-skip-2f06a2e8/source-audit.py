from pathlib import Path
import hashlib,json,subprocess,re
w=Path(__file__).resolve().parent;r=w.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((r/'rust-star/.work/32k-attention-breakdown-ce94f3fb/candidate-build.json').read_text())
changed={'rust-star/runtime/build.rs','rust-star/runtime/src/metal_shim.m'}
for path,h in old['sources'].items():
 if path not in changed:assert sha(r/path)==h,path
kernel=r/'rust-star/runtime/src/compressed_prefill.metal';prior=r/'rust-star/reports/compressed-skip-runs-fbbea533/candidate-kernel.metal'
assert kernel.read_text().replace('rust_star_compressed_prefill','probe_flash_skip_runs')==prior.read_text()
s=(r/'rust-star/runtime/src/metal_shim.m').read_text()
s=s.replace('\n@property(nonatomic, strong) id<MTLComputePipelineState> compressedNativePipeline;\n@property(nonatomic, strong) id<MTLComputePipelineState> compressedNativeKvpadPipeline;','')
for name in ['native-pipeline-helper.txt','native-predicate.txt']:
 t=(w/name).read_text();assert s.count(t)==1;s=s.replace(t,'')
a=(w/'candidate-dispatch.txt').read_text();b=(w/'original-dispatch.txt').read_text();assert s.count(a)==1;s=s.replace(a,b)
assert hashlib.sha256(s.encode()).hexdigest()==old['sources']['rust-star/runtime/src/metal_shim.m']
b=(r/'rust-star/runtime/build.rs').read_text();b=b.replace('    println!("cargo:rerun-if-changed=src/compressed_prefill.metal");\n','').replace('            manifest.join("src/compressed_prefill.metal"),\n','')
assert hashlib.sha256(b.encode()).hexdigest()==old['sources']['rust-star/runtime/build.rs']
# Exactly the production attention source assembly and options from build.rs.
b=(r/'rust-star/runtime/build.rs').read_text();frag=b[b.index('            manifest.join("src/attention_ingress.metal")'):b.index('        "kAttentionIngressSource"')]
paths=[(r/'rust-star/runtime'/x).resolve() for x in re.findall(r'manifest.join\("([^"\n]+)"\)',frag)]
assert len(paths)==9 and paths[-1]==kernel.resolve()
assembled=''.join(''.join(line+'\n' for line in p.read_text().splitlines()) for p in paths)
(w/'assembled.metal').write_text(assembled)
result=dict(passed=True,prior_runtime_commit=old['commit'],unchanged_sources=38,prior_kernel_sha256=sha(prior),candidate_kernel_sha256=sha(kernel),kernel_rename_only=True,entire_original_host_restored_by_removing_scoped_additions=True,entire_original_build_restored=True,assembly_inputs={str(p.relative_to(r.resolve())):sha(p) for p in paths},assembled_sha256=sha(w/'assembled.metal'),native_predicate='prefillLayerChunkStart && position_start == chunk_start && position_start != 0',legacy_prefix_decode_dispatches_unchanged=True)
(w/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Source restoration, 38 unchanged files and assembly pass')
