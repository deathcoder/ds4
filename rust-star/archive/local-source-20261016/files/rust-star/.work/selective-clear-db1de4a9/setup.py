from pathlib import Path
import json,hashlib,datetime,subprocess
r=Path('/Users/deathcodevision/dev/ds4-rust-star-bootstrap');old=r/'rust-star/.work/pool-1k-fcdfef5f';diag=r/'rust-star/.work/scratch-rss-706417ca';w=r/'rust-star/.work/selective-clear-db1de4a9';w.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(old/'plan.json');p.update(schema='rust-star-selective-clear-milestone-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip())
p['source_reference']['sources']=read(diag/'plan.json')['runtime_state']['sources']
p['scope']='One milestone: guarded copy-then-in-place-mutation proof before integration, then narrow four-destination clear omission in the prior bounded scratch pool; fresh full C0 and pool proof before unchanged engine gates. No256K, publication or remote dispatch. Finish verdict, report and stop.'
p['copy_pairs']=read(diag/'plan.json')['copy_pairs'];p['mutation_proof']=dict(sizes=[4,4096,134217728],labels=4,turns=2,payload_bytes=2147549248,guard_bytes=24576,positive_cases=24,shortened_copy_negative_controls=12,missing_mutation_negative_controls=12,mutation='Read each existing byte and XOR it with the corresponding byte of 0x9e3779b9 after the complete blit, before verification. Zeroed and opposing-poison destinations must produce identical expected results; unchanged guards and reused identities. Execute before any runtime edit.')
p['pool_contract']['first_write']='Only four fixed full-copy destinations may omit CPU clear when pool is active, source is NULL and length is exactly134217728. Other101 buffers retain full zero/copy initialization; nonpool behavior unchanged. Prior checked GPU completion and fullcopy-before-read proof mandatory.'
p['engine_retention_gates']=read(diag/'plan.json')['engine_retention_gates'];p['failure_policy']='Preserve every failure. No discretionary measurement repeats or threshold changes. Proof, correctness or ownership failure stops before engine timing; complete a frozen engine block once started, then retain only if all gates pass, otherwise restore accepted source/default.'
(w/'plan.json').write_text(json.dumps(p,indent=2)+'\n');(w/'prior-diagnostic-decision.json').write_bytes((diag/'decision.json').read_bytes());(w/'inventory.json').write_bytes((diag/'inventory.json').read_bytes())
for n in ['metal_shim.m','metal.rs','long_prefill.rs']:
 (w/('baseline-'+n)).write_bytes((r/'rust-star/runtime/src'/n).read_bytes());(w/('prior-'+n)).write_bytes((old/('candidate-'+n)).read_bytes())
(w/'prior-build.json').write_bytes((old/'candidate-build.json').read_bytes());(w/'prior-pool-helper.inc').write_bytes((old/'pool-helper.inc').read_bytes());(w/'prior-source-edits.json').write_bytes((old/'source-edits.json').read_bytes())
assert all(sha(r/n)==h for n,h in p['source_reference']['sources'].items())
(w/'default-identity.json').write_text(json.dumps(dict(sha256=sha(r/'rust-star/.work/runtime-target/release/rust-star')),indent=2)+'\n')
s=(diag/'probe.m').read_text();s=s[:s.index('    assert(([@[@"A"')]+ '    return 3;\n}}\n'
s=s.replace('    id<MTLBuffer> errors=', '    id<MTLComputePipelineState> mutator=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"mutate_existing"] error:&error];assert(mutator&&!error);\n    id<MTLBuffer> errors=',1)
s=s.replace('negative=0,cases=0','negative=0,missing=0,cases=0')
s=s.replace('                    for(NSUInteger role=0;role<2;role++){id<MTLBuffer> dst=role?poisoned:zeroed;encode(e,verifier,dst,(Args){n,256,seed,0},errors);', '                    encode(e,mutator,zeroed,(Args){n,256,seed,0},nil);encode(e,mutator,poisoned,(Args){n,256,seed,0},nil);[e endEncoding];e=[c computeCommandEncoder];assert(e);\n                    for(NSUInteger role=0;role<2;role++){id<MTLBuffer> dst=role?poisoned:zeroed;encode(e,verifier,dst,(Args){n,256,seed,2},errors);')
s=s.replace('assert(e);encode(e,verifier,poisoned,(Args){n,256,seed_for(1,name),0},errors);', 'assert(e);encode(e,mutator,poisoned,(Args){n,256,seed_for(1,name),0},nil);[e endEncoding];e=[c computeCommandEncoder];assert(e);encode(e,verifier,poisoned,(Args){n,256,seed_for(1,name),2},errors);')
anchor='negative++;*(uint32_t *)errors.contents=0;'
assert s.count(anchor)==1;s=s.replace(anchor,anchor+'''
                c=[queue commandBuffer];blit=[c blitCommandEncoder];assert(blit);[blit copyFromBuffer:src sourceOffset:0 toBuffer:poisoned destinationOffset:256 size:n];[blit endEncoding];
                e=[c computeCommandEncoder];assert(e);encode(e,verifier,poisoned,(Args){n,256,seed_for(1,name),2},errors);[e endEncoding];[c commit];wait_ok(c);assert(*(uint32_t *)errors.contents>0);missing++;*(uint32_t *)errors.contents=0;
''')
s=s.replace('@"negative_controls":@(negative)', '@"shortened_copy_negative_controls":@(negative),@"missing_mutation_negative_controls":@(missing)')
assert s.count('encode(e,mutator,')==3 and s.count('seed,2},errors)')==1
(w/'mutation-proof.m').write_text(s)
s=(diag/'probe.metal').read_text();s=s.replace('return kind ? (seed & 255u)*0x01010101u :\n        ((uint)i*1664525u + seed*1013904223u) ^ ((uint)i>>7u);','uint v=((uint)i*1664525u + seed*1013904223u) ^ ((uint)i>>7u);\n    return kind==1u ? (seed & 255u)*0x01010101u : (kind==2u ? v^0x9e3779b9u : v);')
s+='''
kernel void mutate_existing(device uchar *data [[buffer(0)]],
    constant Args &a [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
    ulong base=(ulong)gid*16ul;
    if(base>=a.bytes)return;
    if(base+16ul<=a.bytes) {
        device uint4 *p=(device uint4 *)(data+a.offset+base);
        *p = *p ^ uint4(0x9e3779b9u);
    } else {
        for(ulong j=base;j<a.bytes;j++)data[a.offset+j]^=(uchar)(0x9e3779b9u>>((j%4ul)*8ul));
    }
}
'''
(w/'mutation-proof.metal').write_text(s)
(w/'setup.py').write_bytes(Path(__file__).read_bytes())
j=r/'RUST_STAR_JOURNAL.md';s=j.read_text();anchor='## 2026-09-12 — Close scratch RSS diagnosis with a protocol gap';entry='''## 2026-09-12 — Start selective scratch initialization milestone

- D approved the next milestone. Prospective plan:
  rust-star/.work/selective-clear-db1de4a9/plan.json. First close the missing
  guarded copy-then-in-place-mutation proof, including shortened-copy and
  omitted-mutation negative controls, before any runtime edits.
- If that passes, preserve the prior bounded pool and omit CPU clearing only
  for four exact full-copy destinations. Keep all other 101 initializations,
  checked completion, ownership bounds and every engine gate unchanged.
- Require source/GPU/host/build proofs, fresh 12K internal and normal/profiled/
  observed 32K full C0, then the frozen two-warmup/five-pair engine block.
  Preserve failures; restore on rejection. Report this milestone and next,
  then STOP. No publication, remote dispatch or 256K work.

''';assert anchor in s;j.write_text(s.replace(anchor,entry+anchor,1));print(w)
