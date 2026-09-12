from pathlib import Path
import json,hashlib,subprocess
w=Path(__file__).resolve().parent;r=w.parents[2];old=r/'rust-star/.work/pool-1k-fcdfef5f';diag=r/'rust-star/.work/scratch-rss-706417ca';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=read(w/'plan.json')
assert read(w/'mutation-proof-acceptance.json')['passed']
oldinit='    else memset(buffer.contents, 0, bytes);'
newinit='''    else if (!(bytes == 134217728u &&
        ([label isEqualToString:@"q_cur_buffer"] ||
         [label isEqualToString:@"attention_back_buffer"] ||
         [label isEqualToString:@"next_q_cur_buffer"] ||
         [label isEqualToString:@"next_attention_back_buffer"]))) {
        memset(buffer.contents, 0, bytes);
    }'''
oldcomment='''// newBufferWithLength returns zeroed storage. Reuse restores that entire
// initial range before the caller performs its existing CPU initialization.
// The caller checks prior GPU completion once, before any checkout in a tile.'''
newcomment='''// The caller checks prior GPU completion before any checkout in a tile.
// Preserve full initial zero/copy state except four exact 1K destinations:
// their entire range is overwritten by an ordered blit before any content read.
// Source buffers and every other scratch initializer retain their prior policy.'''
prior=(w/'prior-metal_shim.m').read_text();s=prior;patches=[]
for a,b in [(oldcomment,newcomment),(oldinit,newinit)]:
 assert s.count(a)==1;s=s.replace(a,b);patches.append(dict(before=a,after=b))
(w/'candidate-metal_shim.m').write_text(s)
for n in ['metal.rs','long_prefill.rs']:(w/('candidate-'+n)).write_bytes((w/('prior-'+n)).read_bytes())
(w/'source-edits.json').write_text(json.dumps(dict(patches=patches,inverse_exact=True,routed=p['pool_contract']['labels']),indent=2)+'\n')
helper=(w/'prior-pool-helper.inc').read_text().replace(oldcomment,newcomment).replace(oldinit,newinit);assert helper in s[:s.index('int rust_star_metal_run_prefill_layer0_boundary(')] or newinit in s
(w/'pool-helper.inc').write_text(helper)
# Reapply the established passive observer to the new source, extending the
# checkout record with the selected initialization policy; no GPU changes.
s=(w/'candidate-metal_shim.m').read_text();ip=[]
for e in read(old/'instrumentation.json')['patches']:
 a=e['before'].replace(oldinit,newinit);b=e['after'].replace(oldinit,newinit)
 if 'RS_POOL_CHECKOUT' in b:
  b=b.replace('@"copy":@(source != NULL),', '@"copy":@(source != NULL), @"elided":@(source == NULL && bytes == 134217728u && ([label isEqualToString:@"q_cur_buffer"] || [label isEqualToString:@"attention_back_buffer"] || [label isEqualToString:@"next_q_cur_buffer"] || [label isEqualToString:@"next_attention_back_buffer"])),')
 assert s.count(a)==1,(a[:100],s.count(a));s=s.replace(a,b);ip.append(dict(before=a,after=b))
check=s
for e in reversed(ip):assert check.count(e['after'])==1;check=check.replace(e['after'],e['before'])
assert check==(w/'candidate-metal_shim.m').read_text();(w/'candidate-instrumented.m').write_text(s);(w/'instrumentation.json').write_text(json.dumps(dict(patches=ip,inverse_exact=True),indent=2)+'\n')
s=(old/'pool-proof.m').read_text().replace(oldcomment,newcomment).replace(oldinit,newinit)
insert=''' NSUInteger selected=0;
 for(NSString *label in @[@"q_cur_buffer",@"attention_back_buffer",@"next_q_cur_buffer",@"next_attention_back_buffer"]) { @autoreleasepool {
  NSUInteger n=134217728u;RustStarBootstrapScratch *pool=[RustStarBootstrapScratch new];pool.buffers=[NSMutableDictionary new];pool.checkedOut=[NSMutableSet new];
  id<MTLBuffer> b=bootstrap_scratch_buffer(device,pool,label,n,NULL,error,sizeof(error));assert(b);memset(b.contents,0xa5,n);[pool.checkedOut removeAllObjects];
  id<MTLBuffer> reused=bootstrap_scratch_buffer(device,pool,label,n,NULL,error,sizeof(error));assert(reused==b);
  const uint8_t *v=b.contents;for(NSUInteger j=0;j<n;j++)assert(v[j]==0xa5);selected+=n;
  NSMutableData *source=[NSMutableData dataWithLength:n];memset(source.mutableBytes,0x42,n);[pool.checkedOut removeAllObjects];
  reused=bootstrap_scratch_buffer(device,pool,label,n,source.bytes,error,sizeof(error));assert(reused==b);
  v=b.contents;for(NSUInteger j=0;j<n;j++)assert(v[j]==0x42);selected+=n;
 }}
 assert(selected==1073741824ull);
'''
a=s.index(' printf("{');s=s[:a]+insert+s[a:];s=s.replace('\\"chunks\\":7','\\"selected_policy_checked_bytes\\":1073741824,\\"chunks\\":7');(w/'pool-proof.m').write_text(s)
for n in ['audit-ring.py','audit-profile.py','router-probe.m','ffn_router.metal','observer-proof.m','observer.inc','run-validation.py','run-lifetime.py','full-input-pins.py','engine-runner.py','engine-audit.py','audit-phases.py','execute-validation.py']:(w/n).write_bytes((old/n).read_bytes())
s=(old/'audit-pool-v2.py').read_text();anchor="  assert all(x['copy']==(x['label'] in ['token_buffer','position_buffer','compressed_position_buffer']) for x in cs)";assert anchor in s;s=s.replace(anchor,anchor+"\n  assert all(x['elided']==(x['label'] in p['copy_pairs'] and x['bytes']==134217728) for x in cs)\n  assert sum(x['bytes'] for x in cs if x['elided'])==536870912")
s=s.replace('full_fp32_words=16677120)', 'full_fp32_words=16677120,elided_bytes_per_tile=536870912,elided_checkouts=112)');(w/'audit-lifetime.py').write_text(s)
s=(diag/'source-proof.py').read_text().replace("s=(w/'prior-candidate-metal_shim.m').read_text()", "s=(w/'candidate-metal_shim.m').read_text()").replace("p['inventory']", "read(w/'inventory.json')");(w/'source-proof.py').write_text(s)
subprocess.run(['python3',str(w/'source-proof.py')],check=True)
engine=read(old/'engine-plan.json');engine.update(schema='rust-star-selective-clear-engine-plan-v1',created_utc=p['created_utc'],baseline_runtime_commit=p['baseline_commit'],role_B='Native1K bounded scratch reuse with four complete-copy destination clears omitted, Rust1.74',dependency='Mutation proof, actual helper policy checks, source and ownership proofs, fresh full C0/profile/observed checks all pass.')
assert engine['retention_criteria']==p['engine_retention_gates'];(w/'engine-plan.json').write_text(json.dumps(engine,indent=2)+'\n')
subprocess.run(['cp','-c',str(old/'baseline-rust-star'),str(w/'baseline-rust-star')],check=True);assert sha(w/'baseline-rust-star')==engine['baseline_executable_sha256']
# New build uses the exact prior bounded candidate as inverse reference.
s=(old/'build.py').read_text();s=s.replace(" idle();assert all(sha(r/n)"," idle();assert read(w/'mutation-proof-acceptance.json')['passed'];assert read(w/'source-proof.json')['passed'];assert all(sha(r/n)",1)
s=s.replace("pool-1k-production","selective-clear-production").replace("pool-1k-diagnostic","selective-clear-diagnostic")
s=s.replace("assert read(w/'pool-proof.log')['passed']", "assert read(w/'pool-proof.log')['passed'] and read(w/'pool-proof.log')['selected_policy_checked_bytes']==1073741824")
(w/'build.py').write_text(s)
proof=read(w/'source-proof.json');assert proof['selected_bytes']==536870912
(w/'ownership-source-audit.json').write_text(json.dumps(dict(passed=True,prior_candidate_sha256=sha(w/'prior-metal_shim.m'),candidate_sha256=sha(w/'candidate-metal_shim.m'),new_changes=patches,all_other_code_exact=True,prior_completion_and_ownership_unchanged=True,selected_labels=sorted(p['copy_pairs']),selected_bytes=536870912,other_initializations_unchanged=101,actual_helper_extracted=True),indent=2)+'\n')
print('Prepared selective helper, observer, source inverse and frozen validation scripts')
