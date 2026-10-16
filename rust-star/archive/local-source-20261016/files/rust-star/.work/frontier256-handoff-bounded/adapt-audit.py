from pathlib import Path
w=Path(__file__).resolve().parent
p=w/'audit-results.py';s=p.read_text()
# Pin concrete candidate identity only after freeze; leave tokens that must be replaced.
s=s.replace("assert b['commit']=='0d5f50ff0fc1d725fc2fdd5a23579427647d627a'", "assert b['commit']=='NEW_CANDIDATE_COMMIT'").replace("assert b['executable_sha256']=='1ace3c5031d832d34eb0064fdfd71b3feeb0a0c3000d568496b4828eb82548c5'", "assert b['executable_sha256']=='NEW_CANDIDATE_EXECUTABLE'")
a=s.index('# The independent synthetic GPU regression');b=s.index('# Preserve and verify the failed predecessor',a)
old=s[a:b].replace("w/'regression", "w/'rejected-0d5f50ff/regression").replace("b['regression_verification_sha256']", "read(w/'rejected-0d5f50ff/candidate-build.json')['regression_verification_sha256']").replace("reg['native_shim_sha256']==b['sources']['rust-star/runtime/src/metal_shim.m']", "reg['native_shim_sha256']==read(w/'rejected-0d5f50ff/candidate-build.json')['sources']['rust-star/runtime/src/metal_shim.m']")
extra='''# The new production handoff/reset regression checks every copied word at full shapes.
reg=read(w/'regression/probe-verification.json');assert reg['passed'] and reg['all_copy_checks_and_ranges_preserved']
assert original_sha(w/'regression/probe-verification.json')==b['regression_verification_sha256']
assert reg['production_shim_sha256']==b['sources']['rust-star/runtime/src/metal_shim.m']
for name,h in reg['files'].items():assert original_sha(w/'regression'/name)==h,name
cases=reg['cases'];assert [(x['rows'],x['cycle']) for x in cases]==[(131072,1),(131072,2),(262144,1),(262144,2)]
logged=[json.loads(x) for x in (w/'regression/probe-results.jsonl').read_text().splitlines()]
assert [x for x in logged if 'copied_words' in x]==cases and logged[-1]['passed']
for x in cases:
 n=x['rows'];words=43*128*512
 for l in range(2,43):
  ratio=128 if l%2 else 4;words+=(n//ratio)*512+2*(65536 if l%2 else 8192)
  if ratio==4:words+=(n//4)*128+2*2048
 assert x['copied_words']==words and x['untouched_tail_words']==928000 and x['copies']==229
 assert x['commands']==x['waits']==(43 if n>131072 else 1)
 for key in ['exact','per_command_source_release','residency_preserved','reset_passed']:assert x[key] is True
# Ensure the prior decode regression still covers the current unchanged dispatch/shaders.
oldshim=(w/'rejected-0d5f50ff/candidate-source/rust-star/runtime/src/metal_shim.m').read_text()
newshim=(source_root/'rust-star/runtime/src/metal_shim.m').read_text()
start='/* Encode the same exact KV/state copies for both handoff schedules. */'
oldstart='int rust_star_metal_adopt_prefill_decoder_state('
end='int rust_star_metal_copy_completed_output_logits('
assert oldshim[:oldshim.index(oldstart)]==newshim[:newshim.index(start)]
assert oldshim[oldshim.index(end):]==newshim[newshim.index(end):]
oldbody=oldshim[oldshim.index('            id<MTLBuffer> source_raw',oldshim.index(oldstart)):]
oldbody=oldbody[:oldbody.index('\\n        [blit endEncoding];\\n        const double wall_start')]
assert oldbody.endswith('\\n        }');oldbody=oldbody[:-len('\\n        }')]
oldbody=oldbody.replace('[blit endEncoding];','').replace('if (layer < 2u) continue;','if (layer < 2u) return 1;').replace('if (ratio != 4u) continue;','if (ratio != 4u) return 1;').replace('copies++;','(*copies)++;').replace('copies += 3u;','*copies += 3u;')
newbody=newshim[newshim.index('    id<MTLBuffer> source_raw',newshim.index(start)):];newbody=newbody[:newbody.index('\\n    return 1;\\n}')]
assert re.sub(r'\\s+','',oldbody)==re.sub(r'\\s+','',newbody)
'''
s=s[:a]+old+extra+s[b:]
a=s.index('for name,v in read(w/\'reused-artifact-provenance.json\')')
extra='''# Preserve the second rejection: first retained trace passes, second handoff OOMs.
r2=w/'rejected-0d5f50ff';f2=read(r2/'failed-attempt.json');b2=read(r2/'candidate-build.json')
assert f2['candidate']==b2['commit']=='0d5f50ff0fc1d725fc2fdd5a23579427647d627a'
assert f2['candidate_sha256']==b2['executable_sha256']=='1ace3c5031d832d34eb0064fdfd71b3feeb0a0c3000d568496b4828eb82548c5'
assert f2['passed_native_gates']==10 and f2['failed_gate']=='retained-256k'
assert f2['retained_256k_full_output_cycles_passed']==1 and f2['second_cycle_prefill_full_output_comparison_passed_before_handoff']
assert f2['second_cycle_decode_steps_completed']==0 and not f2['retry_performed']
assert original_sha(r2/'checks/retained-256k.log')==f2['log_sha256']
assert original_sha(r2/'validation-progress.json')==f2['validation_progress_sha256']
assert not (r2/'checks/retained-256k.json').exists() and not (r2/'validation-acceptance.json').exists()
for name,h in b2['sources'].items():assert sha(r2/'candidate-source'/name)==h
for field,file in [('tests','tests.log'),('build','build.log'),('python_tests','python-tests.log')]:assert original_sha(r2/file)==b2[field+'_log_sha256']
p2=read(r2/'validation-progress.json');assert len(p2)==11 and [x['exit'] for x in p2]==[0]*10+[1]
for c in p2:
 log=(r2/'checks'/(c['name']+'.log')).read_text();rss=re.search(r'(\\d+)\\s+maximum resident set size',log);assert rss and int(rss[1])==c['peak_rss_bytes']
failed2=(r2/'checks/retained-256k.log').read_text();assert f2['error'] in failed2
positions=[(int(p),int(t)) for p,t in re.findall(r'decode position (\\d+) all 129280 logits exact, token (\\d+)',failed2)]
ts=struct.unpack('<129I',(w/'reference-262144/selected.u32le.bin').read_bytes());assert positions==list(zip(range(262144,262272),ts[1:]))
assert 'rust_star_metal_adopt_prefill_decoder_state' in oldshim
for x in [json.loads(line) for line in (r2/'memory-observations.jsonl').read_text().splitlines()]:
 assert x['exit']==0 and x['page_size_bytes']==16384
 assert x['pages']=={k:int(v) for k,v in re.findall(r'^([^\\n:]+):\\s+(\\d+)\\.',x['raw_stdout'],re.M)}
'''
s=s[:a]+extra+s[a:]
s=s.replace("'rejected_predecessor_verified':True", "'rejected_predecessors_verified':2,'full_size_handoff_copy_regression_checked':True")
p.write_text(s)
p=w/'package.py';s=p.read_text().replace("tree(w/'rejected-c1047756',r/'rejected-c1047756')", "tree(w/'rejected-c1047756',r/'rejected-c1047756')\ntree(w/'rejected-0d5f50ff',r/'rejected-0d5f50ff')")
p.write_text(s)
