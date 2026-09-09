from pathlib import Path
import json,hashlib,sys,subprocess
r=Path(sys.argv[1]).resolve();d=r/'stopped-metadata-attempt';read=lambda p:json.loads(p.read_text())
stop=read(d/'validation-stopped-for-metadata.json');assert stop['completed_gates']==6 and stop['completed_gates_all_passed'] and stop['active_gate']=='retained-32k' and not stop['active_gate_complete'] and not stop['timed_samples_started']
progress=read(d/'validation-progress.json');assert [x['name'] for x in progress]==['batch-2k','batch-8k','batch-12k','batch-32k-output','batch-32k-internal','batch-32k-profiled']
assert all(x['exit']==0 and x['seconds']>0 and x['peak_rss_bytes']>0 for x in progress)
assert not (d/'checks/retained-32k.json').exists() and (d/'checks/retained-32k.log').exists()
x=read(d/'checks/batch-8k.json');assert x['complete_8k_transformer_claim'] and x['output_logits_c0_claim']
assert x['sparse_transition']['score_kernel']=='kernel_dsv4_indexer_scores_tiled' and x['layer4_prefix_complete']['sparse_attention']['kernels'][0]=='kernel_dsv4_indexer_scores_tiled'
assert read(d/'checks/batch-2k.json')['output_head']['full_logits_c0_bitwise_match']
assert read(d/'checks/batch-12k.json')['chunks']==read(r/'prior-controls/batch-12k.json')['chunks']
for name in ['batch-32k-output','batch-32k-internal','batch-32k-profiled']:
 x=read(d/'checks'/(name+'.json'));assert x['complete_32k_c0_claim'] and x['checked_decode_steps']==128 and x['logits_checksum']==2996662110144943749 and x['decode_logits_checksum']==1716816788919264169 and x['decode_tokens_checksum']==11977321378026793466
assert read(d/'checks/component-audit.json')['passed']
before=hashlib.sha256((d/'checks/component-audit.json').read_bytes()).hexdigest()
subprocess.run([sys.executable,str(d/'checks/audit-components.py')],check=True,stdout=subprocess.DEVNULL)
assert hashlib.sha256((d/'checks/component-audit.json').read_bytes()).hexdigest()==before
log=(d/'writer-regression-before-fix.log').read_text();assert 'FAILED' in log and 'left: 0' in log and 'right: 2' in log and '316 filtered out' in log
b=read(d/'candidate-build.json');assert b['commit'].startswith('7b1a5b58')
for n,h in b['sources'].items():assert hashlib.sha256((r/'stopped-source'/n).read_bytes()).hexdigest()==h
old=(r/'stopped-source/rust-star/runtime/src/metal.rs').read_text();new=(r/'candidate-source/rust-star/runtime/src/metal.rs').read_text()
oldprefix=old.split('\n#[cfg(test)]\nmod tests',1)[0];newprefix=new.split('\n#[cfg(test)]\nmod tests',1)[0]
assert oldprefix.count('kernel_dsv4_indexer_scores_tiled')==2
assert oldprefix.replace('kernel_dsv4_indexer_scores_tiled','rust_star_indexer_scores_tiled64')==newprefix
print('PASS stopped metadata attempt: six numerical gates, partial lifecycle excluded, red regression and exact writer-only correction')
