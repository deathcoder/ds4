#!/usr/bin/env python3
"""Verify this portable evidence bundle without loading the model or running Metal."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    median = statistics.median(values)
    return dict(count=len(values), median=median,
                mad=statistics.median(abs(value - median) for value in values),
                min=min(values), max=max(values))


manifest = read(ROOT / 'artifact-hashes.json')
actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob('*')
          if path.is_file() and path.name != 'artifact-hashes.json'
          and '__pycache__' not in path.parts}
assert actual == set(manifest['artifacts']), 'Missing or unexpected evidence'
for name, expected in manifest['artifacts'].items():
    path = ROOT / name
    assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], name

results = ROOT / 'results'
raw = read(results / 'paired-raw.json')
valid_pairs=[pair for pair in raw['pairs'] if pair['valid']]
invalid_raw=[pair for pair in raw['pairs'] if not pair['valid']]
assert len(valid_pairs)==5 and len(invalid_raw)==1
assert (invalid_raw[0]['context'],invalid_raw[0]['repetition'],invalid_raw[0]['attempt'])==(32768,4,1)
assert invalid_raw[0]['invalid_reason'].startswith('Host shutdown/restart interrupted pair 4 attempt 1')

summary = read(results / 'paired-summary.json')
state = read(results / 'state.json')
audit = read(ROOT / 'independent-audit.json')
assert state['phase'] == 'complete' and audit['passed']
assert len(valid_pairs) == 5
assert [pair['order'] for pair in valid_pairs] == ['AB', 'BA', 'AB', 'BA', 'AB']
assert audit['invalid_attempts']==1
assert all(pair['valid'] and pair['attempt']==(2 if pair['repetition']==4 else 1) for pair in valid_pairs)
invalid=[a for a in state['attempts'] if a['status']=='invalid']
assert len(invalid)==1
bad=invalid[0]
assert (bad['context'],bad['repetition'],bad['attempt'])==(32768,4,1)
assert bad['invalid_reason'].startswith('Host shutdown/restart interrupted pair 4 attempt 1')
interruption=read(ROOT/'inputs/restart-interruption.json')
snapshot=ROOT/'inputs/state-before-restart-recovery.json'
assert digest(snapshot)==interruption['interrupted_state_sha256']
assert interruption['confirmed_no_benchmark_processes'] and interruption['candidate_and_oracle_binaries_rehashed']
before=read(snapshot)['attempts'][-1]
assert before['status']=='running' and before['engines']['oracle']['status']=='running'
assert {k:v for k,v in bad.items() if k not in ('status','invalid_reason','invalidated_at_utc')}=={k:v for k,v in before.items() if k!='status'}
e=bad['engines']['candidate'];directory=results/e['output'];measurement=directory/'measurement.json'
assert digest(measurement)==e['measurement_sha256']
m=read(measurement);assert m['status']=='passed' and m['metrics']==e['metrics']
for artifact in m['artifacts'].values():
 file=directory/artifact['path'];assert file.stat().st_size==artifact['bytes'] and digest(file)==artifact['sha256']

assert digest(results / 'paired-raw.json') == audit['raw_sha256']
assert digest(results / 'paired-summary.json') == audit['summary_sha256']
assert digest(ROOT / 'inputs/host-before.json') == raw['host_manifest_sha256']
assert digest(ROOT / 'inputs/correctness-manifest.json') == raw['correctness_manifest_sha256']
context = summary['contexts'][0]
for role in ('oracle', 'candidate'):
    for metric, expected in context[role].items():
        assert distribution([pair[role][metric] for pair in valid_pairs]) == expected, metric
    evidence = [pair['evidence'][role] for pair in valid_pairs] + state['warmups'][role]
    assert len(evidence) == 6
    for record in evidence:
        directory = results / record['output']
        measurement = directory / 'measurement.json'
        assert digest(measurement) == record['measurement_sha256']
        data = read(measurement)
        assert data['status'] == 'passed'
        for artifact in data['artifacts'].values():
            path = directory / artifact['path']
            assert path.stat().st_size == artifact['bytes'] and digest(path) == artifact['sha256']
        if role == 'candidate':
            engine = read(directory / 'engine-run.json')
            assert engine['selection']['oracle_transcript_match']
            assert engine['selection']['selected_tokens_checksum'] == 11977321378026793466
for metric, expected in context['candidate_over_oracle'].items():
    assert distribution([pair['candidate'][metric] / pair['oracle'][metric]
                         for pair in valid_pairs]) == expected, metric
failure=read(ROOT/'profile-failure.json')
assert audit['profile_status']=='failed' and audit['profile_sha256'] is None
assert not audit['profile_stage_shares_verified']
assert digest(ROOT/'profile-failure.json')==audit['profile_failure_sha256']
assert failure['exit_code']==1 and not failure['profile_artifact_written']
assert digest(ROOT/'profile32k.log')==failure['log_sha256']
assert 'invalid tail timestamps' in (ROOT/'profile32k.log').read_text()
assert not (ROOT/'profile32k.json').exists()
print(f"PASS: {len(actual)} artifacts, 5 valid pairs, 2 warmups, exact transcripts and independent statistics")

# V5 reference, selector attribution, full tensors and lifecycle evidence.
import struct
import math
proof=ROOT/'correctness'
acceptance=read(proof/'oracle-v5-acceptance/manifest.json')
assert acceptance['status']=='complete' and acceptance['conformance']['status']=='passed'
assert acceptance['source']['commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'
assert acceptance['source']['tree']=='a0a33f4a00c3c78cd40350785485203e7c6b0b1d'
assert acceptance['correctness']['status']=='passed'
for ctx in (2048,32768):
 runs=[x for x in acceptance['conformance']['runs'] if x['context']==ctx]
 assert sorted(x['repetition'] for x in runs)==[1,2,3,4]
 assert len({x['logits']['sha256'] for x in runs})==1
 for run in runs:assert digest(proof/'oracle-v5-acceptance'/run['logits']['path'])==run['logits']['sha256']
a=read(proof/'trace-a.json');b=read(proof/'trace-b.json');assert a['hashes']==b['hashes'] and a['rows']==b['rows']==129
trace=read(proof/'conformance-v5.json');assert trace['passed'] and trace['fresh_processes']==2 and trace['full_fp32_rows_per_run']==129
assert trace['executable_sha256']==raw['oracle']['executable_sha256']
frontier=proof/'fixtures/prefill-frontier-32768-v5';decode=proof/'fixtures/prefill-decode-32768-v5'
prefill=(frontier/'batch-prefill-logits.f32le.bin').read_bytes();decoded=(decode/'decode-logits.f32le.bin').read_bytes();tokens=struct.unpack('<129I',(decode/'token-ids.u32le.bin').read_bytes())
assert len(prefill)==129280*4 and len(decoded)==128*129280*4
for run in acceptance['conformance']['runs']:
 values=read(proof/'oracle-v5-acceptance'/run['logits']['path'])['logits'];payload=struct.pack('<129280f',*values)
 if run['context']==32768:assert payload==prefill
 else:assert hashlib.sha256(payload).hexdigest()=='7b5e851884bbb0aa8c2a249c8497af0feccb267cbd0a40e0a4a5aee584ecbfaf'
for directory in (frontier,decode):
 fixture=read(directory/'manifest.json');assert fixture['oracle']['id']=='oracle-v5' and fixture['oracle']['commit']==acceptance['source']['commit']
 for tensor in fixture['tensors']:assert digest(directory/tensor['path'])==tensor['sha256'] and (directory/tensor['path']).stat().st_size==tensor['bytes']
for step in range(129):
 row=prefill if step==0 else decoded[(step-1)*517120:step*517120]
 assert hashlib.sha256(row).hexdigest()==a['hashes'][str(step)]['logits_sha256']
 values=struct.unpack('<129280f',row);assert all(map(math.isfinite,values));selected=max(range(129280),key=values.__getitem__)
 assert selected==tokens[step]==a['hashes'][str(step)]['selected_token'] and selected!=1
assert (proof/'selector-control-32768.f32le.bin').read_bytes()==prefill
assert read(proof/'selector-attribution.json')['differing_words']==0
assert read(proof/'tokenization-v5.json')['token_sha256']==digest(frontier/'token-ids.u32le.bin')
assert read(proof/'oracle-v5-8192.json')['passed']
def word_checksum(payload):
 value=0xcbf29ce484222325
 for (word,) in struct.iter_unpack('<I',payload):value=((value^word)*0x100000001b3)&0xffffffffffffffff
 return value
prefill_checksum=word_checksum(prefill);decode_checksum=word_checksum(decoded)
checks=read(proof/'batch-validation.json');assert len(checks)==7 and all(x['exit']==0 for x in checks)
for name in ('batch-32k-output','batch-32k-internal'):
 r=read(proof/(name+'.json'));assert r['complete_32k_c0_claim'] and r['checked_decode_steps']==128
 assert r['logits_checksum']==prefill_checksum and r['decode_logits_checksum']==decode_checksum and r['decode_tokens_checksum']==word_checksum((decode/'token-ids.u32le.bin').read_bytes()[4:])
retained=read(proof/'batch-retained.json');assert retained['selection']['both_oracle_transcripts_match'] and not retained['paired_protocol_eligible']
assert '312 passed; 0 failed' in (proof/'batch-tests-fixed.log').read_text()
assert 'Ran 93 tests' in (proof/'python-tests.log').read_text() and (proof/'python-tests.log').read_text().rstrip().endswith('OK')
assert read(ROOT/'profile-failure.json')['status']=='failed'
rows=read(proof/'components.json')['rows'];assert {r['chunk_start'] for r in rows}==set(range(0,32768,4096))
for start in range(4096,32768,4096):assert {layer for r in rows if r['chunk_start']==start for layer in range(r['first_layer'],r['last_layer']+1)}==set(range(43))
print('PASS: pinned v5 acceptance, selector attribution, all129 fixture rows, two live full-output gates and retained lifecycle')

hc=read(proof/'hc-v5-acceptance.json');assert hc['passed'] and hc['fresh_repetitions_per_context']==2 and hc['contexts']==[12288,16384,20480,24576]
for end in hc['contexts']:
 a=read(proof/f'long-prefill-chunks-v5/frontier-{end}-a.json');b=read(proof/f'long-prefill-chunks-v5/frontier-{end}-b.json');assert a==b and len(a)==43
 for value in a.values():assert value['bytes']==268435456 and len(value['sha256'])==64
 for rep in ['a','b']:
  record=next(x for x in hc['records'] if x['frontier']==end and x['repeat']==rep);assert digest(proof/f'long-prefill-chunks-v5/frontier-{end}-{rep}.json')==record['hashes_sha256']
 actual=next(x for x in read(proof/'batch-32k-internal.json')['chunks'] if x['end']==end)
 assert actual['hc_c0_match'] and actual['hc_checked_layers']==43
 assert actual['hc_checksums']==[int(a[f'oracle_hc_ffn_post-{layer}_pos{end-4096}.bin']['fnv1a_u32'],16) for layer in range(43)]
old=read(proof/'historical-hc-mismatch/batch-validation.json');assert old[-1]['name']=='batch-32k-internal' and old[-1]['exit']==1
assert read(proof/'historical-hc-mismatch/batch-32k-output.json')['complete_32k_c0_claim']
print('PASS: explicitly versioned repeated internal v5 checkpoints and retained historical mismatch')

selector=read(proof/'selector-v5-acceptance.json');assert selector['passed'] and selector['fresh_repetitions']==2 and selector['rows'][0]==selector['rows'][1]
d=proof/'fixtures/prefill-indexer-selector-4128-v5'
indices=struct.unpack('<16384I',(d/'topk-tile.i32le.bin').read_bytes());old_indices=struct.unpack('<16384I',(proof/'historical-second-tile-topk.i32le.bin').read_bytes());scores=struct.unpack('<65536f',(d/'scores-tile-bits.i32le.bin').read_bytes())
changed=[i for i in range(16384) if indices[i]!=old_indices[i]];assert len(changed)==4
assert all(scores[(i//512)*2048+indices[i]]==scores[(i//512)*2048+old_indices[i]] for i in changed)
assert all(set(indices[i*512:(i+1)*512])==set(old_indices[i*512:(i+1)*512]) for i in range(32))
assert digest(d/'topk-tile.i32le.bin')==selector['rows'][0]['topk_sha256']
assert digest(d/'scores-tile-bits.i32le.bin')==selector['rows'][0]['scores_sha256']
assert read(proof/'selector-ordering-mismatch/batch-validation.json')[-1]['exit']==1
assert read(proof/'batch-8k.json')['second_consecutive_tile']['selector_fixture']=='dwarfstar-oracle-v5-prefill-indexer-selector-4128'
print('PASS: independently repeated equal-score selector ordering, with historical targets and failure preserved')

d=proof/'fixtures/prefill-indexer-selectors-8192-v5';a=read(d/'hashes-a.json');b=read(d/'hashes-b.json');assert a==b and len(a)==21
assert digest(d/'layer4-topk.i32le.bin')==a['4']['indexer_topk']['sha256']
checksum=word_checksum((d/'layer4-topk.i32le.bin').read_bytes());assert checksum==int(a['4']['indexer_topk']['fnv1a_u32'],16)
assert read(proof/'batch-8k.json')['continuation_selector_inventory']=={'fixture':'prefill-indexer-selectors-8192-v5','layer4_topk_checksum':checksum}
assert read(proof/'full-selector-ordering-mismatch/batch-validation.json')[-1]['exit']==1
print('PASS: full continuation selector inventory and actual Rust checksum')

assert read(proof/'remaining-dispatch-mismatch/batch-validation.json')[-1]['exit']==1
print('PASS: retained remaining-layer metadata failure and corrected final live gate')

# The default v5 exporter must package the same pinned sources and new headers.
packaging=read(proof/'default-v5-builder-verification.json')
rebuilt=read(proof/'default-v5-build.json')
measured=read(proof/'oracle-v5-benchmark-build.json')
assert packaging['passed'] and packaging['build_exit']==0
assert digest(proof/'build_benchmark_oracle.py')==packaging['builder_sha256']
assert digest(proof/'default-v5-build.json')==packaging['build_provenance_sha256']
assert rebuilt['oracle_id']=='oracle-v5' and rebuilt['inference_source_commit']==acceptance['source']['commit']
assert rebuilt['benchmark_sha256']==measured['benchmark_sha256']
assert all(rebuilt['source_sha256'][name]==value for name,value in measured['source_sha256'].items())
assert {'third_party/iris/LICENSE','third_party/iris/jpeg.h','third_party/iris/png.h'}<=set(rebuilt['source_sha256'])
assert 'Ran 93 tests' in (proof/'final-python-tests.log').read_text() and (proof/'final-python-tests.log').read_text().rstrip().endswith('OK')
assert 'No rule to make target' in (proof/'default-v5-builder.log').read_text()
print('PASS: default v5 builder, matching pinned sources, third-party dependencies and final Python suite')

assert sum(r['calls'] for r in read(proof/'components.json')['rows'] if r['stage']=='attention' and r['first_layer']==r['last_layer'] and r['first_layer']%2==0)==147
assert raw['candidate']['runtime_configuration']['even_prefill_attention_tile_rows']==4096
assert raw['oracle']['source_commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'
assert digest(ROOT/'baseline-v5-summary.json')==read(ROOT/'baseline-v5-audit.json')['summary_sha256']
assert read(ROOT/'baseline-v5-profile.json')['profile_sha256']==read(ROOT/'baseline-v5-audit.json')['profile_sha256']
print('PASS: complete-chunk attention scheduling and unchanged oracle-v5')

assert sum(r['calls'] for r in read(proof/'components.json')['rows'] if r['stage']=='attention' and r['first_layer']==r['last_layer'] and r['first_layer']%2==0)==147
assert raw['candidate']['runtime_configuration']['odd_prefill_ingress_rows']==4096
assert raw['candidate']['runtime_configuration']['even_prefill_attention_tile_rows']==4096
assert raw['oracle']['source_commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'
assert digest(ROOT/'baseline-attention-summary.json')==read(ROOT/'baseline-attention-audit.json')['summary_sha256']
assert read(ROOT/'baseline-attention-profile.json')['profile_sha256']==read(ROOT/'baseline-attention-audit.json')['profile_sha256']
assert read(ROOT/'correctness/ingress-metadata-attempt-05447277/batch-validation.json')[0]['exit']!=0
print('PASS: complete-chunk ingress scheduling, retained metadata failure and unchanged oracle-v5')

assert 'odd layer 3 schedule mismatch: 48/28/28' in (ROOT/'correctness/ingress-metadata-attempt-05447277/batch-12k.log').read_text()

assert sum(r['calls'] for r in read(proof/'components.json')['rows'] if r['stage']=='attention' and r['first_layer']==r['last_layer'] and r['first_layer']%2==0)==147
config=raw['candidate']['runtime_configuration']
assert config['native_tail_scratch_reuse'] and config['native_tail_direct_hc_output']
assert not config['native_tail_routed_mid_f32_conversion'] and config['native_tail_dispatches']==31
assert raw['candidate']['runtime_configuration']['odd_prefill_ingress_rows']==4096
assert raw['candidate']['runtime_configuration']['even_prefill_attention_tile_rows']==4096
assert raw['oracle']['source_commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'
assert digest(ROOT/'baseline-ingress-summary.json')==read(ROOT/'baseline-ingress-audit.json')['summary_sha256']
assert read(ROOT/'baseline-ingress-profile.json')['profile_sha256']==read(ROOT/'baseline-ingress-audit.json')['profile_sha256']
assert read(ROOT/'correctness/ingress-metadata-attempt-05447277/batch-validation.json')[0]['exit']!=0
print('PASS: native tail scratch/output contract and unchanged oracle-v5')

assert sum(r['calls'] for r in read(proof/'components.json')['rows'] if r['stage']=='attention' and r['first_layer']==r['last_layer'] and r['first_layer']%2==0)==147
config=raw['candidate']['runtime_configuration']
assert config['native_tail_scratch_reuse'] and config['native_tail_direct_hc_output']
assert not config['native_tail_routed_mid_f32_conversion'] and config['native_tail_dispatches']==31
assert raw['candidate']['runtime_configuration']['odd_prefill_ingress_rows']==4096
assert raw['candidate']['runtime_configuration']['even_prefill_attention_tile_rows']==4096
assert raw['oracle']['source_commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'
assert digest(ROOT/'baseline-tail-summary.json')==read(ROOT/'baseline-tail-audit.json')['summary_sha256']
assert read(ROOT/'baseline-tail-profile.json')['profile_sha256']==read(ROOT/'baseline-tail-audit.json')['profile_sha256']
assert read(ROOT/'correctness/ingress-metadata-attempt-05447277/batch-validation.json')[0]['exit']!=0
print('PASS: native tail scratch/output contract and unchanged oracle-v5')

assert config['native_tail_scratch_buffers']==20
assert config['native_output_low_projection']=='eight aligned dense Q8 groups with direct strided stores'
import subprocess,sys
subprocess.run([sys.executable,str(proof/'audit-components.py')],check=True,stdout=subprocess.DEVNULL)
assert read(proof/'component-audit.json')['passed']
assert read(proof/'batch-32k-profiled.json')['complete_32k_c0_claim']
print('PASS: direct dense Q8 projection and exact-gated component profile')

import collections
lp=read(ROOT/'correctness/components.json')
parents={(r['chunk_start'],r['first_layer']):r for r in lp['rows'] if r['stage']=='tail'}
parts=collections.defaultdict(list)
for r in lp['tail_components']['rows']:parts[r['chunk_start'],r['layer']].append(r)
assert set(parts)==set(parents)=={(c,l) for c in range(4096,32768,4096) for l in range(2,43)}
names=['attention_output','ffn_ingress','router','routed_map','routed_gate_up','routed_down','routed_sum','shared_expert','final_hc']
for key,group in parts.items():
 assert [r['component'] for r in group]==names
 assert all(isinstance(r['start_tick'],int) and isinstance(r['end_tick'],int) and r['end_tick']>r['start_tick'] for r in group)
 assert all(b['start_tick']>a['end_tick'] for a,b in zip(group,group[1:]))
 total=sum(r['end_tick']-r['start_tick'] for r in group)
 assert abs(sum(r['attributed_gpu_ms'] for r in group)-parents[key]['gpu_ms'])<1e-6
 for r in group:assert abs(r['attributed_gpu_ms']-parents[key]['gpu_ms']*(r['end_tick']-r['start_tick'])/total)<1e-6
assert len(lp['tail_components']['rows'])==2583
print('PASS: premeasurement exact-gated component accounting; isolated profile failure preserved')

restored=read(ROOT/'restored-runtime-check.json')
assert restored['status']=='passed' and restored['rust_tests_passed']==312 and restored['release_build_passed']
assert restored['source_restored_from'].startswith('9c1d5009')
assert '312 passed; 0 failed' in (ROOT/'restored-tests.log').read_text()
assert 'Finished `release` profile' in (ROOT/'restored-build.log').read_text()
assert len(restored['exactly_restored_files'])==2 and len(restored['rebuilt_executable_sha256'])==64
assert (ROOT/'restoration.patch').stat().st_size>0
print('PASS: exact source-restoration record and fresh 312-test/release build evidence')
