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
summary = read(results / 'paired-summary.json')
state = read(results / 'state.json')
audit = read(ROOT / 'independent-audit.json')
assert state['phase'] == 'complete' and audit['passed']
assert len(raw['pairs']) == 5
assert [pair['order'] for pair in raw['pairs']] == ['AB', 'BA', 'AB', 'BA', 'AB']
assert all(pair['valid'] and pair['attempt'] == 1 for pair in raw['pairs'])
assert digest(results / 'paired-raw.json') == audit['raw_sha256']
assert digest(results / 'paired-summary.json') == audit['summary_sha256']
assert digest(ROOT / 'inputs/host-before.json') == raw['host_manifest_sha256']
assert digest(ROOT / 'inputs/correctness-manifest.json') == raw['correctness_manifest_sha256']
context = summary['contexts'][0]
for role in ('oracle', 'candidate'):
    for metric, expected in context[role].items():
        assert distribution([pair[role][metric] for pair in raw['pairs']]) == expected, metric
    evidence = [pair['evidence'][role] for pair in raw['pairs']] + state['warmups'][role]
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
                         for pair in raw['pairs']]) == expected, metric
profile = read(ROOT / 'profile32k.json')
assert digest(ROOT / 'profile32k.json') == audit['profile_sha256']
assert not profile['paired_protocol_eligible'] and profile['gen_tokens'] == 128
assert profile['selection']['selected_tokens_checksum'] == 11977321378026793466
assert profile['selection']['oracle_transcript_match']
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
checks=read(proof/'batch-validation.json');assert len(checks)==6 and all(x['exit']==0 for x in checks)
for name in ('batch-32k-output','batch-32k-internal'):
 r=read(proof/(name+'.json'));assert r['complete_32k_c0_claim'] and r['checked_decode_steps']==128
 assert r['logits_checksum']==prefill_checksum and r['decode_logits_checksum']==decode_checksum and r['decode_tokens_checksum']==word_checksum((decode/'token-ids.u32le.bin').read_bytes()[4:])
retained=read(proof/'batch-retained.json');assert retained['selection']['both_oracle_transcripts_match'] and not retained['paired_protocol_eligible']
assert '310 passed; 0 failed' in (proof/'batch-tests-fixed.log').read_text()
assert 'Ran 93 tests' in (proof/'python-tests.log').read_text() and (proof/'python-tests.log').read_text().rstrip().endswith('OK')
assert read(ROOT/'profile32k-analysis.json')['passed']
rows=profile['long_prefill_profile']['rows'];assert {r['chunk_start'] for r in rows}==set(range(0,32768,4096))
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
