"""Audit this bounded kernel study without Metal or the model."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics

w = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda name: json.loads((w / name).read_text())
artifacts = read('artifact-hashes.json')['artifacts']
for name, h in artifacts.items():
    assert sha(w / name) == h, name
plan, build, status = read('hypothesis.json'), read('build.json'), read('run-status.json')
assert sha(w / 'hypothesis.json') == build['hypothesis_sha256']
assert sha(w / 'route-order-probe.m') == build['probe_source_sha256']
assert sha(w / 'build.log') == build['build_log_sha256']
for name, h in plan['generated_sha256'].items():
    assert sha(w / name) == h, name
assert status['exit_code'] == 0 and not (w / 'probe.stderr').read_text()
assert plan['created_utc'] < status['started_utc'] < status['ended_utc']
assert plan['runtime_mutation'] is False and plan['no_full_model_jobs'] is True
assert plan['dimensions'] == dict(experts=256, top_k=6, input_width=4096, output_width=2048,
                                  threadgroup_threads=128, threadgroup_bytes=16384)

# Prove the compiled candidate differs only at the two grid-coordinate reads.
base = (w / 'baseline.metal').read_text()
start = base.index('kernel void kernel_mul_mm_id_pair_swiglu_f16_impl(')
end = base.index('\ntypedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl', start)
body = base[start:end]
new = body
for a, b in [('const uint32_t work_index = tgpig.x;', 'const uint32_t work_index = tgpig.y;'),
             ('const int r0 = tgpig.y*NR0;', 'const int r0 = tgpig.x*NR0;')]:
    assert new.count(a) == 1
    new = new.replace(a, b, 1)
assert base.replace(body, new, 1) == (w / 'candidate.metal').read_text()

# Recompute attribution from the accepted raw 32K profile, not its summary.
selection, profile = read('profile-selection.json'), read('accepted-32k-profile.json')
assert sha(w / 'accepted-32k-profile.json') == selection['source_sha256']
assert profile['paired_ineligible'] and profile['tail_components']['profile_perturbed']
assert profile['tail_components']['encoder_serialization'] == 'MTLFence'
stages, components, packets = defaultdict(float), defaultdict(float), defaultdict(list)
parents = {}
for r in profile['rows']:
    assert math.isfinite(r['gpu_ms']) and r['gpu_ms'] >= 0
    stages[r['stage']] += r['gpu_ms']
    if r['stage'] == 'tail':
        k = (r['chunk_start'], r['first_layer'])
        assert k not in parents
        parents[k] = r['gpu_ms']
for r in profile['tail_components']['rows']:
    packets[r['chunk_start'], r['layer']].append(r)
    components[r['component']] += r['attributed_gpu_ms']
assert set(parents) == set(packets) == {(c, l) for c in range(4096,32768,4096) for l in range(2,43)}
names = ['attention_output','ffn_ingress','router','routed_map','routed_gate_up','routed_down','routed_sum','shared_expert','final_hc']
for key, group in packets.items():
    assert [r['component'] for r in group] == names
    last = -1
    ticks = sum(r['end_tick']-r['start_tick'] for r in group)
    for r in group:
        assert last < r['start_tick'] < r['end_tick'] < 2**64-1
        assert abs(parents[key]*(r['end_tick']-r['start_tick'])/ticks-r['attributed_gpu_ms']) < 1e-6
        last = r['end_tick']
    assert abs(sum(r['attributed_gpu_ms'] for r in group)-parents[key]) < 1e-6
assert dict(stages) == selection['stage_gpu_ms']
assert dict(components) == selection['tail_attributed_gpu_ms']
share = components['routed_gate_up']/sum(stages.values())

rows = [json.loads(line) for line in (w / 'probe.jsonl').read_text().splitlines()]
assert rows[0]['type'] == 'start' and rows[0]['device'] == 'Apple M1 Ultra'
assert rows[1]['type'] == 'weights' and rows[1]['bytes_each'] == 256*2048*1056
assert rows[-1]['type'] == 'complete' and rows[-1]['passed']
cases = [r for r in rows if r['type'] == 'case']
expected = [(n,'balanced',False) for n in plan['controls']] + [(r['rows'],r['routing'],True) for r in plan['timed_cases']]
assert [(r['rows'],r['routing'],r['timed']) for r in cases] == expected
all_samples = [r for r in rows if r['type'] == 'sample']
assert len(rows) == 3 + len(cases) + len(all_samples)
totals = 0
summaries = []
for case in cases:
    n, route, timed = case['rows'], case['routing'], case['timed']
    assert case['words'] == n*6*2048
    assert case['work_capacity'] == (n*6+31*256+31)//32
    assert 0 < case['active_tiles'] <= case['work_capacity']
    assert 0 < case['negative_words'] < case['words'] and case['nonzero_words'] > case['words']/10
    sample = [r for r in all_samples if (r['rows'],r['routing']) == (n,route)]
    expected_samples = []
    for iteration in range(9 if timed else 1):
        phase = ('warmup' if iteration < 2 else 'timed') if timed else 'control'
        order = 'AB' if iteration%2 == 0 else 'BA'
        for variant in order:
            expected_samples.append((phase, iteration-2 if timed else iteration, order, variant))
    assert [(r['phase'],r['round'],r['order'],r['variant']) for r in sample] == expected_samples
    for r in sample:
        assert r['exact'] is True and r['guards_intact'] is True
        assert r['words_checked'] == case['words']
        assert math.isfinite(r['gpu_ms']) and r['gpu_ms'] > 0
        totals += r['words_checked']
    if not timed:
        continue
    pairs = defaultdict(dict)
    for r in sample:
        if r['phase'] == 'timed':
            assert r['variant'] not in pairs[r['round']]
            pairs[r['round']][r['variant']] = r['gpu_ms']
    assert sorted(pairs) == list(range(7)) and all(set(p) == {'A','B'} for p in pairs.values())
    ratios = [pairs[i]['B']/pairs[i]['A'] for i in range(7)]
    def dist(values):
        median = statistics.median(values)
        return dict(median=median, mad=statistics.median(abs(v-median) for v in values),
                    minimum=min(values), maximum=max(values))
    med = statistics.median(ratios)
    summaries.append(dict(rows=n,routing=route,pairs=[dict(pair=i+1,**pairs[i],ratio=ratios[i]) for i in range(7)],
                          baseline_gpu_ms=dist([p['A'] for p in pairs.values()]),
                          candidate_gpu_ms=dist([p['B'] for p in pairs.values()]),paired_latency_ratio=dist(ratios),
                          faster_pairs=sum(v<1 for v in ratios),
                          hypothetical_stage_gpu_reduction_fraction=share*(1-med)))
assert totals == sum(r['words_checked'] for r in all_samples)
eligible = all(s['paired_latency_ratio']['median'] < 1 and s['faster_pairs'] >= 5 for s in summaries if s['rows']==4096)
result = dict(passed=True,screening_criteria_met=eligible,production_changed=False,full_model_c0_claim=False,
              engine_throughput_claim=False,profile_routed_gate_up_share=share,
              cases=len(cases),verified_output_comparisons=len(all_samples),fp16_words_compared=totals,
              max_rss_bytes=rows[-1]['max_rss_bytes'],timed_cases=summaries)
assert result == read('result.json'), 'Recorded result differs from independent recomputation'
print(json.dumps(dict(passed=True,artifacts_verified=len(artifacts),screening_criteria_met=eligible,
                     fp16_words_compared=totals,
                     median_paired_ratios=[s['paired_latency_ratio']['median'] for s in summaries]),indent=2))
