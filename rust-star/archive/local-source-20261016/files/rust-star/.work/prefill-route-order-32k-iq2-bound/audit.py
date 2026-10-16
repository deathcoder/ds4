"""Independently audit complete 32K outputs and the matched diagnostic pair."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics

w = Path(__file__).resolve().parent
read = lambda n: json.loads((w/n).read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = read('plan.json')
assert read('validation-acceptance.json')['passed']
checks = {}
for name in plan['live_sequence']:
    result = read(name+'.json')
    assert result == dict(schema='rust-star-long-prefill-chunk-probe-v1',frontier_tokens=32768,
        internal_hc_collection=False,chunks=[],checked_decode_steps=128,
        decode_logits_checksum=1716816788919264169,decode_tokens_checksum=11977321378026793466,
        checked_logits=129280,logits_checksum=2996662110144943749,selected_token=85166,
        complete_32k_c0_claim=True,throughput_claim=False), name
    log = (w/(name+'.log')).read_text()
    assert log.count('all 129280 logits exact, token ') == 128, name
    checks[name] = dict(passed=True,fp32_words=16677120,result_sha256=sha(w/(name+'.json')),
                        log_sha256=sha(w/(name+'.log')))

names = ['attention_output','ffn_ingress','router','routed_map','routed_gate_up','routed_down','routed_sum','shared_expert','final_hc']
expected = {(chunk,layer) for chunk in range(4096,32768,4096) for layer in range(2,43)}
profiles = {}
for role in ['baseline','candidate']:
    filename = role+'-32k-profiled-components.json'
    p = read(filename)
    assert p['paired_ineligible'] is True
    t = p['tail_components']
    assert t['profile_perturbed'] and t['encoder_splitting'] and t['encoder_serialization']=='MTLFence'
    parents = {}
    stages = defaultdict(lambda:dict(gpu_ms=0.,host_elapsed_ms=0.,command_wall_ms=0.,calls=0))
    for row in p['rows']:
        if row['stage']=='tail':
            k = (row['chunk_start'],row['first_layer'])
            assert k not in parents and row['first_layer']==row['last_layer'] and row['calls']==1
            parents[k] = row
        for metric in ['gpu_ms','host_elapsed_ms','command_wall_ms']:
            assert math.isfinite(row[metric]) and row[metric]>=0
            stages[row['stage']][metric]+=row[metric]
        stages[row['stage']]['calls']+=row['calls']
    groups = defaultdict(list)
    for row in t['rows']:
        groups[row['chunk_start'],row['layer']].append(row)
    assert set(groups)==set(parents)==expected and len(t['rows'])==2583
    components = defaultdict(float)
    tails = {}
    coverage = []
    for key, group in groups.items():
        assert [r['component'] for r in group]==names
        last=-1
        ticks=sum(r['end_tick']-r['start_tick'] for r in group)
        for r in group:
            assert isinstance(r['start_tick'],int) and isinstance(r['end_tick'],int)
            assert last<r['start_tick']<r['end_tick']<2**64-1
            assert math.isfinite(r['attributed_gpu_ms']) and r['attributed_gpu_ms']>0
            assert abs(parents[key]['gpu_ms']*(r['end_tick']-r['start_tick'])/ticks-r['attributed_gpu_ms'])<1e-6
            components[r['component']]+=r['attributed_gpu_ms']; last=r['end_tick']
        assert abs(sum(r['attributed_gpu_ms'] for r in group)-parents[key]['gpu_ms'])<1e-6
        tails[key]={r['component']:r['attributed_gpu_ms'] for r in group}
        coverage.append(ticks/(group[-1]['end_tick']-group[0]['start_tick']))
    assert all(0<v<=1 for v in coverage)
    prefix=[r for r in p['rows'] if r['stage']=='initial_bootstrap']
    assert len(prefix)==1 and prefix[0]['first_layer']==0 and prefix[0]['last_layer']==2
    layers=[r for r in p['rows'] if r['stage']=='prefix_layer']
    assert len(layers)==40 and [r['first_layer'] for r in layers]==list(range(3,43))
    profiles[role]=dict(stages=dict(stages),components=dict(components),tails=tails,
        native_tails=len(groups),component_intervals=len(t['rows']),profile_sha256=sha(w/filename),
        coverage=dict(minimum=min(coverage),median=statistics.median(coverage),maximum=max(coverage)))

controls=plan['diagnostic_assessment']['unchanged_tail_controls']
before,after=profiles['baseline'],profiles['candidate']
control_before=sum(before['components'][n] for n in controls)
control_after=sum(after['components'][n] for n in controls)
aggregate=(after['components']['routed_gate_up']/control_after)/(before['components']['routed_gate_up']/control_before)
matched=[]
for k in sorted(expected):
    a,b=before['tails'][k],after['tails'][k]
    c=sum(b[n] for n in controls)/sum(a[n] for n in controls)
    raw=b['routed_gate_up']/a['routed_gate_up']
    matched.append(dict(chunk_start=k[0],layer=k[1],baseline_gate_ms=a['routed_gate_up'],
        candidate_gate_ms=b['routed_gate_up'],gate_ratio=raw,control_ratio=c,normalized_gate_ratio=raw/c))
ratios=[r['normalized_gate_ratio'] for r in matched]
median=statistics.median(ratios)
result=dict(passed=True,checks=checks,profiles={role:{k:v for k,v in p.items() if k!='tails'} for role,p in profiles.items()},
    diagnostic_comparison=dict(aggregate_normalized_gate_ratio=aggregate,
        matched_normalized_gate_ratio=dict(median=median,mad=statistics.median(abs(v-median) for v in ratios),
            minimum=min(ratios),maximum=max(ratios),improved=sum(v<1 for v in ratios),count=len(ratios)),
        component_total_ratios={n:after['components'][n]/before['components'][n] for n in names},
        matched_tails=matched),
    later_engine_timing_justified=aggregate<1 and median<1,engine_speedup_claim=False,
    all_new_frontiers_claim=False,complete_32k_c0=True)
if (w/'audit.json').exists():
    assert read('audit.json') == result, 'Stored audit differs from recomputation'
else:
    (w/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,full_fp32_words_checked=3*16677120,
    profile_packets_checked=2*287,aggregate_normalized_gate_ratio=aggregate,
    median_normalized_gate_ratio=median,improved_matched_tails=sum(v<1 for v in ratios),
    later_engine_timing_justified=result['later_engine_timing_justified']),indent=2))
