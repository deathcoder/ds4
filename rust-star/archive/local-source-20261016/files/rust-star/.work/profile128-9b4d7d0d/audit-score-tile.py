from pathlib import Path
import json,statistics,math,hashlib
w=Path(__file__).resolve().parent
rows=[json.loads(line) for line in (w/'score-tile-probe.jsonl').read_text().splitlines()]
assert rows[-1]=={'passed':True} and not (w/'score-tile-probe.log').read_text()
shapes=[r for r in rows if 'bitwise_match' in r];assert [r['shape'] for r in shapes]==list(range(7))
for r in shapes:assert r['bitwise_match'] and r['checked_words']==r['tokens']*r['compressed_rows']
summary={}
for shape in (5,6):
 pairs=[r for r in rows if r.get('shape')==shape and 'pair' in r]
 assert [r['pair'] for r in pairs]==list(range(1,8))
 for r in pairs:
  assert r['order']==('AB' if r['pair']%2 else 'BA')
  assert all(math.isfinite(r[k]) and r[k]>0 for k in ('baseline_gpu_ms','candidate_gpu_ms','ratio'))
  assert abs(r['candidate_gpu_ms']/r['baseline_gpu_ms']-r['ratio'])<1e-8
 ratios=[r['candidate_gpu_ms']/r['baseline_gpu_ms'] for r in pairs]
 assert statistics.median(ratios)<1
 summary[str(shape)]=dict(pairs=7,baseline_median_ms=statistics.median(r['baseline_gpu_ms'] for r in pairs),candidate_median_ms=statistics.median(r['candidate_gpu_ms'] for r in pairs),paired_ratio_median=statistics.median(ratios),paired_ratio_range=[min(ratios),max(ratios)],improved_pairs=sum(v<1 for v in ratios))
source=(w/'score-tile-baseline.metal').read_text();baseline=source[source.index('kernel void kernel_dsv4_indexer_scores_tiled('):]
candidate=(w/'score-tile-candidate.metal').read_text().split('\n',2)[2]
assert candidate==baseline.replace('kernel_dsv4_indexer_scores_tiled','rust_star_indexer_scores_tiled64').replace('constexpr uint TN = 32;','constexpr uint TN = 64;').replace('i += 128','i += 256').replace('// [32][128]','// [64][128]').replace('// [8][32]','// [8][64]')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x=dict(passed=True,full_score_bitwise_shapes=7,total_checked_words=sum(r['checked_words'] for r in shapes),isolated_pairs=summary,scope='Synthetic isolated score kernel; not full-model C0 or engine throughput',sources={n:sha(w/n) for n in ('score-tile-probe.m','score-tile-baseline.metal','score-tile-candidate.metal','score-tile-combined.metal','score-tile-probe.jsonl','score-tile-probe.log','score-tile-hypothesis.json')})
(w/'score-tile-audit.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
