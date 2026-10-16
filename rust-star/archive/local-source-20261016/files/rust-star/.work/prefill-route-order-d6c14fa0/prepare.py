"""Freeze a synthetic experiment without changing production sources."""
from pathlib import Path
from collections import defaultdict
import datetime
import difflib
import hashlib
import json
import subprocess

w = Path(__file__).resolve().parent
root = w.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = root / 'rust-star/reports/128k-score64-b6b0f125'
artifacts = json.loads((r / 'artifact-hashes.json').read_text())['artifacts']
for name in ['checks/components.json', 'checks/component-audit.json']:
    assert sha(r / name) == artifacts[name]['sha256']
p = json.loads((r / 'checks/components.json').read_text())
stages, tails = defaultdict(float), defaultdict(float)
for row in p['rows']:
    stages[row['stage']] += row['gpu_ms']
for row in p['tail_components']['rows']:
    tails[row['component']] += row['attributed_gpu_ms']
selection = dict(source=str((r / 'checks/components.json').relative_to(root)),
                 source_sha256=sha(r / 'checks/components.json'),
                 stage_gpu_ms=dict(sorted(stages.items(), key=lambda x: -x[1])),
                 tail_attributed_gpu_ms=dict(sorted(tails.items(), key=lambda x: -x[1])),
                 profile_perturbed=True, not_engine_throughput=True)
(w / 'profile-selection.json').write_text(json.dumps(selection, indent=2) + '\n')
files = ['rust-star/runtime/src/moe_output_prefix.metal', 'metal/flash_attn.metal',
         'metal/dense.metal', 'metal/glu.metal', 'metal/moe.metal', 'metal/dsv4_hc.metal']
source = '\n'.join((root / f).read_text() for f in files)
moe = (root / 'metal/moe.metal').read_text()
accepted_moe = subprocess.check_output(['git', 'show', 'b6b0f125:metal/moe.metal'], cwd=root, text=True)
assert moe == accepted_moe
start = moe.index('kernel void kernel_mul_mm_id_pair_swiglu_f16_impl(')
end = moe.index('\ntypedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl', start)
body = moe[start:end]
candidate = body
for old, new in [('const uint32_t work_index = tgpig.x;', 'const uint32_t work_index = tgpig.y;'),
                 ('const int r0 = tgpig.y*NR0;', 'const int r0 = tgpig.x*NR0;')]:
    assert candidate.count(old) == 1
    candidate = candidate.replace(old, new, 1)
assert source.count(body) == 1
for name, text in [('baseline.metal', source), ('candidate.metal', source.replace(body, candidate, 1))]:
    assert not (w / name).exists(), 'Do not overwrite a frozen experiment'
    (w / name).write_text(text)
(w / 'kernel-change.patch').write_text(''.join(difflib.unified_diff(
    body.splitlines(True), candidate.splitlines(True), fromfile='baseline-fused-gate-up',
    tofile='transposed-grid-fused-gate-up')))
plan = dict(status='predeclared', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            candidate_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            accepted_native_comparator='b6b0f125', native_moe_source_matches_accepted=True,
            oracle_source='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4',
            hypothesis='Transpose only grid axes of fused IQ2_XXS routed gate/up to test locality; retain 64x32 MMA tile, arithmetic, activation and stores.',
            scope='Isolated synthetic kernel experiment; not full-model C0 or engine throughput.',
            source_sha256={f: sha(root / f) for f in files},
            generated_sha256={n: sha(w / n) for n in ['baseline.metal', 'candidate.metal', 'kernel-change.patch']},
            dimensions=dict(experts=256, top_k=6, input_width=4096, output_width=2048,
                            threadgroup_threads=128, threadgroup_bytes=16384),
            controls=[1, 31, 32, 33, 127],
            timed_cases=[dict(rows=2048, routing='balanced'), dict(rows=4096, routing='balanced'),
                         dict(rows=4096, routing='skewed')],
            timing=dict(warmups_per_variant=2, alternating_pairs=7,
                        metric='Completed command GPU milliseconds',
                        comparison='All FP16 output words against baseline on every run; finite outputs and guard bytes required.'),
            advance_criteria=dict(all_outputs_bit_exact=True, both_4096_case_median_paired_ratio_below=1.0,
                                  minimum_faster_pairs_per_4096_case=5, no_engine_acceptance=True),
            runtime_mutation=False, no_full_model_jobs=True)
(w / 'hypothesis.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(selection, indent=2))
print('Hypothesis frozen; shader change limited to two index expressions.')
