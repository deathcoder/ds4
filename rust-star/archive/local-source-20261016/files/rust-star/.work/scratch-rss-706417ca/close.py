from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(w/'plan.json');a=read(w/'audit.json');s=read(w/'source-proof.json')
assert all(sha(r/n)==h for n,h in p['runtime_state']['sources'].items())
assert sha(r/'rust-star/.work/runtime-target/release/rust-star')==p['runtime_state']['default_executable_sha256']
assert all(v['final_live_bytes']==0 and v['max_live_bytes']==2493173233 for v in a['raw'].values())
assert all(not any(v['fresh_previous_at_entry']) for v in a['raw'].values() if v['mode'] in 'AB')
processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True)
active=[x for x in processes.splitlines() if Path(x.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','diagnostic-rust-star','baseline-rust-star','ds4','ds4-bench','ds4-server','probe'}]
assert not active
state=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=p['runtime_state']['sources'],default_executable_sha256=p['runtime_state']['default_executable_sha256'],runtime_changed=False,model_runs=0,active_gpu_processes=active)
(w/'final-runtime-state.json').write_text(json.dumps(state,indent=2)+'\n')
decision=dict(closed=True,measurement_criteria_pass=all(a['gates'].values()),full_copy_source_proof_pass=True,guarded_copy_proof_pass=True,prospective_protocol_pass=False,select_for_integration=False,engine_acceptance=False,runtime_changed=False,model_runs=0,protocol_deviation='The frozen plan required an after-copy mutation in the guarded proof before residency measurements. probe.m verifies full copies into zeroed and poisoned destinations, but does not run an in-place mutation after copying. The pre-run checker and audit.py did not enforce this subcondition. Discovered during final source review after all25 measurements; no GPU rerun or threshold change. Preserve the original optimistic audit and use this decision as the milestone verdict.',next_milestone='Close the guarded in-place mutation test gap first. If it passes, evaluate a narrow four-destination clear-elision candidate with other101 initializations and prior checked completion unchanged, fresh full C0 and pool evidence, and unchanged engine RSS/footprint/performance gates. Do not start automatically.')
(w/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
proof='''# Full-copy destination proof

This proof applies only to the native 1024-row branch in the frozen rejected
scratch-pool source, SHA-256 `'''+s['candidate_sha256']+'''`.
The accepted runtime has not been edited. See `source-proof.json` for every
object reference and `source-proof.py` for the reproducible lexical checks.

| Destination | Source | Full-copy source line | Bytes |
|---|---|---:|---:|
'''
for b in s['buffers']:proof+=f"| `{b['destination']}` | `{b['source']}` | {b['copy_line']} | {b['bytes']:,} |\n"
proof+='''
Each allocation and copy uses `q_bytes = rows*q_dim*sizeof(float)` with rows
1024 and q_dim 32768: 134,217,728 bytes. Every copy begins at source and
destination offset zero and covers the entire logical destination. Total:
536,870,912 bytes. Allocation success checks are not content reads.

All destination references were reviewed. Before the copy there is no content
read, kernel binding, copy-source use or alias assignment. The producer compute
encoder ends before the blit; the blit encoder ends before the first kernel
binding that reads or mutates the destination. Layer-1 copies are inside
complete_layer1, which the native 1K branch requires. Encoder failures return
without submitting a consumer. Preserve these paths and the existing checked
prior-command completion before scratch reuse. Keep distinct source/destination
identities, exact lengths and final/error owner cleanup.

After a complete copy, every destination byte equals the corresponding source
byte independently of the destination's previous contents. Therefore removing
only these destination clears cannot change the bytes supplied to subsequent
unchanged consumers. This reasoning does not require a new claim that the
source producers fully write their buffers: preserve those source initializers
and all other 101 initializations. Do not generalize to other sizes or paths.
The source proof is bounded review plus reproducible reference checks, not a
formal proof of every Metal or engine behavior.

The GPU check verifies two opposing poisons against a zeroed destination,
all copied bytes and prefix/suffix guards at sizes 4, 4096 and 134217728,
with destination reuse across two seeds. All 24 cases, 2,147,549,248 payload
bytes and 24,576 guard bytes pass. All 12 shortened-copy controls are detected.
However, the planned in-place mutation after copying is absent. Re-poisoning
before the next copy does not satisfy that condition. The full prospective
protocol is incomplete; `decision.json` overrides the original audit's positive
selection. The missing test belongs before any next integration. No model/C0
claim is made by this synthetic check.
'''
(w/'FIRST_WRITE_PROOF.md').write_text(proof)
rows=''
for mode in 'ABCDE':
 v=a['summary'][mode];rows+=f"| {mode} | {p['modes'][mode]['reuse']} | {p['modes'][mode]['cpu_clear']} | {v['peak_rss']['median']:,} | {v['peak_footprint']['median']:,} |\n"
paired=''
for x in a['pairs']:paired+=f"| {x['block']} | {x['order']} | {x['pool_clear_effect']:,} | {x['selected_saving']:,} | {x['selected_footprint_saving']:,} |\n"
readme='''# Scratch RSS diagnostic — milestone closed with a protocol gap

The 25-process synthetic study supports CPU clearing as an RSS residency cost:
all-buffer clearing adds a paired median 2,493,284,352 bytes, and omitting four
complete-copy destination clears saves 536,805,376 bytes (511.94 MiB). Source
review proves the four destination ranges are fully overwritten before reads.
All measurements and implemented GPU checks pass, but the frozen guarded proof
omitted its planned mutation after the copy. **The full prospective protocol
does not pass; no candidate is selected for integration by this milestone.**
`decision.json` is authoritative; the original optimistic `audit.json` is
preserved unchanged and its inadequate precondition check is disclosed.

Accepted source and the default executable are unchanged. No model inference,
engine timing, remote dispatch, publication or 256K work ran in this milestone.
This is a completed diagnostic outcome, not a retained optimization.

## Design and results

Plan and executable/scripts were frozen before 25 fresh processes in balanced
orders ABCDE, BCDEA, CDEAB, DEABC, EABCD. No exclusions or repeated cells.
Each process uses the same 105 actual scratch lengths, totaling 2,493,173,233
bytes, across four tiles. GPU pattern writes fill the other 101 buffers; full
blits populate the four destinations; GPU verification reads every logical
byte. Only a tiny error counter is read on CPU. CPU clears are the experimental
difference. Modes A/C are synthetic controls, not proposed runtime policies.

Median process peaks from `/usr/bin/time -l`, in bytes:

| Mode | Reuse | CPU clear | RSS | Memory footprint |
|---|---|---|---:|---:|
'''+rows+'''
Paired differences, in bytes (positive means clearing costs more):

| Block | Order | D minus C RSS | D minus E RSS saved | D minus E footprint saved |
|---|---|---:|---:|---:|
'''+paired+'''
The frozen measurement thresholds all pass: D-C is positive in every block
and median >=1,246,586,616; D-E is >=268,435,456 in every block and median
>=402,653,184. Reuse without clearing changes RSS by a median only 49,152 bytes.
D-E footprint saving is a median 147,768 bytes and changes sign in one block:
there is no material allocation/footprint saving claim. Every mode still has
2,493,612,032 peak device-allocated bytes and one logical scratch set. All final
weak samples are zero; fresh modes show no previous generation at tile entry.
Fresh+clear B has two higher transient peaks, retained in the report. Weak
object release does not imply immediate OS residency release; final RSS can
remain high. These observations apply to this host and synthetic workload.

The prior engine candidate failed RSS +2.415% against +2%, despite prefill
-10.243% and footprint +11.663% within +15%. A roughly 512 MiB RSS opportunity
is larger than its roughly 374 MB budget miss, but subtracting these synthetic
bytes from whole-engine peaks is not a validated prediction. Engine allocation,
model pages and timing can change the result. No speed, DwarfStar parity or
statistical significance claim is made here.

## Evidence and deviations

- 25 fresh processes / 100 tiles / 249,317,323,300 verified payload bytes.
- 550 residency snapshots and 100 sets of 105 distinct buffer identities;
  reused identities remain stable throughout each process.
- Guarded proof: 24 cases, 2,147,549,248 payload bytes, 24,576 guard bytes,
  12 detected shortened-copy negative controls. See `FIRST_WRITE_PROOF.md`.
- One host compile failure before GPU execution: the assert macro split an
  Objective-C array literal at commas. Parenthesizing the expression fixed it;
  original source/log and correction are preserved. No measurement retry.
- Final review identified the missing after-copy mutation. The pre-run checker
  checked counts and result status, while audit.py used source status and exit
  code; neither enforced this explicit plan requirement. No frozen source,
  result or threshold was edited. No GPU work followed discovery.
- All 42 accepted source hashes and default executable identity were checked
  before each cell and at closure; `final-runtime-state.json` records closure.
  Prior model/oracle acceptance is historical, not refreshed here.

## Portable verification

Run `python3 verify-report.py` in this directory (Python 3.9+ standard library).
The verifier checks stored/decompressed artifact hashes and frozen identities,
recomputes the source proof and all 25-process calculations, checks process
order and final state, reproduces the protocol-gap verdict, and rejects a
corrupted clear-byte count and a changed pooled buffer identity. No compiler,
GPU, model or original workspace is needed. It verifies captured evidence;
it does not rerun measurements or prove their physical provenance. Compressed
files retain their original names and hashes in `manifest.json`. The executable
is represented by its size and SHA-256, not distributed. `run.py` is historical
execution evidence and must not be invoked as the portable verifier.

## Next milestone — not started

First add the missing guarded copy-then-in-place-mutation proof under a new
prospective plan. If it passes, evaluate only the four proven destination
clears in the bounded scratch-pool candidate, retaining all other initializers,
checked completion and ownership bounds. Require fresh full C0 and pool evidence
before the unchanged engine gates: RSS median <=1.02 and footprint <=1.15,
alongside every existing prefill/order/decode criterion. Stop after this update;
no integration, additional inference or publication has started.
'''
(w/'README.md').write_text(readme)
print(json.dumps(decision,indent=2))
