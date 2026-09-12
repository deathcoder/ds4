# Scratch RSS diagnostic — milestone closed with a protocol gap

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
| A | False | none | 16,875,520 | 2,501,739,096 |
| B | False | all | 2,510,356,480 | 2,503,330,216 |
| C | True | none | 16,531,456 | 2,501,460,592 |
| D | True | all | 2,509,783,040 | 2,502,674,736 |
| E | True | except_four_copy_destinations | 1,973,108,736 | 2,502,658,040 |

Paired differences, in bytes (positive means clearing costs more):

| Block | Order | D minus C RSS | D minus E RSS saved | D minus E footprint saved |
|---|---|---:|---:|---:|
| 1 | ABCDE | 2,493,284,352 | 536,805,376 | 147,768 |
| 2 | BCDEA | 2,492,841,984 | 536,969,216 | 278,816 |
| 3 | CDEAB | 2,493,448,192 | 537,001,984 | 344,376 |
| 4 | DEABC | 2,492,940,288 | 536,231,936 | -392,832 |
| 5 | EABCD | 2,493,366,272 | 536,788,992 | 131,384 |

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
