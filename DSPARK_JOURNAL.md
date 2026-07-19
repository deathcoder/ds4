# DSpark Development Journal

This file is durable working memory for the DSpark-on-ds4 effort. Read it when
resuming after context compaction, switching agents, or feeling unsure why a
particular DSpark change exists.

## Current Phase

Branch: `codex/dspark-observability-0`

Phase 1.38 is complete. After promoting exact shared-expert Q8 proposal rows,
the fresh 32-task cumulative HumanEval reassessment measured exact DSpark at a
`0.8814x` geometric paired ratio versus ordinary baseline, with every output
byte-exact. This is effectively flat against Phase 1.27's controlled
`0.8826x` result despite the isolated shared-Q8 gate's `1.0196x` geometric
gain: baseline and runtime absolute throughput moved between sessions, and the
remaining end-to-end gap is still `11.9%`. Current/historical movement was
`1.0208x` geometrically, with `22/32` tasks improved and a `0.9478x` minimum,
so the predeclared movement gate failed. Keep the unanimous same-session
shared-Q8 promotion, but make no cumulative speedup claim from this run.

Phase 1.39 is complete. The fresh stats-only audit accounts for the frozen
`0.8814x` runtime as `38.298 ms/emitted` target verification,
`7.881 ms/emitted` sidecar, and `0.633 ms/emitted` residual. Target verification
remains dominant at `81.8%` of runtime, and the calibrated parity target is a
`14.6%` reduction in target time. Width 5 still consumes `55.4%` of target
time, at `35.561 ms/position` versus `40.942 ms/position` for width 1. The
proposal schedule has `760` attempts, `684` full and `76` partial outcomes,
with zero fallbacks and byte-exact output throughout.

Phase 1.40 is complete. At stable verifier width 5, the sampled layer totals
are `0.578 ms/row` attention preparation, `1.167 ms/row` serial attention
tail, and `1.061 ms/row` exact FFN. Shared-Q8 promotion moved sampled FFN from
the earlier `1.165 ms/row` to `1.061 ms/row`, while the serial tail remained
roughly flat and is now the largest stage. FFN still has the weakest
width-2-to-width-5 amortization, but widths 2 and 3 each have only one noisy
observation; the twenty width-5 evaluations are the stable optimization guide.

Phase 1.41 is complete. The refreshed serial-tail profile measured a stable
width-5 median of `2.434 ms/row`. Compressor/indexer is now the largest
component at `0.403 ms/row` (`20.1%`), followed by attention at `0.362`,
projection B plus HC at `0.343`, projection A at `0.331`, KV/cache update at
`0.295`, and inverse RoPE at `0.270 ms/row`. The tail is broadly distributed:
whole-suffix batching remains retired, attention already uses fused gather,
and the projection suffix already uses promoted NR4 after NR8 regressed.

Phase 1.42 is complete. The default-off exact compressor projection-prebatch
candidate measured a focused `1.0212x` median paired gain, with all three pairs
positive (`1.0309x`, `1.0120x`, and `1.0212x`) and every output byte-exact.
This clears the decision boundary for broader confirmation; it is not yet a
promoted default.

Phase 1.43 is complete but formally failed. Compressor projection prebatching
measured a `1.0096x` geometric gain and won `31/32` tasks; low-acceptance tasks
gained `1.0094x`. The sole failure was `humaneval_095` at `0.9138x`, below the
predeclared `0.95x` floor. All outputs were byte-exact and every other task was
positive, so do not promote yet and do not discard the broad direction from
this single noisy-looking observation.

Phase 1.44 is complete and passed. `humaneval_095` won all `6/6` repeated
pairs, with a `1.0170x` median paired ratio, `1.0322x` geometric ratio, and a
`1.0066x` minimum. The isolated broad-gate regression was not reproduced.
Exact compressor projection prebatching is now promoted to the default exact
verifier path; `DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH=0` or `off` retains the
legacy serial projection route. The branch is again at a safe checkpoint.

Phase 1.45 is complete. The fresh cumulative 32-task reassessment measured
exact DSpark at a `0.8993x` geometric paired ratio versus ordinary baseline,
leaving a `10.1%` gap to parity. Against the immediately preceding `0.8814x`
cumulative checkpoint, this is `1.0203x` geometric movement with `28/32` tasks
improving and a `0.9513x` minimum movement. Treat the isolated prebatch gate as
the causal evidence and this cross-session cumulative change as supportive.
The runtime is closer, but still below the `0.95x` near-parity boundary.

Phase 1.46 is complete. The refreshed audit accounts for the `0.8993x` runtime
as `36.356 ms/emitted` target verification, `7.596 ms/emitted` sidecar, and
`0.742 ms/emitted` residual. Target verification remains dominant at `81.3%`;
parity requires a calibrated `12.4%` target-time reduction. Width 5 consumes
`54.7%` of target time at `33.359 ms/position`, while width 1 costs
`40.245 ms/position` and consumes `18.8%` of target time.

Phase 1.47 is complete. At stable width 5, the sampled layer totals are
`0.656 ms/row` attention preparation, `1.083 ms/row` serial attention tail,
and `1.070 ms/row` exact FFN. The tail and FFN are effectively tied; FFN has
the slightly weaker width-2-to-width-5 amortization (`0.647x` versus `0.628x`).
The synchronized total is nearly unchanged from Phase 1.40, so use stage rank,
not cross-session absolute profile deltas, to choose the next diagnostic.

Phase 1.48 is complete. At stable width 5, exact FFN accounts for
`2.001 ms/row` across synchronized substages versus a `2.067 ms/row` outer
control. Routed MoE is still dominant at `0.806 ms/row` (`40.3%`) and has the
weakest width-2-to-width-5 amortization at `0.518x`. Shared gate/up and down
are now `0.233` and `0.212 ms/row`; all other components are individually
below `0.230 ms/row`.

Phase 1.49 is prepared. The routed-MoE profiler is repinned to the clean Phase
1.48 FFN artifact at commit `fafcdda` and converted from obsolete pre-hybrid
one-row records to the promoted batch encoder's `gate_up`,
`activation_weight`, `down`, and `sum` stage records. It requires the
`hybrid_exact_down_*` path, F32 mids, exact verifier shapes, frozen width
counts, and byte-identical output before reporting per-row normalized costs.

Phase 1.27 is complete. The cumulative 32-task HumanEval reassessment measured
current exact DSpark at a `0.8826x` geometric paired ratio versus ordinary
baseline, with all outputs byte-exact. This improves the historical clean
`0.8634x` artifact by `1.0222x` geometrically across tasks, with `30/32` tasks
improving, but fails the predeclared `1.05x` movement gate and remains below
near parity. The fresh controlled end-to-end gap is `11.7%`; all 32 tasks are
still slower than baseline. Cross-session absolute t/s moved for both arms, so
the prior fused-gather ablation ratio must not be multiplied into the old
end-to-end result. Next prepare a fresh post-promotion stats-only cost audit to
recalibrate target-verifier, sidecar, and residual costs against this artifact
before choosing another verifier optimization.

Phase 1.28 is complete. The post-promotion audit accounts for the frozen
`0.8826x` runtime as `39.010 ms/emitted` target verification,
`6.831 ms/emitted` sidecar, and `0.210 ms/emitted` residual. Target verification
is still `84.7%` of runtime, and parity requires a calibrated `13.9%` target-time
reduction. The proposal schedule exactly matches the older audit, with no
fallbacks or output drift. Width 5 consumes `55.5%` of target time but costs
only `11.3%` less per position than width 1, confirming that shared exact
multi-row execution remains the primary optimization surface. Next refresh the
width-stratified layer attribution after fused-gather promotion before choosing
another verifier candidate.

Phase 1.29 is complete. The post-promotion width-layer profile identifies exact
FFN as both the largest sampled width-5 stage (`40.8%`) and the
weakest-amortizing stage (`0.707x` width 5 versus width 2). Serial attention is
now `39.2%`, and attention preparation is `19.9%`. Relative to the structurally
identical pre-promotion profile, synchronized width-5 tail time moved by
`0.835x` while FFN stayed at `0.992x`; treat those cross-session ratios as
directional, but the new ranking is clear. Next profile the existing exact FFN
sub-stages by verifier width before choosing a shared multi-row FFN candidate.

Phase 1.30 is complete. The width-stratified exact-FFN profile identifies
routed MoE as both the largest width-5 sub-stage (`40.9%`) and the weakest to
amortize (`0.747x` width 5 versus width 2). Every other FFN sub-stage scales to
`0.431x` or better. The exact implementation intentionally invokes the
one-token routed-MoE arithmetic once per proposal row; those calls already
share the outer Metal command batch, so merely reusing a command buffer is not
a new optimization. Next attribute the existing one-token routed-MoE stages
inside exact verifier batches before choosing a kernel or encoding candidate.

Phase 1.31 is complete. The exact one-row routed-MoE profile assigns `49.5%`
of stable width-5 time to gate/up and `43.7%` to down; activation-weight and
sum are only `3.4%` each. Per-row cost is effectively flat from verifier width
2 through 5, confirming that current exact execution does not amortize either
matrix stage. Earlier observer work proved the generic batched routed weighted
mid exact through width 5 and localized width-5 drift to down/sum. Next build
an opt-in hybrid correctness candidate that batches gate/up and activation but
preserves the exact one-row direct six-expert down/sum arithmetic.

Phase 1.32 is complete. The default-off exact routed-MoE hybrid improved the
focused uninstrumented microbenchmark by `1.0158x` ratio of medians and
`1.0149x` median paired, with all three pairs faster and byte-identical output.
This is directionally consistent with attacking only the routed gate/up share,
but is too small to promote from one short prompt. Next run a frozen 32-task
threshold-0.75 HumanEval default-versus-hybrid gate before changing defaults.

Phase 1.33 is complete and the routed-MoE hybrid passed promotion. On the
frozen 32-task HumanEval workload it achieved `1.0164x` geometric paired
movement, won `32/32` tasks, had a `1.0073x` minimum task ratio, and improved
the low-acceptance subgroup by `1.0119x`. Every output remained byte-exact.
Next promote the hybrid as the exact Metal default with an explicit legacy
opt-out, then rerun the cumulative baseline-versus-DSpark assessment.

Phase 1.34 is complete. The routed-MoE hybrid is now the exact Metal default;
`DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID=0` (or `off`) selects the prior fully
row-wise implementation for regression and attribution. Both default and
legacy-opt-out runtime correctness matrices pass all five scenarios, all
`148` DSpark model-free tests pass, and the cumulative HumanEval harness dry
run confirmed that its runtime arm inherits the promoted default without a
candidate flag. The fresh cumulative run measured current exact DSpark at
`0.8782x` baseline geometrically, a `12.2%` gap, with no task faster than
baseline. Relative to the immediately prior `0.8826x` cumulative artifact the
result is effectively flat within cross-session noise (`0.9950x` geometric,
`1.0033x` median task movement), dominated by two roughly `0.90x` outliers.
Keep the strongly confirmed same-session promotion, but claim no cumulative
speedup from this run. This clean commit and recorded benchmark are a safe
checkpoint for reviewing community progress before choosing another verifier
optimization.

Phase 1.05 is complete. The frozen 32-task HumanEval gate measured a `0.8081x`
median paired DSpark/baseline ratio and `0.7910x` geometric mean on the
promoted scheduler plus exact prefix-checkpoint runtime. This improves over the
prior scheduled artifact's `0.7498x` median: task-level paired-ratio movement
was `1.0651x` median and `1.0647x` geometric, with 26 of 32 tasks improving.
All outputs remained byte-identical, but all 32 DSpark tasks were still slower
than baseline. Phase 1.06 should add stats-only checkpoint attempt, success,
fallback, and avoided-replay-row counters, then attribute a small frozen set;
do not repeat the full 32-task throughput run yet.

Phase 1.06 is complete. On four frozen HumanEval tasks, exact prefix
checkpoints succeeded on every eligible partial batch (`29/29`) with zero
fallbacks and avoided `72` target replay rows. Structural target-position
reductions ranged from `5.6%` to `18.2%`. Checkpoint coverage and reliability
are complete; do not optimize this mechanism further. Phase 1.07 should
instrument scheduler/verifier width economics under the stats gate: selected
width, committed progress, target time by width, and sidecar rounds. This is
needed before attempting a cost-aware scheduler for low-acceptance workloads.

Phase 1.07 instrumentation is prepared and awaiting the frozen four-task
user-run. Runtime stats now record scheduler-selected width, committed progress
and sidecar time by selected width, plus target evaluation count, positions,
and synchronized time by actual verifier width. Sidecar timing is attached to
the draft block that reaches multi-commit; work consumed by another path or
left after the final emitted token is reported separately. The Metal
correctness matrix passed and all five retained records reconciled. Do not tune
a cost-aware scheduler until the four-task width report is collected.

Phase 1.07 is complete. The four-task width report shows that sidecar cost is
effectively fixed at about `18 ms` per proposal across selected widths, while
exact target verification remains close to linear: `44.4 ms/position` at width
one and `39.9 ms/position` at width five. Short selected prefixes are therefore
correctly expensive, but changing the threshold cannot remove the sidecar work
because the full draft block is already computed. Do not start a cost-aware
scheduler implementation from this result. The next candidate is an
env-gated port/comparison of the community PR #502 specialized Metal drafter,
which another branch measured at about `7.6 ms/cycle`. Preserve our exact
verifier and byte-identical output contract during that experiment.

Phase 1.01 is complete. DSpark confidence-prefix scheduling now defaults to
`0.455` when `DS4_DSPARK_CONFIDENCE_THRESHOLD` is absent. An explicit value
still overrides the default, and explicit `0` preserves fixed K=5. The frozen
acceptance, trace, scheduler-ablation, generalization, and historical fixed-K
HumanEval paths now set `0` rather than relying on an absent environment value.
Normal runtime paths inherit the promoted default. Both the promoted-default
and explicit-zero runtime correctness matrices passed byte-for-byte across all
five scenarios; model-free tests, the normal build, DSpark validation, and
shape binding also passed. No throughput benchmark was run automatically.

Phase 1.00 is complete and passed every predeclared promotion criterion. On the
12-task non-code gate, threshold `0.455` improved over fixed K=5 on `5/6` math
and `6/6` chat tasks. Scheduled/fixed medians were `1.0831x` math, `1.3928x`
chat, and `1.2122x` overall; overall geometric mean was `1.2222x`, and the
minimum task ratio was a neutral `0.9950x`. Every three-mode output matched
byte-for-byte. This permits promoting `0.455` to the DSpark runtime default,
while preserving `DS4_DSPARK_CONFIDENCE_THRESHOLD=0` as explicit fixed K=5.
All benchmark/control paths labeled fixed K=5 must set `0` after promotion.

Phase 0.99 is complete. The frozen 32-task HumanEval scheduled run at threshold
`0.455` raised the median paired DSpark/baseline ratio from Phase 0.94's
historical fixed-K `0.6840x` to `0.7498x`; geometric mean rose from `0.6794x`
to `0.7430x`. The historical per-task ratio improved on 30 of 32 tasks, while
all 32 scheduled tasks remained slower than ordinary target decoding. This
validates confidence scheduling as a material DSpark optimization on code, but
does not yet justify a universal raw-confidence default from a code-only study.
Keep `0.455` opt-in until a predeclared non-code generalization gate passes; do
not tune the threshold further.

Phase 0.98 is complete. Its clean, uninstrumented runtime gate found a stable
confidence-scheduler win on both predeclared HumanEval tasks. Threshold `0.38`
won all six pairs with aggregate median `1.0695x`; threshold `0.455` won all six
with aggregate median `1.1582x`, including `1.1988x` on low-acceptance
`humaneval_152` and `1.1150x` on high-acceptance `humaneval_079`. Select `0.455`
for the next gate without further tuning, but keep it opt-in until a frozen
32-task baseline-versus-scheduled throughput run confirms end-to-end behavior.

Phase 0.97 is complete. Its 32-task HumanEval trace reproduced the prior output
and acceptance audit exactly. Raw confidence supports a stable DeepSpec-style
prefix policy: leave-one-task-out thresholds differed by at most `0.005` from
the in-sample choices. At the conservative `~0.38` policy, local progress was
`0.990x`, the target-position proxy was `0.936x`, and target-eval/sidecar-round
amplification was `1.010x`; the balanced `~0.455` policy measured `0.976x`,
`0.899x`, and `1.025x`. This is promising enough for a guarded runtime
ablation, not promotion. No confidence scheduler changes runtime behavior yet.
The next phase should implement the official prefix rule behind an opt-in
threshold and compare fixed K=5, `0.38`, and `0.455` on predeclared low- and
high-acceptance HumanEval tasks before another full 32-task run.

Phase 0.96 is complete but not promoted. The exact Metal verifier has an opt-in
in-place attention/inverse-RoPE fusion that preserves the interleaved
autoregressive row schedule and removes the separate inverse-RoPE dispatch.
Generated output is byte-identical across the correctness matrix, but the fused
shader is not hidden-state bit-exact. Its user-run throughput gate measured only
a `1.0034x` median paired ratio with one of three pairs slightly negative. Keep
the candidate opt-in as research evidence; its marginal, interference-sized
gain does not justify making bounded arithmetic drift the default.

Phase 0.95 is complete. The exact Metal verifier now defaults to exact FFN
batching, row-independent attention preparation, and the promoted 2-5-row Q8
projection microbatch while retaining cache-mutating attention work in serial
autoregressive row order. Phase 0.94 showed that the Q8 promotion improved
absolute DSpark throughput on all 32 HumanEval tasks and raised the median
paired ratio from `0.6456x` to `0.6840x`, but all tasks remain slower than
baseline. Phase 0.95 showed that verifier cost per proposal row is effectively
task-invariant, while low acceptance amplifies the number of proposal rows paid
per emitted token. The next phase should optimize a narrow part of the retained
serial attention tail in place, preserving exact interleaved row order and not
reviving the rejected whole-suffix batching design.

Historical Phase 0.38 was deliberately diagnostic: `--dspark FILE` validates an official
DSpark drafter GGUF, binds every tensor needed by a future runtime path, checks
the expected DeepSeek V4 Flash DSpark shapes, and exposes a diagnostic
`--dspark-probe` bridge for target-layer hidden-state capture plus
`main_proj/main_norm`, sidecar block execution, Markov-biased logits, and
confidence scores. The per-row Markov/confidence/logit combiner now lives
behind a private `dspark_draft_step_cpu` scratch/result boundary so later
runtime work can call one draft row without inheriting probe logging. A lazy
private `dspark_session_state` now owns the sidecar `main_x` window, per-stage
KV rows, draft block buffers, confidence buffer, and draft-step scratch. The
target-context-to-sidecar-block preparation now lives behind
`dspark_session_prepare_from_target_context`, with probe logging supplied by
callbacks. A development-only `DS4_DSPARK_PROBE=1` ordinary-session hook now
captures target hidden states from the normal graph path into session-owned GPU
scratch and builds session-owned dry draft candidates at sync/eval time without
emitting or accepting them. A second development-only hook,
`DS4_DSPARK_VERIFY=1`, uses the same live capture/build path but logs concise
prepared rows, then verifies the previous first draft candidate against the
target argmax from the current logits and the actual token passed to eval. It
now also runs a private acceptance-eligibility gate over the verifier result,
using optional confidence and logit-margin thresholds. Its no-commit simulator
now checks each later candidate against exact target suffix logits through the
normal one-token decode kernel, then restores the target compressor frontiers
before normal evaluation continues. It does not enable DSpark speculative
decoding. A third development-only mode, `DS4_DSPARK_COMMIT_PROBE=1`, can
commit the already-selected first token only when it is `stream_eligible`. The
probe uses the ordinary target decode/capture path and then skips its duplicate
normal eval, so output remains the sampled token. It never commits un-emitted
DSpark suffix rows.

Phase 0.21 adds the first actual multi-token *greedy CLI* delivery experiment,
behind the separate explicit `DS4_DSPARK_MULTI_COMMIT=1` environment variable.
It uses the exact suffix simulator to establish an accepted greedy prefix,
restores the virtual target state, then replays the accepted tokens through the
ordinary target graph so the live checkpoint, logits, and DSpark capture state
remain identical to sequential greedy decoding. The session returns that
committed ordered token batch to the caller for immediate emission; it keeps no
hidden output queue. When fewer than two tokens can be committed, it safely
falls back to one ordinary target token. The CLI enables this only for
`--temp 0` one-shot and chat generation. Server, agent, and eval frontends are
intentionally not wired yet because their forced-token, stop, tool, and
structural control paths need an explicit queue-aware policy. This path is
correctness instrumentation, not a performance path: its restore/replay means
accepted tokens are evaluated twice, and no benchmark claim follows from it.

Phase 0.22 removes that restore/replay from the multi-commit path while keeping
the same greedy output contract. A capture-aware target top-only decode now
retains DSpark hidden-state capture as it verifies each next suffix row. Once
row 0 is stream-eligible, every target decode that tests a later candidate is
also an actual committed prefix token: it advances the target graph, target
context, and session checkpoint directly. If a suffix is rejected, ds4 reads
the already-produced next logits once and returns that live accepted prefix. If
the selected limit is fully accepted, only the final accepted token takes the
ordinary full-logit decode needed for the next sampling step. The previous
rollback simulator remains in observation-only verifier/commit-probe paths;
the greedy multi path neither snapshots nor restores target frontiers. SSD
streaming uses the existing full-logit capture helper for this experiment. This
reduces duplicate target evaluation, but CPU sidecar drafting, target-context
readback, and explicit diagnostic logging still mean it is not a benchmarkable
production runtime.

Phase 0.23 makes target-layer capture incremental. The capture carrier now
stores a local `pos0` plus span length, so its raw GPU and CPU HC buffers are
sized only to the largest active prompt/suffix span, not the full sidecar
window. Completion reads and averages only those new rows into the persistent
target context; normal decode therefore transfers one new target HC row per
captured layer rather than rereading the whole prefix each time. The averaged
target context and the sidecar's CPU working state are still separate full
session allocations, and all `main_proj`, stage-block, head, Markov, and
confidence computation remains CPU-only. This is a storage/capture foundation
for the GPU sidecar port, not a performance claim or a benchmarkable runtime.

Phase 0.24 adds the first GPU sidecar computation boundary behind
`DS4_DSPARK_GPU_BRIDGE=1`. For every captured span, the bridge uses the existing
GPU HC weighted-sum primitive to average four streams for target layers
`40,41,42`, packs those rows on device, runs the sidecar Q8_0 `main_proj`, and
applies F32-weighted `main_norm` into a persistent session-owned GPU `main_x`.
Only those two sidecar weight ranges are mapped. The bridge reads back packed
context and normalized `main_x` for parity against the CPU implementation; the
default `main_x` max-absolute tolerance is `0.005`, configurable through
`DS4_DSPARK_GPU_BRIDGE_TOLERANCE`. A mismatch is counted and logged but cannot
change CPU-authoritative drafting or output. Greedy bridge-only CLI runs are
routed through the ordinary session loop so capture actually occurs. SSD
streaming is deliberately unsupported in this diagnostic bridge. Stage blocks,
heads, Markov, and confidence remain CPU-only, and no performance claim follows.

Phase 0.25 adds a complete GPU stage-0 parity path behind the separate explicit
`DS4_DSPARK_GPU_STAGE0=1` environment variable. This mode implies the Phase
0.24 bridge/capture mode, maps only the validated `mtp.0` layer spans, consumes
the persistent GPU `main_x`, and owns a dedicated session-side GPU raw-KV
buffer plus stage-output buffer. It deliberately does not reuse or mutate any
target-model KV cache. DSpark's five block queries are all allowed to see the
full context-plus-block KV set, matching the CPU/reference contract; the GPU
path therefore issues five unmasked single-query attention calls instead of
reusing normal causal batch prefill attention. The attention path uses
DSpark RoPE, FP8 non-RoPE KV rounding, F16 raw-KV storage, grouped output
projection, and HC post. The FFN then borrows the target graph's transient
batch workspace with directional steering disabled, but uses stage-0 sidecar
weights and leaves target persistent state untouched.

CPU stage execution, later DSpark stages, draft candidates, and generated
tokens remain authoritative. For a fair diagnostic, the opt-in path reads the
exact persistent GPU `main_x` back and reruns CPU stage 0 with that same input;
the comparator has its own host KV scratch, and GPU attention/full-stage HC are
compared against this same-input reference.
This intentionally duplicates CPU stage work and synchronizes/readbacks, so it
is not benchmarkable. Parity requires finite CPU/GPU HC and defaults to a
`0.03` relative-RMS full-stage tolerance, configurable with
`DS4_DSPARK_GPU_STAGE0_TOLERANCE`. Absolute max/RMS, reference RMS, relative
RMS, bridge-input drift, and attention-boundary drift remain logged for
regression visibility. A mismatch is counted but cannot alter CPU drafting or
output. The current implementation requires target graph prefill capacity at
least the five-token DSpark block and continues to reject SSD streaming through
the bridge guard. Existing `DS4_METAL_GRAPH_DUMP_*` controls can select the
`dspark_stage0_*` diagnostic checkpoints.

Phase 0.26 generalizes the GPU sidecar state and runner to stage-indexed arrays
and adds stage 1 behind `DS4_DSPARK_GPU_STAGE1=1`. Stage 1 implies stage 0 and
the GPU bridge, maps its own three sidecar spans, owns independent GPU KV and
output buffers plus independent host reference scratch, and consumes the
persistent GPU stage-0 HC output through a device-to-device copy. That copy is
encoded only after the GPU command batch begins because `ds4_gpu_tensor_copy`
requires an active batch. A per-stage validity chain prevents stage 1 from
running if stage 0 did not produce a valid GPU output, and labeled execution
boundaries identify setup, HC, Q/KV, attention, output, FFN, or command failures
without enabling tensor dumps.

Stage-1 parity uses a CPU reference fed the exact GPU stage-0 HC readback and
the exact persistent GPU `main_x`; its host KV scratch is separate from both
authoritative CPU stages and GPU sidecar KV. Stage 0 keeps its `0.03` default
relative-RMS tolerance, while stage 1 defaults to `0.04` through the independent
`DS4_DSPARK_GPU_STAGE1_TOLERANCE` override. The wider stage-1 band accounts for
occasional routed-FFN amplification while attention-boundary relative RMS stays
tightly logged. CPU stage 0/1/2 execution, draft construction, and generated
tokens remain authoritative. Stage 2 and the final HC/Markov/confidence heads
are still CPU-only, and all GPU-stage modes remain synchronization-heavy parity
diagnostics rather than performance paths.

Phase 0.27 enables the final transformer stage behind
`DS4_DSPARK_GPU_STAGE2=1`. Stage 2 implies stages 1 and 0 plus the bridge, and
uses the already-generalized stage arrays without introducing another execution
path: it maps its own three sidecar spans, allocates its own GPU KV/output and
host parity scratch, and consumes persistent GPU stage-1 HC through the same
command-batched device copy. Its same-input CPU reference receives the exact
GPU stage-1 HC readback and persistent GPU `main_x`, with independent host KV.
Stage 2 shares the later-stage `0.04` relative-RMS default and has the separate
`DS4_DSPARK_GPU_STAGE2_TOLERANCE` override.

All three DSpark transformer stages can now execute as one GPU HC chain for
parity, but none of their GPU outputs are authoritative. The CPU three-stage
chain still feeds the final HC head, final norm, Markov/confidence heads, draft
candidates, and generated tokens. Stage 2 does not map or execute those global
output tensors yet. This mode still performs CPU reference stages, readbacks,
logging, and synchronization, so it remains unsuitable for performance claims.

Phase 0.28 extends that diagnostic GPU chain through the stage-2 HC collapse
and final weighted RMS norm behind `DS4_DSPARK_GPU_HEAD=1`. This option implies
GPU stages 2/1/0 and the bridge. It maps only `hc_head_base`, `hc_head_fn`,
`hc_head_scale`, and `final_norm`, then runs the existing batched GPU primitives
over persistent GPU stage-2 HC into session-owned `gpu_head_plain` and
`gpu_head_norm` tensors. The parity reference reruns the CPU head from the exact
GPU stage-2 readback, so the comparison isolates head implementation drift from
the larger CPU/GPU transformer-stage drift. The ordinary CPU HC chain still
feeds base logits, Markov logits, confidence, draft candidates, and generated
tokens. Head mismatches are counted and logged but remain nonfatal. The default
same-input relative-RMS tolerance is `1e-5`, configurable through
`DS4_DSPARK_GPU_HEAD_TOLERANCE`. Readbacks and duplicate CPU execution keep this
strictly in correctness-observer territory, not a benchmarkable runtime.

Phase 0.29 extends the observer through base and Markov logits behind
`DS4_DSPARK_GPU_LOGITS=1`, which implies the complete GPU head/stage/bridge
chain. It adds reusable Metal BF16-to-F32 row gathering and batched BF16
matrix-vector projection primitives for the sidecar's Markov tensors. After the
authoritative CPU draft loop has selected its five predecessor tokens, the GPU
path gathers those exact `markov_w1` rows, projects persistent GPU
`final_norm` through the base Q8 LM head, projects the Markov embeddings through
BF16 `markov_w2`, adds both vocabulary vectors, and computes top-1 ids on
device. Using CPU-selected predecessor tokens is intentional: it compares all
five rows without allowing a GPU top-token difference to change later observer
inputs.

The same-input CPU reference consumes the exact GPU final-norm readback and the
same predecessor-token list. Base, Markov, and combined logit relative RMS plus
same-input and CPU-authoritative top-token agreement are logged. The default
relative-RMS band is `0.01`, configurable with
`DS4_DSPARK_GPU_LOGITS_TOLERANCE`; it is governed by the existing Q8 GPU/CPU
output-projection arithmetic, while the new BF16 Markov path is substantially
tighter. Confidence, candidate storage, acceptance, commits, and generated
tokens remain CPU-authoritative. Full vocabulary readbacks and duplicate CPU
projections make this a correctness observer only.

Phase 0.30 completes the diagnostic GPU candidate record behind
`DS4_DSPARK_GPU_CONFIDENCE=1`, which implies GPU logits and the entire preceding
GPU chain. It maps the small BF16 `confidence_proj` vector and adds a reusable
split-input BF16 dot primitive: the kernel consumes persistent GPU
`final_plain` and the already-gathered GPU Markov embedding directly, avoiding
an intermediate 4096+256 feature-concatenation buffer. It writes both confidence
logits and stable-sigmoid probabilities for all five rows. The observer combines
those outputs with GPU combined-logit top-2 values into a separate
`gpu_candidate[16]` block containing predecessor token, top/alternate token and
logits, confidence logit, and confidence probability. Draft reset invalidates
this block, and no production path consumes it.

Confidence parity uses a CPU reference fed exact GPU `final_plain` readback and
the same Markov predecessor rows. It separately logs drift from the existing
CPU-authoritative confidence, which includes accumulated transformer-stage
differences and is not charged to the same-input confidence kernel. The default
logit-relative/probability-absolute tolerance is `1e-5`, configurable through
`DS4_DSPARK_GPU_CONFIDENCE_TOLERANCE`. Candidate top-2 values are assembled from
the GPU combined-logit readback and checked against the GPU top-1 reduction.
The completed GPU candidate block remains observational and its rows 1-4 still
use CPU-selected predecessor tokens; acceptance, commits, and generated output
remain CPU-authoritative.

Phase 0.31 adds an independent self-fed GPU candidate chain behind
`DS4_DSPARK_GPU_CHAIN=1`, which implies the complete GPU confidence/logit/head
chain. Phase 0.30's CPU-fed GPU candidate structs are preserved first. The new
shadow runner then reuses the temporary Markov/logit tensors row by row: it
starts from the block anchor token, gathers one BF16 Markov embedding, projects
and combines one vocabulary row, reduces top-1 on GPU, reads back only that id,
and uses it as the next row's predecessor. After all five dependent rows are
complete, confidence is projected in one batch and a separate
`gpu_chain_candidate[16]` block is assembled. Draft reset invalidates both GPU
candidate blocks. Completion and parity are represented separately by
`gpu_chain_candidate_valid` and `gpu_chain_parity_ok`; partial sequential
failure also invalidates the rewritten shared GPU-logit scratch.

Each self-fed row is compared with CPU math given that exact GPU-selected
predecessor, isolating arithmetic parity even if a future token divergence
changes later inputs. The observer also reports agreement with the GPU top-1
reduction, the earlier CPU-fed GPU candidates, and current CPU-authoritative
candidates. It inherits `DS4_DSPARK_GPU_LOGITS_TOLERANCE` and
`DS4_DSPARK_GPU_CONFIDENCE_TOLERANCE`; no looser chain-specific threshold was
introduced. The five per-row command synchronizations/readbacks are deliberate
diagnostic scaffolding, not a production schedule. No verifier, acceptance,
commit, or output path reads the self-fed candidate block yet.

Phase 0.32 adds the first guarded GPU proposal-source experiment behind
`DS4_DSPARK_GPU_CANDIDATES=1`, which implies the complete self-fed GPU chain.
The CPU candidate block is still built first and remains untouched until GPU
preparation finishes. A GPU block may replace `d->draft[]` only when it is
complete, `gpu_chain_parity_ok`, row-count compatible, vocabulary-valid,
predecessor-continuous from the anchor, finite, and has confidence probabilities
inside `[0,1]`. Selection copies the self-fed candidates into `d->draft[]`,
updates confidence scratch and the final predecessor, and records the active
source. Any failed invariant or parity result increments fallback counters and
leaves the original CPU proposals active.

This is the first phase where GPU candidate values can affect which speculative
tokens are proposed, including GPU confidence values used by the existing gate.
They still cannot directly authorize output. The ordinary target logits verify
row 0, each suffix row is checked through the normal target decode path, and a
miss falls back through the established exact stream behavior. CPU sidecar
execution, full parity observers, vocabulary readbacks, and the five per-row GPU
synchronizations are still mandatory in this mode, so it is explicitly not a
performance runtime or benchmark target.

Phase 0.33 broadens guarded-proposal correctness coverage without changing the
runtime path. The executable integration harness
`tests/dspark_gpu_candidates_correctness.sh` compares exact greedy stdout from
no-DSpark baseline runs and `DS4_DSPARK_GPU_CANDIDATES=1` plus direct
multi-commit runs. It covers numeric reasoning, an Italian factual response, a
medium factual-recall prompt near the sidecar window, and a resumed two-turn
chat. A separate strict-confidence case requires chain parity failure, CPU
proposal fallback, no GPU-source selection, and unchanged stdout. The harness
uses local model defaults with `DS4_TEST_MODEL` and `DS4_TEST_DSPARK_MODEL`
overrides, retains logs automatically on failure, and never interprets printed
token rates.

The medium fixture is intentionally sized so its rendered prompt reaches
context 119 while remaining under the current 128-token DSpark sidecar window.
An earlier 203-token draft fixture correctly produced identical target output
but skipped DSpark; it was shortened so the regression actually exercises GPU
proposal selection near the eligibility boundary.

Phase 0.34 replaces the five synchronized self-fed GPU rows with ordered
dispatches in one Metal command buffer. A small BF16 gather kernel reads row
zero from the host-supplied anchor and later rows from the previous device
top-id; existing parallel matvec, add, top-k, and confidence kernels do the
remaining work. `DS4_DSPARK_GPU_CHAIN_HOST=1` restores the old host-stepped
schedule as a diagnostic oracle. The default device schedule still performs
the duplicate CPU proposal block and full parity readbacks, so it is not yet a
lean runtime or benchmark target.

Phase 0.35 adds a separate proposal-runtime boundary behind
`DS4_DSPARK_GPU_RUNTIME=1`. It implies the GPU candidate chain but bypasses the
CPU vocabulary draft loop and the CPU-fed GPU logits/confidence observers. The
self-fed chain now computes its own batched base logits, performs a device
top-2 reduction, gathers only the ten selected logit values, and reads back
those ids/values plus five confidence records. Structural and numeric checks
gate the resulting proposal block, and exact target verification remains the
only authority for commits and output.

`DS4_DSPARK_GPU_CANDIDATES=1` retains the full parity-observer behavior,
including CPU proposals, full-vocabulary comparisons, tolerance checks, and
strict fallback. Phase 0.35 does not yet remove the earlier bridge, transformer
stage, or HC-head CPU references/readbacks from runtime mode; those remain the
next lean-sidecar boundary. Neither mode is a benchmark target yet.

Phase 0.36 extends the runtime boundary through the complete sidecar. Captured
target HC now feeds GPU averaging and `main_proj/main_norm` before any host
capture conversion; runtime returns from capture without reading HC, packed
context, or `main_x`. Preparation creates only the five initial token HC
embeddings on CPU, then executes stages 0/1/2 and the HC head with device-owned
intermediate state. It skips CPU main projection, CPU transformer stages, CPU
head references, and every bridge/stage/head output readback.

Runtime GPU failure is fatal to that DSpark preparation cycle because no CPU
sidecar fallback was computed. The normal target stream remains authoritative
and unchanged. `DS4_DSPARK_GPU_CANDIDATES=1` still runs the complete CPU/GPU
observer with parity references, tolerances, readbacks, and fallback. Runtime
still allocates common CPU session scratch, borrows target graph batch workspace,
and emits development logs, so this phase is not yet a benchmark target.

Phase 0.37 makes successful runtime execution quiet by default. The explicit
`DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS=1` switch restores successful mapping,
bridge, stage, head, compact-chain, candidate-source, verifier, acceptance, and
multi-commit telemetry. Observer modes remain verbose without this switch, and
runtime failures still report errors even when diagnostics are off.

The compact proposal boundary has one required GPU command-buffer completion.
After that wait, host code copies 120 bytes per five-row block from managed
buffers: five top-1 ids, ten top-2 ids, ten selected logit values, five
confidence logits, and five confidence probabilities. These five tiny copies
do not add GPU waits. Top-2 margins and confidence are needed by the existing
gate, and candidate ids are needed by exact target verification.

Phase 0.38 adds a reproducible user-run baseline/runtime benchmark protocol,
without executing it. `speed-bench/run_dspark_comparison.py` requires explicit
`--confirm-idle`, strips inherited `DS4_DSPARK_*` variables, keeps runtime
diagnostics off, clears inherited profiling/tracing/timing/dump instrumentation,
and alternates baseline/runtime order across measured pairs. Non-instrumentation
`DS4_*` tuning variables are preserved and recorded.
It uses a fixed greedy 64-token workload that remains within the current
128-token sidecar window and rejects any baseline/runtime stdout drift.

The runner preserves raw stdout/stderr, CSV measurements, commands, git and
machine metadata, process and thermal snapshots, model sizes/mtimes, prompt
hash, and median plus paired-speedup summaries under ignored
`speed-bench/local-runs/`. One warmup per mode, three measured pairs, and a
10-second cooldown are defaults. Only the user may run the actual measurement
after confirming the machine is idle.

The design choice was to keep this separate from legacy `--mtp`. We do not
guess dynamically whether `--mtp` points at legacy MTP or DSpark. The first
hook is explicit and conservative:

- `--mtp FILE` remains the legacy MTP addon path.
- `--dspark FILE` means "validate this as a DSpark drafter sidecar".
- Supplying both is an error.
- DSpark validation happens before the `--inspect` early return, so
  `./ds4 --inspect --model base.gguf --dspark ds4-dspark.gguf` should validate
  the sidecar.
- `--dspark-probe` is a development diagnostic. It requires `--dspark`, runs a
  prompt through graph layer slices ending at target layers `40,41,42`,
  averages the captured HC streams to 4096-wide context vectors, runs
  DSpark `main_proj/main_norm`, executes the three-stage sidecar block on CPU,
  runs BF16 Markov-biased logits and BF16 confidence scores with an internal
  dry-run argmax chain, prints vector/logit stats, and exits without emitting,
  accepting, appending, generating, or speculating tokens.
- `DS4_DSPARK_PROBE=1` is a development-only ordinary-session hook. During
  normal sync/eval it copies target-layer hidden states from the live graph
  path into session-owned GPU scratch, reads/means those HC streams into the
  DSpark target context, prepares the live session-owned DSpark state, builds a
  real internal `d->draft[]` candidate block, logs those dry draft rows, and
  then returns to normal generation. It is deliberately diagnostic, non-fatal,
  and not a benchmark or production runtime path.
- `DS4_DSPARK_VERIFY=1` is the first verifier scaffold. It enables the same
  live target-state capture as the probe hook, builds `d->draft[]` for each
  prefix, and on the next eval compares `d->draft[0]` to both the target argmax
  from the current logits and the actual token selected by the sampler/caller.
  It logs cumulative `target_hit` and `token_hit` counters, resets the draft
  after verification, and does not accept, emit, append, or speculate tokens.
- The verifier gate has two optional thresholds:
  `DS4_DSPARK_VERIFY_MIN_CONFIDENCE` in `[0,1]` and
  `DS4_DSPARK_VERIFY_MIN_MARGIN` in `[0,+inf)`. A draft is
  `greedy_eligible` when it is fresh, matches the target argmax, and passes
  both thresholds. It is `stream_eligible` only when it is also the actual
  token selected by the current normal generation stream. Both are logged and
  counted only; neither causes a commit.
- `DS4_DSPARK_VERIFY=1` also runs a no-commit exact acceptance simulator. It
  checks the first candidate from the current target logits, then evaluates
  each required hypothetical predecessor with the exact one-token target decode
  kernel to check later rows. It reports the resulting greedy acceptance depth,
  stops at a target miss, gate failure, context limit, or diagnostic failure,
  and restores target compressor frontiers before normal evaluation continues.
  It never changes the session checkpoint, host logits, sampling, output, or
  token acceptance. A failed rollback invalidates the session and returns an
  error rather than continuing with possibly changed target state.
- `DS4_DSPARK_COMMIT_PROBE=1` implicitly enables the verifier and adds one
  explicit state-transition experiment. If row 0 passes the gate and equals the
  actual selected token (`stream_eligible`), ds4 first restores the hypothetical
  suffix, then evaluates and commits that same selected token through the
  ordinary graph decode/capture path. It emits no token itself and skips the
  duplicate ordinary eval. If the gate or stream check fails, it logs `skipped`
  and falls through unchanged. It never commits rows 1+ because they have not
  been emitted and would alter the normal sampled stream.
- `DS4_DSPARK_GPU_STAGE0=1` implies `DS4_DSPARK_GPU_BRIDGE=1` behavior and
  routes greedy CLI runs through the ordinary session loop. It maps/runs only
  stage 0 on GPU, compares it against CPU, and by itself never feeds the GPU
  output into stage 1 or token generation.
- `DS4_DSPARK_GPU_STAGE1=1` implies GPU stage 0, device-copies stage-0 GPU HC
  into the generic runner, and maps/runs stage 1 with independent KV/output and
  parity state. It does not feed GPU stage 1 into CPU stage 2 or drafting.
- `DS4_DSPARK_GPU_STAGE2=1` implies GPU stages 0/1, device-copies stage-1 GPU HC
  into the generic runner, and maps/runs stage 2 with its own KV/output and
  parity state. It does not feed GPU stage 2 into the final HC or draft heads.
- `DS4_DSPARK_GPU_HEAD=1` implies the complete GPU stage chain, maps/runs the HC
  collapse and final norm from persistent GPU stage-2 HC, and compares both
  outputs with a same-input CPU reference. CPU logits and drafting remain
  authoritative.
- `DS4_DSPARK_GPU_LOGITS=1` implies the GPU head chain, maps the two BF16 Markov
  tensors, and compares GPU base/Markov/combined logits plus top-1 ids against a
  same-input CPU reference. The observer's later-row predecessor tokens come
  from the completed CPU draft block; GPU token selection never feeds back.
- `DS4_DSPARK_GPU_CONFIDENCE=1` implies GPU logits, maps/runs the split-input
  BF16 confidence projection, and assembles a complete separate GPU candidate
  block. It remains diagnostic; later rows still use CPU predecessor tokens and
  no verifier or commit path reads the GPU candidates.
- `DS4_DSPARK_GPU_CHAIN=1` implies GPU confidence and builds a second candidate
  block whose rows feed their own GPU top tokens into subsequent Markov
  embeddings. It compares against same-input CPU, CPU-fed GPU, and authoritative
  CPU candidates but remains entirely observational.
- `DS4_DSPARK_GPU_CANDIDATES=1` implies the self-fed chain and may replace CPU
  proposals only after complete structural and parity validation. Failed gates
  retain CPU proposals; successful GPU proposals still require exact target
  verification before any commit.
- `DS4_DSPARK_GPU_RUNTIME=1` implies GPU candidates but skips CPU vocabulary
  drafting and all sidecar parity observers. Captured HC, `main_x`, transformer
  stages, HC head, vocabulary logits, and Markov dependencies stay on device;
  only compact top-2 and confidence records are read back. Structural checks
  and exact target verification remain mandatory.
- `DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS=1` restores successful runtime progress
  and statistics logs. Leave it unset for manual performance measurements.
- `tests/dspark_gpu_candidates_correctness.sh` is the reproducible real-model
  correctness matrix for the guarded source. It is optional, accelerator-heavy,
  and separate from model-free `ds4_test`; `DS4_TEST_DSPARK_MODE=runtime`
  selects the compact runtime matrix while the default remains `observer`. It
  checks exact output and source behavior, not performance.
- Probe/verify/commit-probe hooks alone are not called by the CLI `--temp 0`
  argmax fast path. GPU bridge, GPU stages 0/1/2, GPU head/logits, and greedy
  multi-commit modes do explicitly route through the ordinary session loop. Use a seeded
  nonzero-temperature smoke for the other ordinary-session diagnostics.
- No benchmarks should be run automatically by Codex. The user wants tok/s
  benchmarks to be run manually on an otherwise idle machine.

## Sources And Contract

Primary references used for the current phase:

- ds4 discussion: <https://github.com/antirez/ds4/issues/468>
- DeepSpec repo: <https://github.com/deepseek-ai/DeepSpec>
- Official DeepSeek V4 Flash DSpark inference source:
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/model.py>
- Converted DSpark GGUF sidecar for validation:
  <https://huggingface.co/buckets/lobanov/ds4/tree/ds4flash-dspark.gguf>
  - Direct download:
    <https://huggingface.co/buckets/lobanov/ds4/resolve/ds4flash-dspark.gguf?download=true>
  - Size shown by Hugging Face on 2026-07-05: 11.5 GB.
  - Expected byte size from HTTP headers: `11489939840`.
  - Local target path: `gguf/ds4flash-dspark.gguf`.
  - Resume command:
    ```sh
    curl -L --fail -C - \
      -o /Users/deathcodevision/dev/ds4/gguf/ds4flash-dspark.gguf \
      'https://huggingface.co/buckets/lobanov/ds4/resolve/ds4flash-dspark.gguf?download=true'
    ```
- A shallow DeepSpec checkout was used under `/tmp/DeepSpec`; do not assume it
  still exists.

DSpark contract we encoded for DeepSeek-V4-Flash DSpark:

- Three draft stages: `mtp.0`, `mtp.1`, `mtp.2`.
- Block size: `5`.
- Noise token id: `128799`.
- Markov rank: `256`.
- Target base layer ids: `40, 41, 42`.
- Stage 0 global inputs: `mtp.0.main_proj.weight`,
  `mtp.0.main_norm.weight`.
- Stage 2 output/head tensors: `mtp.2.norm.weight`,
  `mtp.2.hc_head_base.weight`, `mtp.2.hc_head_fn.weight`,
  `mtp.2.hc_head_scale.weight`,
  `mtp.2.markov_head.markov_w1.weight`,
  `mtp.2.markov_head.markov_w2.weight`,
  `mtp.2.confidence_head.proj.weight`.
- Every stage requires the DS4 layer-style suffixes:
  `hc_attn_fn.weight`, `hc_attn_scale.weight`, `hc_attn_base.weight`,
  `attn_norm.weight`, `attn_q_a.weight`, `attn_q_a_norm.weight`,
  `attn_q_b.weight`, `attn_kv.weight`, `attn_kv_a_norm.weight`,
  `attn_sinks.weight`, `attn_output_a.weight`, `attn_output_b.weight`,
  `hc_ffn_fn.weight`, `hc_ffn_scale.weight`, `hc_ffn_base.weight`,
  `ffn_norm.weight`, `ffn_gate_inp.weight`, `exp_probs_b.bias`,
  `ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`,
  `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`,
  `ffn_down_shexp.weight`.

Important nuance: metadata keys under `deepseek4.dspark.*` are treated as
optional. If absent, ds4 uses the defaults above. If present, values must match.
This avoids rejecting current GGUFs that only communicate DSpark-ness through
tensor names.

Phase 0.5 shape/binding facts from the local
`gguf/ds4flash-dspark.gguf` sidecar:

- `exp_probs_b.bias` is present in all three stages and is now required. The
  earlier Phase 0 note that treated it as optional was wrong.
- The sidecar has 81 tensors: 9 global/output tensors plus 24 per-stage tensors
  across 3 stages.
- `mtp.0.main_proj.weight` is Q8_0 `[12288,4096]`, i.e.
  `[3*N_EMBD,N_EMBD]`.
- Each DSpark stage has the same DS4 layer-style attention/FFN shape as the
  legacy one-block MTP validator expects.
- `mtp.2.markov_head.markov_w1.weight` and `markov_w2.weight` are BF16
  `[markov_rank,N_VOCAB]` = `[256,129280]`.
- `mtp.2.confidence_head.proj.weight` is BF16 1D
  `[N_EMBD + markov_rank]` = `[4352]`.
- The current phase still has no DSpark token acceptance/emission, no
  production GPU sidecar runtime path, no expert stats, and no benchmark
  claims.
- Official DeepSeek-V4-Flash-DSpark inference captures each target layer by
  appending `h.mean(dim=2)` after layers in `dspark_target_layer_ids`. The
  active probe therefore uses mean over the four HC streams as the DSpark
  `main_proj` bridge. The earlier next-layer-attn-HC-pre/output-HC-head
  collapse hypothesis is retained only as a probe comparison, not as runtime
  contract.

## Files Changed So Far

- `ds4.h`
  - Added `dspark_path` to `ds4_engine_options`.
  - Added public `ds4_dspark_config`.
  - Added validation helper prototypes:
    `ds4_dspark_config_init_defaults` and
    `ds4_dspark_tensor_names_validate`.

- `ds4.c`
  - Added DSpark default config and tensor-name validation helpers.
  - Added model-backed DSpark validation for required tensors and optional
    metadata.
  - Added `ds4_dspark_weights` and `dspark_weights_bind` for the official
    DSpark sidecar tensor layout.
  - Added DSpark layout validation for global inputs, three per-stage
    transformer blocks, final HC head tensors, Markov heads, and the confidence
    head.
  - Added `ds4_engine_dspark_probe`, a graph-backend-only diagnostic that uses
    layer-slice evaluation to capture target-layer HC states, averages HC
    streams into the official DSpark target-state bridge, runs DSpark
    `main_proj/main_norm`, executes a CPU-only diagnostic sidecar block through
    all three DSpark stages, and now runs BF16 Markov bias plus BF16 confidence
    prediction in a dry-run argmax chain.
  - Factored the row-level BF16 Markov embedding/projection, base LM-head
    logits, corrected logits top-2, and confidence score into private
    `dspark_draft_step_scratch` / `dspark_draft_step_result` /
    `dspark_draft_step_cpu` helpers. This is still probe-only, but it is the
    first reusable runtime-shaped boundary for DSpark draft rows.
  - Added lazy private `dspark_session_state` on graph sessions. It owns the
    DSpark sidecar `main_x` raw window, per-stage sidecar KV row buffers,
    draft-block HC/head buffers, confidence logits, and the draft-step scratch.
    The state is reset on session invalidation, layer-slice timeline commits,
    and rewind. Normal `--dspark` validation/generation does not allocate it
    until a DSpark diagnostic/runtime path explicitly asks for it.
  - Extracted `dspark_session_prepare_from_target_context`, which takes
    flattened target-layer context vectors plus an anchor token and fills the
    session-owned `main_x`, sidecar stage KV rows, block hidden states, and
    final plain/norm buffers. The probe supplies callbacks for main/state and
    stage logging; the helper itself does not print or emit tokens.
  - Added `DS4_DSPARK_PROBE=1`, a development-only ordinary-session hook that
    logs DSpark dry draft rows at sync/eval time. It now captures target-layer
    HC from live graph execution into session-owned GPU scratch, calls
    `dspark_session_prepare_from_target_context` on the live session, and does
    not emit, accept, append, or speculate tokens.
  - Added an optional `ds4_target_hc_capture` carrier through normal Metal
    prefill/decode wrappers. The carrier copies post-layer HC at target layers
    `40,41,42` into diagnostic scratch only when the env hook has explicitly
    enabled it; default callers still pass `NULL`.
  - Phase 0.23 makes that carrier span-local. It records the absolute capture
    start plus new-row count, writes GPU scratch at a local offset, and only
    reads/means the new rows into persistent target context on completion.
    Raw GPU/CPU HC scratch grows only to the largest active capture span rather
    than allocating for the whole sidecar window; the averaged context remains
    available for later DSpark preparation.
  - Phase 0.24 adds session-owned GPU bridge tensors for mean weights,
    layer-major target means, packed target context, projection scratch, and
    persistent `main_x`. `DS4_DSPARK_GPU_BRIDGE=1` maps only `main_proj` and
    `main_norm`, executes their GPU bridge for each capture span, and reports
    packed-context plus normalized-`main_x` CPU parity. CPU `main_x` remains the
    input to every current sidecar stage.
  - Added `dspark_draft_candidate` rows and `draft_valid` to
    `dspark_session_state`, plus `dspark_session_build_draft_candidates`.
    The builder converts the prepared DSpark block into session-owned draft
    candidate rows (`prev_token`, draft token, runner-up, logits, confidence)
    without logging or accepting tokens. The standalone probe and
    `DS4_DSPARK_PROBE=1` logger now consume that state.
  - Added `DS4_DSPARK_VERIFY=1`, an opt-in verifier scaffold for ordinary
    graph sessions. It reuses the live capture/build path, keeps cumulative
    target-argmax and sampled-token hit counters in `dspark_session_state`, and
    compares the previous first DSpark candidate before normal eval consumes
    the next token. It is observation-only and resets the draft after each
    verification attempt.
  - Added a private DSpark verifier gate with cumulative `greedy_eligible` and
    `stream_eligible` counters. The gate is controlled by
    `DS4_DSPARK_VERIFY_MIN_CONFIDENCE` and
    `DS4_DSPARK_VERIFY_MIN_MARGIN`; it only reports whether a candidate would
    be eligible for a future acceptance path and still never commits DSpark
    tokens.
  - Reworked the private no-commit acceptance simulator into an exact target
    suffix verifier. After its free row-0 check, it uses the ordinary one-token
    target decode kernel to check each later candidate and restores the saved
    compressor frontiers before the normal eval path resumes. It records
    checked rows, greedy acceptance depth, rejection/context/diagnostic states,
    and treats a failed restore as a session error.
  - Split generic speculative rollback frontiers from legacy MTP-only scratch
    in the graph allocator. A session created with `DS4_DSPARK_VERIFY=1` or
    `DS4_DSPARK_COMMIT_PROBE=1` gets only the generic frontier buffers needed
    for rollback; it does not receive MTP hidden-state scratch, MTP raw cache,
    or MTP speculative logits.
  - Added `DS4_DSPARK_COMMIT_PROBE=1`, an explicit first-token commit probe.
    It only acts for a `stream_eligible` row-0 candidate, restores the virtual
    suffix first, advances that actual selected token through the normal target
    graph/capture path, rebuilds the next DSpark candidate block, and prevents
    the caller from evaluating the same token twice. It never commits suffix
    candidates or emits output itself.
  - Factored the graph target-token evaluation/capture sequence into private
    `ds4_session_eval_graph_target_token`, shared by normal graph eval and the
    commit probe so their state transitions stay identical.
  - Added `DS4_DSPARK_MULTI_COMMIT=1`, an explicit greedy-only multi-token
    experiment. Phase 0.21 exact-verified a draft suffix, restored virtual
    target state, and replayed the accepted prefix for correctness.
  - Phase 0.22 replaces replay with a capture-aware top-only target decode for
    the greedy multi path. Each verified predecessor is retained as an actual
    committed prefix token; a suffix rejection reads the current target logits
    once, while a fully accepted limit evaluates only its final token normally.
    The generic rollback simulator remains available only to observation paths.
  - Added the public greedy-only session entry point
    `ds4_session_eval_dspark_greedy`; it returns zero when ordinary evaluation
    should proceed and never owns an emission queue.
  - The sidecar probe still does not emit, accept, append, generate, or
    speculate tokens.
  - Added `dspark_model`, `dspark_config`, and `dspark_ready` to `ds4_engine`.
  - `ds4_engine_open` now rejects `--mtp` plus `--dspark`, opens the DSpark
    sidecar, validates/binds it, and logs that runtime is disabled by default.
  - `ds4_engine_close` closes the DSpark model if it was opened.
  - The DSpark path does not set `mtp_ready`, bind legacy MTP draft weights, or
    affect generation.

- Frontend parsers
  - `ds4_cli.c`, `ds4_server.c`, `ds4_eval.c`, and `ds4_agent.c` all accept
    `--dspark FILE`.
  - `ds4_cli.c` accepts `--dspark-probe` as a one-shot diagnostic; the server,
    eval, and agent frontends do not expose this probe.
  - `ds4-agent` was easy to miss because shared help mentioned the flag once
    `ds4_help.c` was updated. When adding runtime flags, always check all
    frontends.
  - Phase 0.21 routes only the CLI one-shot and chat greedy loops through the
    returned DSpark token batch when `DS4_DSPARK_MULTI_COMMIT=1`. Server,
    agent, and eval deliberately remain sequential until their forced-token
    and structural-output handling has a safe batch policy.
  - Phase 0.24 routes bridge-only greedy CLI generation through the ordinary
    session loop when `DS4_DSPARK_GPU_BRIDGE=1`; this does not implicitly enable
    multi-commit.

- `ds4_help.c`
  - Runtime full help now documents `--dspark FILE`.
  - Diagnostics help now documents `--dspark-probe`.

- `README.md`
  - Added a short validation/probe note next to the existing MTP text.

- `tests/ds4_test.c`
  - Added `--dspark-validation`, a model-free test for defaults, rejection of
    legacy/ordinary MTP-looking names, and acceptance of a synthetic full DSpark
    tensor-name list.
  - Added `--dspark-shape-binding`, a model-free sanity test for the documented
    DeepSeek V4 Flash DSpark shape arithmetic. The real GGUF inspect path is
    what validates those dimensions against `ds4.c`'s private model-shape
    constants.

- `implementation_plan.md`
  - Imported from the user/Claude handoff. The plan direction was validated:
    Phase 0.5 should be shape and binding validation only, not runtime work.
  - Earlier adjustment made during Phase 0.5: keep Markov/confidence validation
    dimension-only, and do not widen the public API just to expose private
    shape constants to the model-free test. This was superseded in Phase 0.9
    after the actual sidecar tensors were confirmed as BF16 and the diagnostic
    probe began executing them.

## Verification Already Run

Build/check commands run after Phase 0/0.5 implementation:

```sh
make ds4 ds4-server ds4-eval ds4-agent ds4_test
./ds4_test --dspark-validation
./ds4_test --dspark-shape-binding
./ds4 --help runtime | rg -- '--dspark'
./ds4-server --help runtime | rg -- '--dspark'
./ds4-eval --help runtime | rg -- '--dspark'
./ds4-agent --help runtime | rg -- '--dspark'
git diff --check
```

No tok/s benchmark was run.

Real DSpark sidecar validation:

```sh
./ds4 --inspect \
  --model /Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /Users/deathcodevision/dev/ds4/gguf/ds4flash-dspark.gguf
```

Result on 2026-07-05: validation passed.

```text
ds4: DSpark drafter validated: /Users/deathcodevision/dev/ds4/gguf/ds4flash-dspark.gguf (block=5 target_layers=40,41,42 main_proj=[12288,4096] stages=3; runtime not enabled yet)
```

Phase 0.5 completion checks run on 2026-07-05:

```sh
make -B ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
make -B ds4
./ds4 --inspect \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf
make ds4 ds4-server ds4-eval ds4-agent ds4_test
```

The inspect output confirmed:

```text
ds4: DSpark drafter validated: gguf/ds4flash-dspark.gguf (block=5 target_layers=40,41,42 main_proj=[12288,4096] stages=3; runtime not enabled yet)
```

Phase 0.6 probe checks run on 2026-07-05:

```sh
make ds4 ds4-server ds4-eval ds4-agent ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
./ds4 --help diagnostics | rg -- '--dspark|--dspark-probe'
./ds4 --inspect \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
git diff --check
```

The real probe completed on the downloaded base+DSpark GGUFs. It captured
target layers `40,41,42`, collapsed them with the then-provisional
HC-to-plain bridge, ran `main_proj/main_norm`, and exited before
generation/speculation.
Observed last-token vector stats were finite:

```text
layer 40 hc rms=3.39173 plain rms=3.93296
layer 41 hc rms=3.54454 plain rms=4.50141
layer 42 hc rms=5.36406 plain rms=7.51114
main_proj rms=26.5668
main_norm rms=0.0882359
```

Phase 0.7 mean-HC bridge checks run on 2026-07-05:

```sh
make -B ds4
./ds4_test --dspark-validation --dspark-shape-binding
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
```

The probe now uses official mean-HC target states and keeps the old collapse
hypothesis only as a comparison. Observed last-token vector stats were finite:

```text
layer 40 hc rms=3.39173 mean_hc rms=2.65261 legacy_plain rms=3.93296
layer 41 hc rms=3.54454 mean_hc rms=2.75814 legacy_plain rms=4.50141
layer 42 hc rms=5.36406 mean_hc rms=4.00292 legacy_plain rms=7.51114
main_proj rms=17.8532
main_norm rms=0.0882224
```

The legacy comparison diffs for the same token were:

```text
layer 40 max_abs=7.43734 rms_abs=1.43737
layer 41 max_abs=16.5391 rms_abs=2.38056
layer 42 max_abs=92.0704 rms_abs=4.38825
```

Phase 0.8 sidecar-backbone probe checks run on 2026-07-05:

```sh
make ds4 ds4-server ds4-eval ds4-agent ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
./ds4 --help diagnostics | rg -- '--dspark-probe'
git diff --check
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
```

The probe captured target layers, projected all prompt rows through
`main_proj/main_norm`, built the DSpark noise-token block, executed all three
sidecar stages on CPU, applied the final DSpark HC head/norm, and computed row-0
base LM-head logits. Markov bias, confidence prediction, sampling, acceptance,
generation, and speculation were not executed.

Observed stats were finite:

```text
stage 0 hc block rms=4.15588
stage 1 hc block rms=5.53495
stage 2 hc block rms=9.07774
final_plain block rms=0.192528
final_norm block rms=0.22184
row0 base_logits rms=2.7109 top1=19923 logit=18.6582 top2=23166 logit=16.1603
```

Phase 0.9 Markov/confidence diagnostic checks run on 2026-07-05:

```sh
make -B ds4
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
```

The probe now validates and executes the BF16 Markov and confidence tensors in
diagnostic mode. It uses an internal dry-run argmax chain to choose the previous
token id for the next Markov row, but it does not emit, accept, append,
generate, or speculate those tokens.

Observed stats were finite:

```text
row0 base_logits rms=2.7109
row0 markov_bias rms=8.17752
row0 markov_logits rms=8.45111
confidence logits min=-0.0725716 max=2.41473 rms=1.1772
dry row 0 prev=128822 top1=19923 logit=20.6812 confidence=0.917943
dry row 1 prev=19923 top1=3 logit=19.901 confidence=0.715982
dry row 2 prev=3 top1=52780 logit=21.2248 confidence=0.559488
dry row 3 prev=52780 top1=236 logit=28.3849 confidence=0.604713
dry row 4 prev=236 top1=22651 logit=12.3115 confidence=0.481865
```

Phase 0.10 draft-step factoring checks run on 2026-07-05:

```sh
make ds4 ds4-server ds4-eval ds4-agent ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
make ds4_cpu.o
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
```

The real probe produced the same dry-run rows as Phase 0.9 after the refactor.
`make ds4_cpu.o` still reports the pre-existing CPU-only warning set, but no
DSpark probe helper warnings remain.

Phase 0.11 session-state checks run on 2026-07-06:

```sh
make ds4 ds4-server ds4-eval ds4-agent ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
make ds4_cpu.o
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
```

The real probe again produced the same dry-run rows and finite stats after
moving sidecar block/cache scratch into `dspark_session_state`. No token was
emitted, accepted, appended, generated, or speculated.

Phase 0.12 prepare-helper checks run on 2026-07-06:

```sh
make ds4 ds4-server ds4-eval ds4-agent ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
make ds4_cpu.o
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
```

The probe output order and dry-run rows stayed the same after moving
`main_x`, block initialization, stage execution, and final head/norm into
`dspark_session_prepare_from_target_context`.

Phase 0.13 ordinary-session dev-probe checks run on 2026-07-06:

```sh
make ds4 ds4-server ds4-eval ds4-agent ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
make ds4_cpu.o
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
DS4_DSPARK_PROBE=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 -p "Hello"
```

The standalone probe kept the same dry-run rows as Phase 0.12. The
ordinary-session dev hook logged the same `sync` rows for the prompt prefix,
then logged an `eval` block after the one normal generated token. The generated
token stream was still ordinary target-model generation; DSpark emitted,
accepted, appended, and speculated no tokens. The command prints normal CLI
prefill/generation rates, but this was a correctness smoke test, not a tok/s
benchmark.

Observed `sync` dry rows for `"Hello"`:

```text
row 0 prev=128822 top1=19923 confidence=0.917943
row 1 prev=19923 top1=3 confidence=0.715982
row 2 prev=3 top1=52780 confidence=0.559488
row 3 prev=52780 top1=236 confidence=0.604713
row 4 prev=236 top1=22651 confidence=0.481865
```

Observed first `eval` dry rows after the one normal generated token:

```text
row 0 prev=23166 top1=3 confidence=0.469495
row 1 prev=3 top1=1730 confidence=0.877522
row 2 prev=1730 top1=588 confidence=0.600236
row 3 prev=588 top1=442 confidence=0.707425
row 4 prev=442 top1=85 confidence=0.777417
```

Phase 0.14 live-target-context capture checks run on 2026-07-06:

```sh
make ds4 ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
make ds4-server ds4-eval ds4-agent
make ds4_cpu.o
git diff --check
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
DS4_DSPARK_PROBE=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 -p "Hello"
make ds4-server ds4-eval ds4-agent
make ds4_cpu.o
```

The standalone probe kept the same dry-run rows as Phase 0.13. The
ordinary-session dev hook no longer creates a separate capture session. Its
`sync` rows for `"Hello"` still match the standalone probe exactly:

```text
row 0 prev=128822 top1=19923 confidence=0.917943
row 1 prev=19923 top1=3 confidence=0.715982
row 2 prev=3 top1=52780 confidence=0.559488
row 3 prev=52780 top1=236 confidence=0.604713
row 4 prev=236 top1=22651 confidence=0.481865
```

The first `eval` rows changed versus Phase 0.13 because they now use the live
one-token decode path's captured target-layer HC rather than a replayed
layer-slice prefix. That is the intended runtime-shaped context:

```text
row 0 prev=23166 top1=3 confidence=0.452657
row 1 prev=3 top1=1730 confidence=0.879072
row 2 prev=1730 top1=588 confidence=0.596213
row 3 prev=588 top1=1762 confidence=0.699362
row 4 prev=1762 top1=440 confidence=0.538978
```

`make ds4_cpu.o` still reports the same pre-existing CPU-only warning set
(`output_hc`, streaming auto-cache helpers, expert profile helpers,
`sleep_sec`, first-bound-layer helper, and cancel callback). No new DSpark
warnings remained after this phase.

Phase 0.15 draft-candidate state checks run on 2026-07-06:

```sh
make ds4 ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
DS4_DSPARK_PROBE=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 -p "Hello"
make ds4-server ds4-eval ds4-agent
make ds4_cpu.o
```

The standalone probe kept the same dry-run rows and row-0 stats after the
manual Markov/confidence loop was replaced by
`dspark_session_build_draft_candidates`. The env-gated ordinary-session hook
also logged rows from `d->draft[]`; no DSpark token was emitted, accepted,
appended, or speculated.

Observed `sync` rows for `"Hello"` still matched the standalone probe:

```text
row 0 prev=128822 top1=19923 confidence=0.917943
row 1 prev=19923 top1=3 confidence=0.715982
row 2 prev=3 top1=52780 confidence=0.559488
row 3 prev=52780 top1=236 confidence=0.604713
row 4 prev=236 top1=22651 confidence=0.481865
```

Observed first `eval` rows in this smoke run, after the normal generated token
`19923`, were:

```text
row 0 prev=19923 top1=3 confidence=0.882954
row 1 prev=3 top1=342 confidence=0.778569
row 2 prev=342 top1=4571 confidence=0.743389
row 3 prev=4571 top1=7692 confidence=0.643284
row 4 prev=7692 top1=304 confidence=0.69097
```

`make ds4_cpu.o` still reports the same pre-existing CPU-only warning set.

Phase 0.16 verifier-scaffold checks run on 2026-07-06:

```sh
make ds4 ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
DS4_DSPARK_VERIFY=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 -p "Hello"
DS4_DSPARK_PROBE=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 -p "Hello"
make ds4-server ds4-eval ds4-agent
make ds4_cpu.o
```

The verifier-only smoke prepared a first draft after sync, then checked that
draft before normal eval consumed the sampled token:

```text
prepared at sync: anchor_token=128822 next_draft=19923 confidence=0.917943
eval next: draft=19923 target_top=19923 token=23166 target_hit=1/1 token_hit=0/1
prepared at eval: anchor_token=23166 next_draft=3 confidence=0.452657
```

This shows why the verifier logs both target argmax and selected token: the
DSpark draft agreed with the target model's greedy next token in this smoke,
while the normal sampler selected a different token. Generation remained normal
target-model generation; DSpark emitted, accepted, appended, and speculated no
tokens. The command prints normal CLI prefill/generation rates, but this was a
correctness smoke test, not a tok/s benchmark.

The probe-only smoke still printed the verbose `sync`/`eval` dry rows from
`d->draft[]`. `make ds4_cpu.o` still reports the same eight pre-existing
CPU-only warnings.

Phase 0.17 verifier-gate checks run on 2026-07-09:

```sh
make ds4 ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
DS4_DSPARK_VERIFY=1 \
DS4_DSPARK_VERIFY_MIN_CONFIDENCE=0.9 \
DS4_DSPARK_VERIFY_MIN_MARGIN=2.0 \
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 -p "Hello"
```

The thresholded verifier smoke reported:

```text
draft=19923 target_top=19923 token=23166
target_hit=1/1 token_hit=0/1
greedy_eligible=yes greedy_hit=1/1
stream_eligible=no stream_hit=0/1
confidence=0.917943 min_confidence=0.9
margin=3.01909 min_margin=2
```

This confirms the gate distinguishes a greedy acceptance candidate from a token
that would preserve the current sampled stream. DSpark still emitted, accepted,
appended, and speculated no tokens. The command prints normal CLI
prefill/generation rates, but this was a correctness smoke test, not a tok/s
benchmark.

Phase 0.18 no-commit acceptance-simulator checks run on 2026-07-09:

```sh
make ds4 ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
DS4_DSPARK_VERIFY=1 \
DS4_DSPARK_VERIFY_MIN_CONFIDENCE=0.9 \
DS4_DSPARK_VERIFY_MIN_MARGIN=2.0 \
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 -p "Hello"
```

Expected first verification semantics: when the first candidate passes the
greedy gate, the simulator reports `checked_rows=1/5`,
`known_accept_depth=1`, and `unverified_suffix=4`. Those four later rows are
not counted as accepted: verifying them requires target logits for hypothetical
suffix positions, which remains future work. The smoke is a correctness check,
not a tok/s benchmark.

The thresholded real-sidecar smoke produced exactly that first verification:

```text
draft=19923 target_top=19923 token=19923
greedy_eligible=yes stream_eligible=yes
checked_rows=1/5 known_accept_depth=1
first_gate=yes unverified_suffix=4
```

The generated stream remained normal target-model generation. The command did
print normal CLI rates, but no tok/s benchmark was run. `make ds4_cpu.o` still
has the same eight pre-existing CPU-only warnings.

Phase 0.19 exact-suffix verifier checks run on 2026-07-09:

```sh
make ds4 ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
make ds4-server ds4-eval ds4-agent
make ds4_cpu.o
git diff --check
DS4_DSPARK_VERIFY=1 \
DS4_DSPARK_VERIFY_MIN_CONFIDENCE=0.9 \
DS4_DSPARK_VERIFY_MIN_MARGIN=2.0 \
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 -p "Hello"
DS4_DSPARK_VERIFY=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0.7 --seed 42 -p "Hello"
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --nothink -n 2 --temp 0.7 --seed 42 -p "Hello"
```

With the confidence/margin thresholds above, the real sidecar verifier checked
two rows and found a one-token greedy prefix:

```text
checked_rows=2/5 greedy_accept_depth=1
stop=confidence rejected_row=1 rollback=restored
```

With default thresholds it checked three rows, accepted the first two under
greedy policy, and stopped at a target miss on row 2:

```text
checked_rows=3/5 greedy_accept_depth=2
stop=target miss rejected_row=2 rollback=restored
```

The seeded two-token ordinary-session smoke produced `Hello!` both with the
DSpark verifier enabled and in the no-DSpark baseline. During that verifier
run, the first cycle reported a greedy depth of two and the second a greedy
depth of one; each reported `rollback=restored`. These are correctness smokes,
not tok/s benchmarks.

Phase 0.20 first-token commit-probe checks run on 2026-07-09:

```sh
make ds4 ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
make ds4-server ds4-eval ds4-agent
make ds4_cpu.o
git diff --check
DS4_DSPARK_COMMIT_PROBE=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0.7 --seed 42 -p "Hello"
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --nothink -n 2 --temp 0.7 --seed 42 -p "Hello"
DS4_DSPARK_COMMIT_PROBE=1 \
DS4_DSPARK_VERIFY_MIN_CONFIDENCE=0.99 \
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0.7 --seed 42 -p "Hello"
DS4_DSPARK_COMMIT_PROBE=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 5 --seed 42 -p "Hello"
```

With default thresholds, both selected stream tokens were eligible and the
probe committed them once through the ordinary target path:

```text
commit_probe=ready
committed actual token=19923 at pos=10
committed actual token=3 at pos=11
```

The probe rebuilt its next candidate block after each commit and emitted no
extra DSpark tokens. Its output, `Hello!`, exactly matched the seeded no-DSpark
baseline. With `DS4_DSPARK_VERIFY_MIN_CONFIDENCE=0.99`, both rows reported
`commit_probe=skipped` and still produced the same normal output. These are
correctness smokes, not tok/s benchmarks.

The fixed high-temperature smoke selected token `6208` while the target and
DSpark draft were both `19923`. It correctly reported `stream_eligible=no` and
`commit_probe=skipped`, then ordinary evaluation produced the sampled token.

Phase 0.21 greedy multi-commit checks run on 2026-07-09:

```sh
make ds4 ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
make ds4-server ds4-eval ds4-agent
make ds4_cpu.o
git diff --check
DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0 -p "Hello"
DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_MULTI_COMMIT=1 DS4_DSPARK_VERIFY_MIN_CONFIDENCE=0.99 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0 -p "Hello"
```

The two-token run verified and returned two committed tokens, producing
`Hello!`, exactly matching the normal greedy baseline. The one-token boundary
correctly logged `verified depth below two` and returned the normal single
target token. The four-token run made two multi-token cycles (two returned
tokens each) and produced `Hello! How can`, again exactly matching baseline.
With `DS4_DSPARK_VERIFY_MIN_CONFIDENCE=0.99`, both rows were rejected as not
stream eligible and each fell back to one target token, still producing
`Hello!`. These are correctness smokes, not tok/s benchmarks.

Phase 0.22 direct-state multi-commit checks run on 2026-07-09:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_MULTI_COMMIT=1 DS4_DSPARK_VERIFY_MIN_CONFIDENCE=0.99 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0 -p "Hello"
printf 'Hello\\n/exit\\n' | DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0
git diff --check
```

The direct four-token run emitted `Hello! How can`, exactly matching its fresh
greedy baseline; a shell stdout equality assertion compared the two strings.
Its first multi cycle retained two target-decoded tokens and
stopped at a target miss; its second retained two at the generation limit.
Neither cycle restored or replayed the target prefix. The one-token-capacity
and strict-confidence runs correctly committed ordinary single tokens. The
piped chat run returned a verified two-token batch and emitted `Hello!`.
These are correctness smokes, not tok/s benchmarks.

Phase 0.23 incremental-capture checks run on 2026-07-09:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
baseline=$(./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --nothink -n 4 --temp 0 -p "Hello" 2>/dev/null)
direct=$(DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello" 2>/dev/null)
test "$baseline" = "$direct"
printf 'Hello\nTell me one word\n/exit\n' | \
  DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0
DS4_DSPARK_MULTI_COMMIT=1 DS4_DSPARK_VERIFY_MIN_CONFIDENCE=0.99 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0 -p "Hello"
git diff --check
```

The exact stdout comparison still produced `Hello! How can` for baseline and
DSpark. The two-turn piped chat completed initial capture, a resumed eight-token
sync span, direct multi-commit, and single-token fallbacks without a capture
failure. The strict-confidence smoke still emitted `Hello!` through ordinary
single-token fallback. These are correctness smokes, not tok/s benchmarks.

Phase 0.24 GPU-bridge checks run on 2026-07-10:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_GPU_BRIDGE=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0 -p "Hello"
DS4_DSPARK_GPU_BRIDGE=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
printf 'Hello\nTell me one word\n/exit\n' | \
  DS4_DSPARK_GPU_BRIDGE=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 2 --temp 0
DS4_DSPARK_GPU_BRIDGE=1 DS4_DSPARK_GPU_BRIDGE_TOLERANCE=0.0001 \
DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
git diff --check
```

Fresh prefill, one-token decode spans, and the resumed eight-token chat suffix
all reported exact packed-context parity (`context_max=0`). With the default
`0.005` threshold, every observed normalized `main_x` span passed; the largest
max-absolute difference was `0.00338316` and the largest observed RMS
difference was `0.000551707`. The four-token bridge run emitted exactly
`Hello! How can`, matching a separately captured no-DSpark baseline string.
The forced `0.0001` tolerance reported `result=fail` and incremented mismatch
counters while CPU-authoritative generation still emitted `Hello`. These are
correctness/parity smokes, not tok/s benchmarks.

Phase 0.25 GPU-stage-0 checks run on 2026-07-10:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_GPU_STAGE0=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_GPU_STAGE0=1 DS4_DSPARK_GPU_STAGE0_TOLERANCE=0.001 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_GPU_STAGE0=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
git diff --check
```

Stage 0 mapped three sidecar spans totaling `3.51 GiB`. Across ordinary greedy
prefix lengths 10 through 14, same-input attention relative RMS ranged from
`0.0127689` to `0.0141918`; complete stage relative RMS ranged from
`0.0129634` to `0.0246368`. All CPU and GPU HC values were finite and all five
contexts passed the default `0.03` relative-RMS band. The strict `0.001` run
reported failures and incremented mismatch counters while CPU-authoritative
generation remained unchanged. The stage-0 plus direct multi-commit run passed
all three prepared-prefix checks and emitted exactly `Hello! How can`, matching
the established greedy baseline. The ordinary stage-0-only run emitted the
same string. These are correctness/parity smokes, not tok/s benchmarks.

Phase 0.26 GPU-stage-1 checks run on 2026-07-10:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_GPU_STAGE1=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_GPU_STAGE1=1 DS4_DSPARK_GPU_STAGE1_TOLERANCE=0.001 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_GPU_STAGE1=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
git diff --check
```

Stage 1 mapped three additional sidecar spans totaling `3.51 GiB` and consumed
stage-0 GPU HC through the command-batched device copy. Across ordinary greedy
prefix lengths 10 through 14, stage-1 same-input attention relative RMS ranged
from `0.00749991` to `0.00869288`; complete stage relative RMS ranged from
`0.0115571` to `0.034785`. All CPU and GPU attention/full-stage HC values were
finite and all five contexts passed the stage-1 `0.04` relative-RMS band. The
strict `0.001` override reported failures while CPU-authoritative output stayed
unchanged. Stage-1 plus direct multi-commit passed at prefixes 10, 12, and 14
and emitted exactly `Hello! How can`, matching the established greedy baseline.
The ordinary stage-1 run emitted the same string. These are correctness/parity
smokes, not tok/s benchmarks.

Phase 0.27 GPU-stage-2 checks run on 2026-07-10:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_GPU_STAGE2=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_GPU_STAGE2=1 DS4_DSPARK_GPU_STAGE2_TOLERANCE=0.001 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_GPU_STAGE2=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
git diff --check
```

Stage 2 mapped three additional sidecar spans totaling `3.51 GiB` and consumed
stage-1 GPU HC through the generic command-batched device copy. Across ordinary
greedy prefix lengths 10 through 14, same-input attention relative RMS ranged
from `0.00865846` to `0.00981461`; complete stage relative RMS ranged from
`0.0096529` to `0.0188689`. All CPU and GPU attention/full-stage HC values were
finite and all five contexts passed the stage-2 `0.04` relative-RMS band. The
strict `0.001` override reported failures while CPU-authoritative output stayed
unchanged. Stage-2 plus direct multi-commit passed at prefixes 10, 12, and 14
and emitted exactly `Hello! How can`, matching the established greedy baseline.
The ordinary stage-2 run emitted the same string. These are correctness/parity
smokes, not tok/s benchmarks.

Phase 0.28 GPU-head checks run on 2026-07-10:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_GPU_HEAD=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_GPU_HEAD=1 DS4_DSPARK_GPU_HEAD_TOLERANCE=1e-7 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_GPU_HEAD=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
git diff --check
```

The head mapped three coalesced sidecar spans totaling `0.14 MiB`. Across
ordinary greedy prefix lengths 10 through 14, same-input HC-collapse relative
RMS ranged from `2.02519e-7` to `4.72896e-7`; final-norm relative RMS ranged
from `1.60932e-7` to `2.5731e-7`. Every reference and GPU value was finite and
all contexts passed the calibrated `1e-5` default. The separate
`input_norm_max/input_norm_rms` fields preserve visibility into accumulated
transformer-stage drift without charging it to the same-input head comparison.
The strict `1e-7` override reported failures and incremented mismatch counters
while CPU-authoritative generation still emitted `Hello`. GPU head plus direct
multi-commit passed at prefixes 10, 12, and 14 and emitted exactly
`Hello! How can`. These are correctness/parity smokes, not tok/s benchmarks.

Phase 0.29 GPU-logit checks run on 2026-07-10:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_GPU_LOGITS=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_GPU_LOGITS=1 DS4_DSPARK_GPU_LOGITS_TOLERANCE=0.001 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_GPU_LOGITS=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
git diff --check
```

The two adjacent BF16 Markov tensors coalesced into one mapped span totaling
`126.25 MiB`. Across ordinary greedy prefix lengths 10 through 14, same-input
base-logit relative RMS ranged from `0.00444636` to `0.00495267`, Markov-logit
relative RMS from `4.83308e-8` to `5.14558e-8`, and combined-logit relative RMS
from `0.00133959` to `0.00216699`. Every value was finite. All 25 GPU top ids
matched the same-input CPU references, and all 25 also matched the existing
CPU-authoritative draft tokens. Every context passed the calibrated `0.01`
default. The explicit `0.001` override reported failures while generation still
emitted `Hello`. GPU logits plus direct multi-commit passed at prefixes 10, 12,
and 14 and emitted exactly `Hello! How can`. These are correctness/parity
smokes, not tok/s benchmarks.

Phase 0.30 GPU-confidence checks run on 2026-07-11:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_GPU_CONFIDENCE=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_GPU_CONFIDENCE=1 DS4_DSPARK_GPU_CONFIDENCE_TOLERANCE=1e-7 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_GPU_CONFIDENCE=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
git diff --check
```

The BF16 confidence vector mapped as one `8.50 KiB` sidecar span. Across
ordinary greedy prefix lengths 10 through 14, same-input confidence-logit
relative RMS ranged from `4.82943e-8` to `2.80202e-7`; confidence-probability
max error was at most `1.19209e-7`. Every reference and GPU output was finite,
and all 25 assembled candidate top tokens agreed with the GPU top-1 reduction.
All contexts passed the calibrated `1e-5` default. Upstream GPU-chain drift,
logged separately, changed CPU-reference-versus-authoritative probabilities by
as much as `0.0139291`; this does not indicate a confidence-kernel mismatch but
must be considered before GPU candidate authority. The explicit `1e-7`
override reported failures while CPU-authoritative generation still emitted
`Hello`. GPU confidence plus direct multi-commit passed at prefixes 10, 12, and
14 and emitted exactly `Hello! How can`. These are correctness/parity smokes,
not tok/s benchmarks.

Phase 0.31 independent GPU-chain checks run on 2026-07-11:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_GPU_CHAIN=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
DS4_DSPARK_GPU_CHAIN=1 DS4_DSPARK_GPU_CONFIDENCE_TOLERANCE=1e-7 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_GPU_CHAIN=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
git diff --check
```

Across ordinary greedy prefix lengths 10 through 14, all 25 self-fed GPU rows
matched their exact-predecessor CPU reference, the GPU top-1 reduction, the
earlier CPU-fed GPU candidate chain, and the CPU-authoritative candidate chain.
The component ranges were unchanged from phases 0.29-0.30 because no token
divergence changed the Markov rows: Markov relative RMS remained
`4.83308e-8`-`5.14558e-8`, combined-logit relative RMS
`0.00133959`-`0.00216699`, confidence-logit relative RMS
`4.82943e-8`-`2.80202e-7`, and probability max error at most `1.19209e-7`.
All five contexts passed the inherited defaults. The explicit `1e-7`
confidence override made both confidence and chain checks report failures while
CPU-authoritative generation still emitted `Hello`. Self-fed GPU chain plus
direct multi-commit passed at prefixes 10, 12, and 14 and emitted exactly
`Hello! How can`. These are correctness/parity smokes, not tok/s benchmarks.

Phase 0.32 guarded GPU-candidate checks run on 2026-07-11:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_GPU_CANDIDATES=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_GPU_CANDIDATES=1 DS4_DSPARK_GPU_CONFIDENCE_TOLERANCE=1e-7 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 1 --temp 0 -p "Hello"
DS4_DSPARK_GPU_CANDIDATES=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --nothink -n 4 --temp 0 -p "Hello"
git diff --check
```

The normal one-token run selected complete five-row GPU proposal blocks at both
prepared prefixes and still emitted `Hello`. With the explicit `1e-7`
confidence tolerance, both self-fed blocks completed but failed parity; the
source gate logged `selected=0`, retained CPU proposals twice, and still emitted
`Hello`. In the direct multi-commit run, GPU proposals were selected at prefixes
10, 12, and 14. GPU confidence values were visible in the prepared records
(for example `0.915544` versus the earlier CPU-authoritative `0.917943` at
prefix 10), proving the proposal copy was active. Exact target verification
committed two tokens, stopped at the expected target miss, later committed two
through the accepted limit, and emitted exactly `Hello! How can`. These are
correctness smokes, not tok/s benchmarks.

Phase 0.33 broader GPU-candidate correctness checks run on 2026-07-11:

```sh
bash -n tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

The harness reported:

```text
PASS prompt: reasoning
PASS prompt: italian
PASS prompt: medium_context
PASS chat: resumed two-turn session
PASS fallback: strict parity rejection
DSpark GPU candidate correctness matrix: PASS
```

Every baseline/GPU pair had byte-identical stdout. The medium fixture generated
`brass lens`, entered DSpark at context 119, and continued selecting
parity-passing GPU proposal blocks through contexts 120 and 123. The resumed
chat selected GPU proposals on more than one preparation cycle and remained
byte-identical to baseline. The strict `1e-7` confidence case observed chain
`result=fail`, selected no GPU blocks, logged the expected CPU fallback, and
matched baseline output. This matrix is a correctness integration test, not a
tok/s benchmark.

An untargeted `./ds4_test` run entered its long-context GPU prefill case and was
stopped after about one minute to avoid occupying the accelerator indefinitely.
Do not treat the full test suite as passed for this phase; the focused DSpark
tests and real-sidecar correctness smokes above did pass.

Phase 0.34 device-resident self-fed chain checks run on 2026-07-11:

```sh
make ds4
DS4_DSPARK_GPU_CANDIDATES=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --ctx 4096 --nothink --temp 0 -n 1 -p Hello
DS4_DSPARK_GPU_CANDIDATES=1 DS4_DSPARK_GPU_CHAIN_HOST=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --ctx 4096 --nothink --temp 0 -n 1 -p Hello
./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

The default GPU candidate path now encodes all five dependent Markov rows in
one Metal command buffer. Each row gathers its BF16 Markov embedding from the
previous row's device top-id, then uses the existing parallel BF16 matvec, add,
and top-k kernels. The confidence projection follows in the same batch. This
removes the five top-id host readbacks and command-buffer waits from the
self-fed dependency loop while preserving parallel vocabulary work.

`DS4_DSPARK_GPU_CHAIN_HOST=1` retains the previous host-stepped schedule as a
diagnostic oracle. A direct device/oracle smoke produced byte-identical stdout,
the same proposal ids, and matching parity values across two five-row chain
attempts. Both schedules passed all ten rows and selected GPU proposals. The
full phase 0.33 correctness matrix also passed unchanged on the device schedule,
including resumed chat and forced strict-parity fallback. The focused builds
and DSpark validation tests passed; the CPU build emitted the same eight known
unused-code warnings.

An initial monolithic 512-thread kernel was tested and discarded before this
phase was committed. Although numerically correct, one threadgroup serialized
the five full-vocabulary rows and was clearly unsuitable. Do not restore that
design. The committed ordered-dispatch design preserves the existing parallel
matvec and top-k implementations.

This is still a diagnostic correctness mode, not a benchmarkable production
path: it computes the CPU sidecar proposal block first and performs full GPU
logit/confidence readbacks for parity. Exact target verification remains
authoritative. No tok/s benchmark was run and no performance claim is made.

Phase 0.35 compact proposal-runtime checks run on 2026-07-11:

```sh
make ds4
DS4_DSPARK_GPU_RUNTIME=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --ctx 4096 --nothink --temp 0 -n 1 -p Hello
bash -n tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime ./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

The direct runtime smoke had byte-identical baseline/runtime stdout. It
prepared two five-row blocks, reported `compact_top2=10` and
`compact_confidence=5`, selected GPU proposals, and emitted no proposal-level
logits, confidence, or chain parity records. The default observer matrix passed
reasoning, Italian, context-119, resumed chat, and strict parity fallback. The
runtime matrix passed the same four output/session cases and asserted that no
proposal-level parity observer ran. Every compared stdout pair was byte
identical. All requested binaries and focused DSpark tests passed; the CPU
object emitted the same eight known warnings.

Runtime mode deliberately has no tolerance fallback because it does not run a
CPU parity reference. Its safety boundary is compact structural/numeric
validation followed by the unchanged exact target verifier. The observer mode
remains available for tolerance-driven regression diagnosis. Earlier
bridge/stage/head parity execution is still present in runtime mode, so no tok/s
benchmark was run and no performance claim is made.

Phase 0.36 device-resident sidecar checks run on 2026-07-11:

```sh
make ds4
DS4_DSPARK_GPU_RUNTIME=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --ctx 4096 --nothink --temp 0 -n 1 -p Hello
bash -n tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime ./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

The direct smoke had byte-identical baseline/runtime stdout across two
preparation cycles. The first capture kept rows `0..9` on device and the second
incrementally appended row `10`. Each cycle reported runtime success for the
bridge, stages 0/1/2, HC head, and compact proposal chain, then selected the
five-row GPU block. No DSpark GPU parity record appeared.

The observer matrix again passed reasoning, Italian, context 119, resumed chat,
and strict parity fallback. The runtime matrix passed the four output/session
cases and now requires runtime records from bridge through proposal chain while
rejecting any DSpark GPU parity record. Every baseline/DSpark stdout pair was
byte-identical. All requested binaries and focused DSpark tests passed; the CPU
object emitted the same eight known warnings.

Runtime capture no longer calls `ds4_gpu_tensor_read` for target HC. Runtime
preparation no longer executes CPU `main_proj/main_norm`, transformer stages,
or HC head, and does not allocate their parity buffers on a fresh runtime
session. Common CPU scratch is still allocated by session construction, runtime
logs are still emitted, and the GPU stages still borrow target graph transient
batch tensors. No tok/s benchmark was run and no performance claim is made.

Phase 0.37 quiet-runtime checks run on 2026-07-12:

```sh
make ds4
DS4_DSPARK_GPU_RUNTIME=1 DS4_DSPARK_MULTI_COMMIT=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --ctx 4096 --nothink --temp 0 -n 1 -p Hello
bash -n tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime ./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

The quiet smoke exercised runtime plus direct multi-commit. Its stderr contained
the one-time sidecar validation line but no per-cycle bridge/stage/head/chain,
candidate-source, prepared-draft, verifier, acceptance-simulator, or
multi-commit telemetry. Failures remain ungated. The runtime correctness
harness now sets `DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS=1` explicitly so it can
continue requiring every runtime-stage record.

The observer matrix passed reasoning, Italian, context 119, resumed chat, and
strict parity fallback. The diagnostic runtime matrix passed all four
output/session cases. All baseline/DSpark stdout pairs were byte-identical. All
requested binaries and focused DSpark tests passed; the CPU object emitted the
same eight known warnings.

Synchronization audit: one `ds4_gpu_end_commands()` commits and waits for the
complete five-row compact proposal batch. The subsequent managed-buffer reads
copy 120 bytes total and perform no command submission or wait. This host
boundary is currently required because confidence/margin gating and exact
target verification are CPU-controlled. No tok/s benchmark was run. The next
measurement must be initiated manually by the user on an otherwise idle
machine, with runtime diagnostics unset.

Phase 0.38 benchmark-protocol checks run on 2026-07-12:

```sh
python3 -m py_compile speed-bench/run_dspark_comparison.py
python3 speed-bench/run_dspark_comparison.py --dry-run --allow-dirty
git diff --check
```

Only syntax compilation and dry-run validation were executed. Dry-run resolved
the local binary, base GGUF, DSpark GGUF, and fixed prompt, then printed the
exact baseline and runtime commands. It explicitly reported that runtime
and inherited instrumentation diagnostics are forcibly unset and that no model
execution occurred. Synthetic parser/summary assertions also passed. No tok/s
benchmark was run.

The actual user command after committing this phase is:

```sh
python3 speed-bench/run_dspark_comparison.py --confirm-idle
```

Defaults are one baseline and one runtime warmup, then three alternating
measured pairs with 10-second cooldowns. Runtime sets only
`DS4_DSPARK_GPU_RUNTIME=1` and `DS4_DSPARK_MULTI_COMMIT=1`. Every output must
match the first baseline byte-for-byte. Results are written beneath
`speed-bench/local-runs/<timestamp>/`; inspect `summary.md`, `results.csv`, and
`metadata.json`, while retaining raw files for diagnosis.

Downloaded sidecar byte size:

```text
/Users/deathcodevision/dev/ds4/gguf/ds4flash-dspark.gguf
11489939840 bytes
```

Phase 0.39 runtime-efficiency instrumentation added on 2026-07-12:

- `DS4_DSPARK_GPU_RUNTIME_STATS=1` enables aggregate counters and timers only
  when the GPU runtime is active. It emits one machine-readable stderr record
  when the session closes; no per-cycle stats logging was added.
- Acceptance depth counts the number of tokens returned by each direct
  multi-commit attempt. `target_evals_avoided` is defined as emitted tokens
  minus actual target transformer evaluations, clamped at zero.
- Target timing covers the Metal target decode call and stops before target-HC
  capture completion. Bridge, stages 0/1/2, HC head, and proposal-chain timing
  are separate non-overlapping sidecar buckets and include their GPU waits.
- The comparison harness enables this record for runtime runs, requires exactly
  one record, stores every raw field in `results.csv`, and reports median
  acceptance depth, target evaluations per emitted token, and component time
  normalized per emitted token.

The first user-run comparison before this instrumentation measured 23.94 t/s
baseline versus 21.51 t/s runtime, a 0.8985x ratio of medians across three
pairs. Outputs were byte-identical and paired ratios were consistently near
0.90x despite unavoidable background activity. Inspection of the verifier
showed serial one-token target evaluations for accepted draft suffixes, so the
next user-run comparison should establish whether any target transformer calls
are actually avoided and quantify sidecar cost. Codex must not run that tok/s
comparison. The user accepts that the available machine cannot be made fully
idle; `--confirm-idle` now means best-effort readiness, with process metadata
retained for interpretation.

Phase 0.39 checks:

```sh
make ds4
DS4_DSPARK_GPU_RUNTIME=1 DS4_DSPARK_MULTI_COMMIT=1 \
  DS4_DSPARK_GPU_RUNTIME_STATS=1 ./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --ctx 256 --nothink --temp 0 -n 1 -p Hello
python3 -m py_compile speed-bench/run_dspark_comparison.py
python3 speed-bench/run_dspark_comparison.py --dry-run --allow-dirty
./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime ./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

The one-token correctness smoke emitted exactly one parsable stats record and
reported one emitted token, one target evaluation, and zero target evaluations
avoided. Its first-call component timings include initialization and are not a
performance result. Synthetic stats parser/summary assertions passed. Both GPU
candidate matrices passed, all baseline/runtime outputs were byte-identical,
all requested binaries and focused DSpark tests passed, and the CPU object
emitted the same eight known warnings. No tok/s benchmark was run by Codex.

Phase 0.40 exact batched target verification added on 2026-07-12:

- The pre-change user benchmark at commit `d828f6f` reproduced the earlier
  result: 23.85 t/s baseline versus 21.43 t/s runtime, or 0.8985x by ratio of
  medians. All three paired ratios were near 0.90x.
- Draft acceptance was already excellent: average depth 4.923, with twelve
  depth-5 commits and one depth-4 commit across 64 emitted tokens. The serial
  verifier nevertheless performed 64 target graph calls for those 64 tokens,
  avoiding none.
- `metal_graph_verify_decode_exact` now verifies up to sixteen rows with the
  normal one-token target decode kernels and autoregressive cache-update order,
  while scheduling all rows layer-by-layer in one GPU command stream. DSpark
  currently uses at most five rows.
- The verifier snapshots compressed-attention frontiers before mutation. A
  full accept commits the already-advanced target state directly. A partial
  accept restores the original frontier and reruns only the accepted prefix as
  a second exact batch. Any recoverable snapshot, capture, or verifier failure
  restores state and uses the existing serial path; an unrestoreable target
  state remains a hard error.
- Exact batching is currently selected only for GPU DSpark runtime on resident
  non-streaming graphs. SSD-streaming graphs retain serial verification because
  their layer mapping/readahead contract needs separate work.
- Target-layer HC rows are captured inside the exact verifier command stream,
  then passed through the existing GPU bridge once the accepted target prefix
  is known. This preserves the device-resident DSpark continuation path.
- Runtime stats now distinguish target graph calls from token positions
  processed and report exact-batch attempts/full accepts/partial accepts/
  fallbacks. Sidecar timing is split into prefill and generation component
  buckets; the benchmark summary normalizes only generation-side work per
  emitted token.

Phase 0.40 correctness checks include a ten-token direct smoke whose output was
byte-identical to baseline. It exercised one full and three partial exact-batch
accepts with no batch fallback, using eight target calls for ten emitted tokens.
This is correctness evidence only, not a speed result. The observer and runtime
GPU candidate matrices both passed reasoning, Italian, medium-context, and
resumed-session cases with byte-identical outputs. The runtime matrix now also
requires a diagnostics-only exact-batch success record, so a silent serial
fallback fails correctness validation. All requested binaries and focused
DSpark tests passed; the CPU object emitted the same eight known warnings. The
benchmark parser/dry run and actual-record summary assertions passed. No tok/s
benchmark was run by Codex.

The user benchmark at commit `285931c` measured 23.90 t/s baseline versus
19.54 t/s exact-batch runtime, or 0.8176x by ratio of medians. All three paired
ratios were near 0.82x. Exact batching worked as designed: 14 target calls
processed 69 target positions for 64 emitted tokens, with twelve full accepts,
one depth-4 partial accept, and no fallback. The target verifier cost 47.234 ms
per emitted token and generation-side DSpark cost 3.707 ms/token, closely
accounting for the measured 51.18 ms/token runtime. The exact path reduced host
synchronizations but continued to run one-token decode kernels for every target
position; partial restore/replay added four repeated accepted positions. Real
compute batching is required for a speed win.

Phase 0.41 fast-verifier parity observer added on 2026-07-12:

- `DS4_DSPARK_FAST_VERIFY_OBSERVER=1` is opt-in and requires GPU DSpark runtime.
  It runs the existing `metal_graph_verify_suffix_tops` batch kernels as a
  shadow operation from the exact verifier's saved target frontier, reads row
  tops and final logits, restores the frontier, and then runs the unchanged
  exact verifier as production authority.
- DSpark speculative graph allocation now owns `spec_logits` whenever generic
  speculation is enabled, rather than hiding that shared verifier scratch
  inside the legacy-MTP-only allocation block.
- Observer records include intermediate row-top agreement, final top ids,
  final-logit maximum and relative RMS difference, non-finite counts, fast and
  exact verifier milliseconds, and their local ratio. This option is cleared
  by the user benchmark harness and cannot affect normal runtime measurements.
- The correctness harness accepts
  `DS4_TEST_DSPARK_FAST_VERIFY_OBSERVER=1`, requires an observer record for each
  runtime case, and fails if any record reports parity failure.

Observed evidence covered a ten-token direct partial-accept smoke plus
reasoning, Italian, medium-context, and resumed-session matrix cases. All 21
intermediate row tops and all eight final top tokens matched exact verification;
all user-visible outputs remained byte-identical to baseline. Final full-logit
relative RMS drift ranged from about 0.0059 to 0.0755, with no non-finite values.
Warm five-row observations were approximately 1.49x and 1.61x faster than the
exact verifier, while short/cold observations were noisy and sometimes slower.
This supports an explicitly opt-in fast runtime experiment, not unconditional
promotion. No tok/s benchmark was run by Codex.

Phase 0.41 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_TEST_DSPARK_MODE=runtime \
  DS4_TEST_DSPARK_FAST_VERIFY_OBSERVER=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_gpu_candidates_correctness.sh
./ds4_test --mtp-verify-depth
bash -n tests/dspark_gpu_candidates_correctness.sh
git diff --check
```

All requested builds and DSpark checks passed; the CPU object emitted the same
eight known warnings. The MTP depth test reported success but skipped its model
run because `DS4_TEST_MTP` was not configured. The shared speculation allocation
therefore has compile coverage here but no local legacy-MTP GGUF runtime check.

Phase 0.42 opt-in fast authoritative verification added on 2026-07-12:

- `DS4_DSPARK_FAST_BATCH_VERIFY=1` selects the compute-batched target verifier
  only when GPU DSpark runtime is active. Exact batched verification remains the
  default.
- `metal_graph_verify_suffix_tops` now accepts the normal DSpark target-HC
  capture object and records all accepted-span target layers from its batched
  hidden-state tensor. The existing MTP callers pass no capture and retain their
  previous behavior.
- Full fast accepts commit the already-advanced target state and captured HC.
  Partial accepts restore the original compressed-attention frontier and rerun
  only the accepted prefix through the fast verifier. A recoverable fast failure
  restores state and retries the exact batch verifier; if exact batching also
  fails recoverably, the existing serial verifier remains the final fallback.
- Runtime stats add fast verifier calls, failures, and successful exact
  fallbacks. Diagnostics identify whether a committed batch used `fast` or
  `exact` authority and report fast-to-exact fallback transitions.
- `speed-bench/run_dspark_comparison.py --fast-verifier` explicitly sets the
  fast runtime option after clearing inherited DSpark variables. The default
  harness remains exact. Both modes continue to require byte-identical stdout.
- The correctness harness accepts `DS4_TEST_DSPARK_FAST_VERIFY_RUNTIME=1`,
  requires successful fast-batch records, and retains every existing output and
  resumed-session assertion.

A ten-token direct partial-accept smoke was byte-identical to baseline and used
seven fast target calls with zero failures/fallbacks. It exercised full commits
plus restore-and-fast-prefix replay. The fast runtime matrix passed reasoning,
Italian, medium-context, and resumed-session cases with byte-identical outputs.
These are correctness checks, not throughput measurements. No tok/s benchmark
was run by Codex.

Phase 0.42 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
DS4_TEST_DSPARK_MODE=runtime ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
  DS4_TEST_DSPARK_FAST_VERIFY_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
python3 -m py_compile speed-bench/run_dspark_comparison.py
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --allow-dirty --fast-verifier
bash -n tests/dspark_gpu_candidates_correctness.sh
git diff --check
```

All requested builds, focused DSpark tests, exact runtime cases, and opt-in fast
runtime cases passed. The CPU object emitted the same eight known warnings. A
parser/summary assertion consumed the real ten-token fast stats record and
confirmed seven fast calls, zero failures, and zero exact fallbacks.

The user benchmark at commit `66460a4` produced the first real DSpark speed win:
23.87 t/s baseline versus 29.70 t/s fast runtime, or 1.2442x by ratio of
medians. Paired speedups were tightly grouped from 1.2396x to 1.2463x despite
background activity. All stdout hashes were identical. Thirteen proposal cycles
averaged depth 4.923, with twelve full accepts and one depth-4 partial accept;
14 fast calls had zero failures or exact fallbacks. Target verification cost
29.876 ms per emitted token, generation-side DSpark cost 3.608 ms/token, and
the measured runtime was 33.67 ms/token versus baseline's 41.89 ms/token. This
is approximately 24.4% higher throughput and 19.6% lower steady generation
latency. Prefill/startup remained poor at roughly 3 t/s versus 46 t/s baseline.

Phase 0.43 broadened fast-runtime correctness on 2026-07-12:

- Added `tests/dspark_fast_verifier_soak.sh` plus Spanish and structured-JSON
  prompts. The soak covers the fixed 64-token multi-cycle generation prompt,
  code completion,
  32-token Italian, 24-token Spanish, structured JSON, the existing near-window
  memory prompt, strict margin gating, and a longer resumed two-turn chat.
- Every case compares baseline and fast stdout byte-for-byte. Eligible cases
  require a successful fast-batch commit. All cases reject fast verifier
  failures, target-capture loss, and sidecar-window overflow.
- The code-completion vector emits EOS after one token and therefore validates
  first-row fallback rather than a fast commit.
- `DS4_DSPARK_VERIFY_MIN_MARGIN=100` deliberately rejects the first proposal;
  diagnostics now identify the specific `first row margin` reason, and the soak
  requires that fallback record.
- The first long resumed-chat attempt found a genuine second-turn divergence:
  baseline ended with `Rayleigh scattering makes sky blue.` while approximate
  fast authority produced `Rayleigh scattering scatters blue light.` The first
  turn and every one-shot case remained identical.
- Fast authority is now suspended permanently after a session with a valid
  checkpoint is synchronized again. The first turn may use fast batching; the
  resumed sync emits a suspension record and exact batching becomes
  authoritative. Session invalidation clears the suspension for a new cold
  session.
- The final soak passed all cases and verified the resumed fast-to-exact
  transition with a byte-identical complete transcript. No tok/s benchmark was
  run by Codex.

Phase 0.43 checks:

```sh
./tests/dspark_fast_verifier_soak.sh
DS4_TEST_DSPARK_MODE=runtime \
  DS4_TEST_DSPARK_FAST_VERIFY_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
bash -n tests/dspark_fast_verifier_soak.sh
git diff --check
```

The final soak and established fast runtime matrix passed. All requested builds
and focused tests passed; the CPU object emitted the same eight known warnings.

Phase 0.44 warm-prefill benchmark harness added on 2026-07-12:

- The previous generation harness launches a new process for every sample, so
  its prefill number conflates actual fresh-session prefill with first-process
  Metal pipeline creation, page residency, and DSpark graph initialization.
- `ds4-warm-prefill-bench` opens one engine and renders the same default
  no-thinking chat prompt as the CLI. It records the first fresh session as
  `cold`, optional extra conditioning sessions, and measured `warm` fresh
  sessions. Every session has a new KV cache; only engine/process/GPU state is
  retained. Session creation and destruction are outside the sync timer.
- Every sample records prompt tokens, seconds, t/s, target argmax, and an FNV-1a
  hash over the complete target logits vector. This makes prefill equivalence a
  prerequisite for reporting performance.
- `speed-bench/run_dspark_warm_prefill.py` alternates baseline/runtime child
  order over three process pairs by default. Each child records one cold and
  three warm samples while holding its engine open. Paired ratios use each
  child's median warm throughput; summaries separately report cold and warm
  medians.
- Runtime children set only `DS4_DSPARK_GPU_RUNTIME=1` and
  `DS4_DSPARK_MULTI_COMMIT=1`. The runner clears inherited DSpark and timing
  instrumentation variables; runtime stats and diagnostics remain disabled.
- A real run requires explicit `--confirm-ready`. It captures git, hardware,
  thermal, process, environment, command, and model metadata plus all raw
  stdout/stderr under `speed-bench/local-runs/warm-prefill-<timestamp>/`.
- Codex did not execute a real-model or tok/s benchmark. The user owns the timed
  run after the phase is committed.

Phase 0.44 implementation-only checks:

```sh
make ds4-warm-prefill-bench
./ds4-warm-prefill-bench --help
python3 -m py_compile speed-bench/run_dspark_warm_prefill.py
python3 speed-bench/run_dspark_warm_prefill.py --dry-run --allow-dirty
# Inline synthetic CSV assertions also cover parsing and paired summary math.
git diff --check
```

Phase 0.45 consolidated Metal sidecar residency added on 2026-07-12:

- The first warm-prefill run showed stable warmed fresh-session throughput of
  50.47 t/s baseline versus 49.28 t/s runtime (0.9764x), but first-sync runtime
  throughput was 2.88 t/s versus 47.59 t/s. All target-logit hashes matched.
- Retained stderr identified the cause: lazy DSpark bridge, stage, head, logits,
  and confidence setup repeatedly appended a sidecar range, destroyed the
  current global Metal residency set, and requested residency for every
  accumulated view. A runtime process ended with 17 overlapping shared buffers
  after many sequential 0.8-1.0 second requests.
- Active Metal DSpark runtime now maps the complete 10.7 GiB sidecar tensor-data
  range once during engine setup, mirroring the established MTP sidecar setup.
  DSpark remains opt-in, and CPU/CUDA behavior is unchanged. Lazy component map
  checks remain in place but return immediately because the full range is
  already covered.
- Retained correctness logs show exactly two mapping/residency events per
  process: the base model ends with two shared views, then the complete sidecar
  adds one view and ends with three. No mapping event occurs during bridge,
  stage, head, or chain execution.
- The warm-prefill summary now includes full child wall time and child wall
  minus all sync durations. These fields distinguish a real reduction in total
  startup from work merely moved outside the prefill timer.
- Codex did not run a tok/s benchmark. A user rerun is required before making a
  performance claim about the consolidation.

Phase 0.45 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
DS4_TEST_KEEP_LOGS=1 DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
  DS4_TEST_DSPARK_FAST_VERIFY_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
python3 -m py_compile speed-bench/run_dspark_warm_prefill.py
python3 speed-bench/run_dspark_warm_prefill.py --dry-run --allow-dirty
git diff --check
```

The user reran the warm-prefill benchmark at commit `c8bc495`. Consolidated
residency reduced runtime first-sync time from roughly 13.53 seconds to 0.85
seconds and runtime child wall time from 16.83 seconds to 5.00 seconds. Baseline
child wall was 4.03 seconds. Warm prefill remained stable at 50.55 t/s baseline
versus 49.39 t/s runtime (0.9770x), while cold sync reached 47.31 versus 45.64
t/s (0.9646x). Runtime non-sync overhead increased by only about 0.90 seconds,
matching the single consolidated sidecar residency request; the old repeated
first-sync residency cost was removed rather than merely moved across the
timer. All logits hashes matched.

Phase 0.46 rolling DSpark window added on 2026-07-12:

- The DeepSeek DSpark paper explicitly specifies sliding-window attention of
  128 for the V4 DSpark backbone. The official V4-Flash-DSpark config combines
  `sliding_window=128` with a 1,048,576-token maximum target context. The prior
  behavior that skipped DSpark after a 128-token total prefix was therefore a
  local implementation limitation, not an architectural restriction.
- Target-layer HC capture now intersects each evaluated target span with the
  final 128-position window. A long chunked prefill copies only the relevant
  source rows into bounded capture storage instead of allocating or retaining
  the complete target prefix.
- DSpark session state tracks absolute window origins separately from compact
  row counts. Advancing a full window drops expired `main_x` rows, shifts the
  retained GPU prefix through scratch storage, and appends newly projected
  rows only after target verification commits.
- CPU and Metal sidecar attention now use absolute RoPE positions while storing
  context and block KV in compact window-relative slots. Speculative restore
  paths do not mutate the durable window before a successful capture finish.
- Added deterministic 187-token factual and 151-token sustained-generation
  rendered prompt cases. Observer parity, exact runtime, and fast runtime all
  remain byte-identical to baseline. The sustained case generates 64 tokens
  and requires at least five successful fast commits while the window rotates.
- A separate integration check used the retained issue-468 `code_4k` fixture.
  Its local rendered prompt reached 1,917 tokens, initialized window
  `1789:1917`, then advanced through full and partial fast commits with
  byte-identical output and no capture skip or fallback.
- No tok/s benchmark was run by Codex. The next phase can import the exact 8k
  comparison corpus and build a paired user-run benchmark matching the retained
  issue-468 workload.

Phase 0.46 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
bash -n tests/dspark_gpu_candidates_correctness.sh \
  tests/dspark_fast_verifier_soak.sh
DS4_TEST_DSPARK_MODE=observer ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_fast_verifier_soak.sh
git diff --check
```

Phase 0.47 issue-468 comparison harness added on 2026-07-12:

- Imported the exact `code_8k`, `synthesis_8k`, and `grounded_8k` fixtures from
  lobanov/ds4 branch `dspark-research/issue468` at commit
  `8a4675edeae52c65729cadecb378670b83057067`. Their byte sizes and SHA-256
  hashes are pinned in `speed-bench/issue468/provenance.json`; the runner
  refuses changed corpus files.
- Froze the published longer-prompt MTP table in `mtp_reference.json`. This
  prevents a future upstream branch update from silently changing our
  comparison baseline.
- Added `run_dspark_issue468_comparison.py`. It matches the source workload at
  `ctx=16384`, `n=128`, temperature zero, seed one, and intentionally does not
  pass `--nothink`, matching the retained upstream harness.
- The authoritative throughput pass clears inherited DSpark and instrumentation
  variables, enables GPU runtime, multi-commit, and the fast verifier, and does
  not enable runtime stats. It uses one warmup per mode per prompt and three
  alternating measured pairs with a ten-second cooldown by default.
- Output identity is enforced separately for each prompt across warmup,
  baseline, runtime, and optional stats runs. A mismatch or unexpected stats
  record aborts the benchmark while retaining raw files.
- `--stats-pass` adds one instrumented runtime run per prompt only after all
  throughput samples. Its diagnostics are written to separate CSV/JSON fields
  and never enter throughput medians.
- Reports include per-prompt and aggregate paired ratios plus percentage-point
  comparisons to published MTP K=2 and K=5 deltas. Absolute t/s across machines
  is explicitly non-comparable, and MTP versus DSpark is not presented as a
  controlled head-to-head benchmark.
- Real execution requires user confirmation with `--confirm-ready`. Codex did
  not run the model or any tok/s benchmark in this phase.

Phase 0.47 implementation-only checks:

```sh
python3 -m py_compile speed-bench/run_dspark_issue468_comparison.py
python3 speed-bench/run_dspark_issue468_comparison.py --dry-run --allow-dirty
# Inline synthetic assertions cover paired summary math, reference deltas,
# complete stats parsing, and no-stats versus stats-pass environments.
shasum -a 256 speed-bench/issue468/*_8k.txt
git diff --check
```

Phase 0.48 long-corpus fast-verifier correction on 2026-07-12:

- The user's first issue-468 run stopped during the `code_8k` warmup because
  fast DSpark output differed from baseline. This was a real generated-text
  difference, not stdout formatting: the streams shared their opening and then
  chose different continuations.
- A correctness-only replay with exact DSpark verification produced the exact
  same 618-byte stdout and SHA-256
  `895ccde85426c18f46a75c62c8c933e9e02ce81feb29436843d9563635785ee0`
  as baseline. This clears the rolling sidecar window and exact verifier for
  this case and isolates the failure to fast target verification.
- The fast shadow observer, with exact verification still authoritative,
  passed its first five-token proposal (`4/4` intermediate row tops and final
  top matched). Its second three-token proposal failed with only `1/2`
  intermediate row tops matching, while the final top still matched. Fast
  authority therefore accepted a suffix token that exact greedy verification
  rejected.
- This is consistent with the fast verifier's implementation: it reuses the
  generic prefill batch kernels, whose source contract explicitly does not
  promise numerical equivalence to autoregressive decode. Prior short-context
  parity was useful evidence but not a correctness proof.
- The issue-468 harness now defaults to exact verification. Experimental fast
  authority requires explicit `--fast-verifier` and is documented as invalid
  for performance reporting on this corpus. We must not report the aborted
  fast warmup's timing as a DSpark result.
- Next engineering work should make the compute-batched target verifier
  numerically authoritative or introduce a genuinely correctness-preserving
  fallback criterion. Merely extending the prompt soak cannot make an
  approximate verifier safe.

Phase 0.48 checks:

```sh
# Correctness-only model replay; timings were ignored.
# Exact DSpark stdout matched the retained code_8k baseline byte-for-byte.
# Fast observer reported pass on proposal one and row-top failure on proposal two.
python3 -m py_compile speed-bench/run_dspark_issue468_comparison.py
python3 speed-bench/run_dspark_issue468_comparison.py --dry-run --allow-dirty
python3 speed-bench/run_dspark_issue468_comparison.py \
  --dry-run --allow-dirty --fast-verifier
# Inline assertions verify exact-default and explicit-fast environments.
git diff --check
```

Phase 0.49 long-context fast-verifier localization added on 2026-07-12:

- Added proposal-scoped Metal dump labels for the fast shadow pass and exact
  authoritative pass. `DS4_DSPARK_FAST_VERIFY_LAYER_TRACE=1` suppresses
  unrelated prefill dumps and writes `fast-p<start>` / `exact-p<start>` files
  through the existing `DS4_METAL_GRAPH_DUMP_*` filters. This instrumentation
  is observer-only and inactive by default.
- Observer records now include the absolute proposal start and minimum fast
  target top-2 margin. Failed parity prints the exact row, fast top, exact top,
  and fast margin. Top-2 values are gathered on GPU using existing DSpark
  scratch; no full intermediate vocab rows are copied to the host.
- Layer traces on `code_8k` localized the first large divergence to target layer
  2, the first ratio-4 compressed-attention layer. Layers 0-1 agreed near
  `1e-6` relative RMS; layer 2 jumped to roughly 2-3 percent, already at the
  attention output before its FFN.
- The batch path switched ratio-4 attention to sparse top-512 as soon as it had
  more than 512 compressed rows, while exact decode intentionally remained
  dense until 1,024. Both paths now share
  `metal_graph_ratio4_decode_sparse_ready`, including the index-row condition.
- A five-row proposal can straddle the 1,024-row transition. The vectorized
  branch previously applied the final row's sparse policy to every earlier
  row. Such proposals now use the existing per-row attention loop, preserving
  each row's dense/sparse visibility.
- Those semantic fixes reduced layer-2 attention drift at the 4,096-token
  boundary from about 10-14 percent relative RMS to roughly `1e-5`. Above the
  boundary, fast and exact selected the same 512 compressed rows in the same
  order for every traced row.
- Full `code_8k` shadow observation still found two near-tie row-top flips at
  starts 4096 and 4109. Their fast target margins were 0.129288 and 0.0221252.
  Exact authority remained byte-identical to baseline.
- A temporary `0.2` fast-target-margin fallback made `code_8k` byte-identical,
  using ten exact fallbacks across 38 batch attempts. It was removed after
  `synthesis_8k` still diverged. On synthesis, all 33 shadow proposals passed
  row-top and final-top parity when each started from exact state, proving that
  repeated fast commits accumulate approximate target/cache state even when
  every isolated proposal appears safe.
- Disabling batch-only F16 shortcuts did not change the mismatches or drift and
  was also removed. The remaining problem is architectural: a target verifier
  must preserve exact autoregressive state. Margin heuristics, more soak cases,
  or occasional re-anchoring cannot establish that contract.
- Fast authority remains explicit experimental research and is invalid for
  issue-468 throughput reporting. The next implementation phase should build
  exact compute-batched decode kernels, preserving one-token cache update order
  and arithmetic while amortizing weight access. Exact replay after every fast
  proposal would restore correctness but likely erase the measured speed gain.
- All model executions in this phase were correctness diagnostics. Their timing
  lines were ignored; Codex did not run or report a tok/s benchmark.

Phase 0.49 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
DS4_TEST_DSPARK_MODE=runtime \
  DS4_TEST_DSPARK_FAST_VERIFY_OBSERVER=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_fast_verifier_soak.sh
# Correctness-only issue-468 model replays covered full code and synthesis
# streams; exact authority matched baseline. Timings were ignored.
python3 -m py_compile speed-bench/run_dspark_issue468_comparison.py
python3 speed-bench/run_dspark_issue468_comparison.py --dry-run --allow-dirty
git diff --check
```

Phase 0.50 exact output-head microbatch added on 2026-07-12:

- Chose the output head as the first exact-state microbatch slice. The exact
  verifier already holds every proposal row's authoritative final hidden state
  contiguously in one of its two batch HC buffers; the active buffer is selected
  from target-layer parity. Output-head scratch and `spec_logits` do not overlap
  target KV/cache state.
- Refactored `metal_graph_encode_output_head_batch` to accept an explicit HC
  input tensor. The unsafe full fast verifier retains its existing batch HC
  input, while exact-head observation and runtime use the authoritative exact
  hidden-state buffer.
- Added `DS4_DSPARK_EXACT_HEAD_BATCH_OBSERVER=1`. It runs a shadow batched head
  over exact hidden states, then leaves the existing serial heads authoritative.
  Each proposal reports intermediate row-top parity, final-top parity, and
  final-logit maximum/RMS drift.
- Full `code_8k`, `synthesis_8k`, and `grounded_8k` correctness runs produced
  66, 52, and 58 observer records respectively. All 176 records matched every
  intermediate row top and final top. Complete stdout remained byte-identical
  to baseline for all three prompt families.
- On `code_8k`, final-logit maximum drift had median `1.14441e-5` and maximum
  `2.09808e-5`; RMS drift stayed around `1e-6`. Replacing batched row norms with
  serial one-row norm kernels did not change these values, proving the small
  difference originates elsewhere in batched head arithmetic; that experiment
  was removed.
- Added opt-in `DS4_DSPARK_EXACT_HEAD_BATCH=1`. It batches only the `n-1`
  intermediate heads used for draft acceptance. The final row always runs the
  original serial output head, so continuation logits remain exact. All target
  layers, compressed/raw caches, target HC capture, and final continuation state
  retain the exact verifier path. A batch-head setup failure falls back to all
  serial heads.
- Added `--exact-head-batch` to the issue-468 runner and metadata. It is mutually
  exclusive with the unsafe `--fast-verifier` experiment. Inherited values are
  cleared with other `DS4_DSPARK_*` variables.
- This is a component milestone, not a full exact compute-batched verifier and
  not a speed claim. No tok/s benchmark was run by Codex. The user can later
  compare exact baseline versus exact-head batching before we decide whether the
  next microbatch target should be HC/QKV projection, FFN projection, or another
  weight-heavy row-independent stage.

Phase 0.50 checks:

```sh
# Correctness-only full code/synthesis/grounded runs used:
DS4_DSPARK_GPU_RUNTIME=1 DS4_DSPARK_MULTI_COMMIT=1 \
DS4_DSPARK_EXACT_HEAD_BATCH=1 \
DS4_DSPARK_EXACT_HEAD_BATCH_OBSERVER=1 ./ds4 ...
# All outputs matched their baseline byte-for-byte; timings were ignored.
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_EXACT_HEAD_BATCH=1 \
DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS=1 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
# Retained logs contained successful exact-head batches for all five cases.
./tests/dspark_fast_verifier_soak.sh
python3 -m py_compile speed-bench/run_dspark_issue468_comparison.py
python3 speed-bench/run_dspark_issue468_comparison.py \
  --dry-run --allow-dirty --exact-head-batch
# The harness rejects --exact-head-batch combined with --fast-verifier.
git diff --check
```

Phase 0.51 exact-head direct ablation prepared on 2026-07-13:

- The user ran the issue-468 suite with `--exact-head-batch`; the unsafe fast
  verifier was disabled. Median paired ratios were `0.4322x` for `code_8k`,
  `0.5160x` for `synthesis_8k`, and `0.4762x` for `grounded_8k`, with an
  aggregate `0.4762x`. All paired output hashes matched and the measurements
  were stable. This reconfirmed that the authoritative exact verifier is much
  slower than baseline on these workloads.
- That run was broader than necessary for the question at hand: it compared
  exact-head DSpark with no DSpark, but did not include ordinary exact DSpark as
  a control. It therefore cannot tell us whether batching the intermediate
  output heads helps. A direct short ablation should have come first.
- Added stats-only exact verifier components: target-layer time, batched-head
  time, serial-head time, and exact-head batch attempts/successes. Clock reads
  and accumulation occur only under `DS4_DSPARK_GPU_RUNTIME_STATS=1`; normal
  uninstrumented throughput behavior is unchanged.
- Added `speed-bench/run_dspark_exact_head_ablation.py`. It directly compares
  ordinary exact DSpark with exact-head DSpark, defaults to one 64-token
  `code_8k` pair, alternates order across additional pairs, requires
  byte-identical output, and reports target component milliseconds per emitted
  token. Both modes are instrumented, so reported t/s is context only.
- Real execution requires `--confirm-ready`. Codex did not run this ablation;
  the user remains responsible for all timed benchmark runs.

Phase 0.51 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
DS4_DSPARK_EXACT_HEAD_BATCH=1 \
DS4_DSPARK_GPU_RUNTIME_STATS=1 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
# All five correctness cases matched baseline. Timings were ignored; real
# stats records reported successful exact-head batches.
./tests/dspark_fast_verifier_soak.sh
python3 -m py_compile speed-bench/run_dspark_exact_head_ablation.py
python3 speed-bench/run_dspark_exact_head_ablation.py \
  --dry-run --allow-dirty
git diff --check
```

Phase 0.51 user-run result on 2026-07-13:

- The direct `code_8k` ablation completed one 64-token pair with identical
  output hashes, identical acceptance statistics (`avg_depth=2.666667`, 35
  target evaluations, 119 target positions), and 26/26 successful exact-head
  batches.
- Ordinary exact target verification cost `87.829 ms/emitted`: `80.187` in
  target layers, `2.773` in serial output heads, and `4.870` residual.
  Exact-head cost `87.089 ms/emitted`: `80.530` in target layers, `0.882` in
  batched intermediate heads, `0.763` in the required serial continuation
  heads, and `4.915` residual.
- The head component therefore fell from `2.773` to `1.645 ms/emitted`, a
  `40.7%` local reduction. Whole-target time fell by `0.740 ms/emitted` or
  `0.84%` (`0.9916x`). The component result shows that head batching works, but
  target-layer work is about `91%` of exact verification and dominates the
  remaining cost.
- Because this was one ordered pair without a warmup, the sub-1% whole-target
  delta should not be treated as a precise performance claim. The unchanged
  workload counters and successful component timing are enough to conclude
  that repeating the full issue-468 suite would not change the engineering
  decision: retain the correct head microbatch, then move to numerically exact
  batching inside the target layers.
- Raw results:
  `speed-bench/local-runs/head-ablation-20260713-002713/results.csv` (ignored by
  git and local to the user's machine).

Phase 0.52 exact target-layer profiling prepared on 2026-07-13:

- Source inspection confirmed that exact verification is layer-major but calls
  the complete one-token decode layer once per proposal row. The existing
  prefill batch path already has separate attention and FFN halves, but those
  kernels are not numerically authoritative: earlier fast-verifier work proved
  that small batch-state differences accumulate across cycles.
- The natural first implementation boundary is after attention HC expansion:
  `after_attn_hc` is the complete per-row input to the FFN, and the graph owns
  batch storage plus an existing batched FFN implementation. Promoting that path
  immediately would still require splitting the monolithic decode function and
  proving hidden-state parity, so first identify whether FFN is actually the
  dominant half and which projection deserves the refactor.
- Added `speed-bench/run_dspark_exact_layer_profile.py`. It runs an unprofiled
  exact reference, then profiles the first, middle, and last exact decode layer
  by default. It clears inherited DSpark/instrumentation flags, never enables
  the fast verifier, and requires every profiled stdout stream to match the
  exact reference byte-for-byte.
- The harness parses the existing `DS4_METAL_DECODE_STAGE_PROFILE` records into
  attention/FFN shares and the five largest stages per representative layer.
  That profiler synchronizes at every stage boundary and disables some overlap,
  so the values are diagnostic ordering evidence only. They are not production
  timings and must not be compared with t/s results.
- Added a profiler-only command-buffer fence before the selected layer in the
  exact verifier. Without it, that layer's first `attn_hc_pre` boundary also
  waited for queued work from all preceding layers. The fence is guarded by the
  existing decode-stage profile option and adds no normal-runtime GPU command
  or synchronization work.
- Real execution requires `--confirm-ready`; Codex did not run the profile.

Phase 0.52 harness correction on 2026-07-13:

- The first user run completed the reference and Flash layers 0 and 30, then
  failed cleanly because the original default also requested layer 60. That was
  a harness bug: DeepSeek V4 Flash has 43 layers (`0..42`), while layer 60 is
  valid only for the 61-layer Pro shape. All three completed stdout files had
  the same hash and their raw logs remain usable.
- Replaced the hard-coded layer range with `ds4 --inspect` discovery before any
  model process starts. Defaults are now first/middle/last (`0,21,42` for
  Flash), and explicit out-of-range layers are rejected immediately.
- Added `--resume-dir`. It validates the retained reference command, prompt
  hash, context, and token count, reuses matching reference/profile files, and
  runs only missing requested layers. The failed run can therefore be completed
  with explicit layers `0,30,42` while preserving its useful layer 0/30 work.
- Correction validation discovered `layers: 43`, derived defaults `0,21,42`,
  rejected explicit layer 60 before model execution, and accepted the retained
  reference plus 690 records each from layers 0 and 30. Layer 42 was correctly
  identified as the only missing requested profile. No timed recovery run was
  performed by Codex.

Phase 0.52 user-run result and robust aggregation on 2026-07-13:

- The resumed profile reused layers 0/30 and collected layer 42. All three had
  46 profiled rows and output remained identical to the retained exact
  reference.
- The first report averaged stage totals. Four `attn_hc_pre` samples at layers
  30/42 took `27.976..45.579 ms` while the stage median remained only
  `0.303..0.313 ms`. Those synchronization/residency stalls inflated the mean
  report to apparent attention shares of `74.2%` and `78.3%`; they are not
  representative HC-pre compute.
- Changed aggregation to sum each stage's median and report any stage maximum
  above five times its median separately. The retained CSV needs no rerun.
  Typical synchronized splits are layer 0: `3.592 ms/row`, attention `1.906`
  (`53.1%`), FFN `1.685` (`46.9%`); layer 30: `4.115 ms/row`, attention
  `2.423` (`58.9%`), FFN `1.692` (`41.1%`); and layer 42: `4.104 ms/row`,
  attention `2.414` (`58.8%`), FFN `1.690` (`41.2%`).
- No single synchronized stage dominates. Median leaders are attention
  (`0.521..0.532 ms`) on compressed layers, routed MoE (`0.479..0.483 ms`), Q
  path (`0.462..0.470 ms`), and attention output (`0.419..0.424 ms`).
- The FFN remains the best first shadow boundary even though attention is the
  larger half: it represents about 41% of late-layer work, is row-independent
  once authoritative `after_attn_hc` states exist, and already has a batched
  implementation. The next phase should compare that batched FFN's final HC
  against serial exact FFN at selected layers while leaving serial output
  authoritative.

Phase 0.53 exact FFN batch shadow observer added on 2026-07-13:

- Split `metal_graph_encode_decode_layer` internally at the existing
  attention/FFN boundary. The original full-layer function remains a wrapper
  over the same code, while attention-only and FFN-only wrappers are available
  for exact-verifier experiments. The ordinary runtime path and all existing
  callers retain full serial decode semantics.
- Added `DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=<layer>`. For proposal
  batches wider than one at that layer, exact one-token attention writes every
  authoritative `after_attn_hc` row into batch storage. The existing prefill
  FFN then runs as a shadow over those exact inputs and its final HC is read
  before serial one-token FFN overwrites the same destination.
- Serial FFN remains authoritative for target capture, subsequent layers,
  caches, logits, and committed state. A shadow setup/encode/read failure logs
  `result=shadow-fallback` and still proceeds through serial FFN. The observer
  reports bit-exact rows plus aggregate HC maximum and RMS drift; it makes no
  performance claim.
- Correctness matrices at layers 0, 30, and 42 each covered five prompt/chat
  cases and six observer records (19 proposal rows). Every generated stream
  matched its baseline. No shadow row was bit-exact. Layer 0 maximum drift was
  `1.19209e-6` with maximum RMS `6.5744e-8`; layer 30 maximum drift was
  `2.86102e-6` with maximum RMS `2.86022e-7`; layer 42 maximum drift was
  `4.57764e-5` with maximum RMS `7.60317e-7`.
- A correctness-only `code_8k` replay at layer 42 compared ordinary exact
  DSpark with the observer for 32 generated tokens. Outputs were byte-identical
  and timings were ignored. Nine records covered 41 proposal rows, with no
  bit-exact rows, maximum HC drift `9.15527e-5`, median RMS `1.13459e-6`, and
  maximum RMS `1.56805e-6`.
- The existing batched prefill FFN must not become authoritative wholesale.
  Its small differences grow materially at late layers even from exact
  attention inputs. The next phase should shadow intermediate FFN boundaries
  (HC pre/norm, router, routed MoE, shared expert, HC post) to find the first
  divergent component, then replace or constrain only that component with
  serial-equivalent arithmetic.
- Strengthened `tests/dspark_gpu_candidates_correctness.sh`: when the observer
  layer env is present, every GPU case must contain an exact/drift comparison
  and no shadow fallback.

Phase 0.53 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
# CPU-only compilation retained the same eight known unused warnings.
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
# A separate code_8k exact-versus-observer replay used -n 32 and cmp; output
# was identical and timings were ignored.
./tests/dspark_fast_verifier_soak.sh
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.54 FFN drift localized to the HC projection on 2026-07-13:

- Expanded the selected-layer FFN observer to capture both shadow-batch and
  serial-exact intermediates without changing stage scheduling. Shadow tensors
  are read first; during serial FFN, each row's one-token scratch is copied into
  the now-disposable batch buffers before the next row overwrites it.
- Compared the normalized HC input (`flat_hc`), HC mixing projection (`hc_mix`),
  recombined FFN input (`ffn_cur`), normalized FFN activation, router IDs and
  weights, and final HC. Reports now name the first divergent boundary.
- Across correctness matrices at layers 0, 30, and 42, all 18 observations
  reported `first=hc_projection`. The normalized HC input was bit-exact in
  every row. Maximum `hc_mix` drift was `9.15527e-5`, `9.05991e-6`, and
  `3.43323e-5` respectively. All 342 selected expert IDs matched. Final HC
  maxima remained `1.19209e-6`, `2.86102e-6`, and `4.57764e-5`.
- A correctness-only layer-42 `code_8k` replay covered nine observations and 41
  proposal rows. Output remained byte-identical to the retained ordinary exact
  reference and timings were ignored. Input normalization remained bit-exact;
  `hc_mix` max was `3.05176e-5`, all 246 expert IDs matched, and final HC max
  was `9.15527e-5`.
- Source inspection explains the boundary. One-token decode dispatches
  `kernel_mul_mv_f16_f32_4`; batches of 2-5 enter the `n_tok <= 8` extended
  multi-vector kernel, whose reduction order differs. The kernel arguments and
  decode matvec already support a row index (`tgpig.y`, `nb11`, `ne1`). The next
  implementation should expose an exact F16 multi-row matvec that dispatches
  the original decode kernel over `grid.y = n_tokens`, preserving each row's
  reduction order while sharing one dispatch.
- This phase only enriches the opt-in shadow observer. Serial FFN remains
  authoritative and no throughput benchmark or runtime speed claim was made.
  The correctness script now requires a recognized `first=` boundary in every
  observer record and still rejects shadow fallback.

Phase 0.54 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
# CPU-only compilation retained the same eight known unused warnings.
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
# A separate code_8k layer-42 observer replay used -n 32 and cmp; output was
# identical and timings were ignored.
./ds4_test --dspark-validation --dspark-shape-binding
bash -n tests/dspark_gpu_candidates_correctness.sh
git diff --check
```

Phase 0.55 exact Metal F16 decode-row projection added on 2026-07-13:

- Added Apple/Metal-only `ds4_gpu_matmul_f16_decode_rows_tensor`. It dispatches
  the same `kernel_mul_mv_f16_f32[_4]` pipeline and reduction parameters used by
  one-token decode, but sets the activation row count in the existing matvec
  arguments and dispatches `grid.y = n_tokens`. Each row therefore retains the
  decode reduction order while 2-5 proposal rows share one Metal dispatch.
- Added an explicit `exact_hc_projection` argument to the internal batched FFN.
  Every production, prefill, and DSpark-stage caller passes `false`; only the
  selected-layer FFN shadow observer passes `true`. On non-Apple builds that
  research selection returns shadow fallback without referencing the Metal
  primitive.
- CUDA and ROCm implementations were briefly considered solely because the GPU
  API header is shared. The user clarified that this work targets Metal only;
  those edits were removed, the declaration is guarded by `__APPLE__`, and no
  CUDA/ROCm source change remains.
- Correctness matrices at layers 0, 30, and 42 each produced six observations
  over 19 proposal rows. Input normalization, `hc_mix`, HC recombination, and
  FFN norm became bit-exact in all observations. The first remaining divergence
  moved to `router_weights`; all 342 expert IDs still matched. Maximum router
  weight drift was `1.19209e-7`, `1.49012e-7`, and `2.08616e-7`; final HC maxima
  were `5.96046e-7`, `1.90735e-6`, and `3.05176e-5`.
- A correctness-only layer-42 `code_8k` replay covered nine observations and 41
  proposal rows. Output matched the retained exact reference byte-for-byte and
  timings were ignored. HC projection through FFN norm remained bit-exact, all
  246 expert IDs matched, router-weight max was `1.19209e-7`, and final HC max
  was `6.10352e-5`.
- This proves the exact-row Metal primitive fixes the first FFN divergence, but
  it remains observer-only and is not yet a runtime speed claim. The next phase
  should capture router logits/probabilities or route the F16 router projection
  through the same exact-row primitive to distinguish projection drift from
  batch softmax/top-k arithmetic.

Phase 0.55 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
# CPU-only compilation retained the same eight known unused warnings.
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
# A separate code_8k layer-42 observer replay used -n 32 and cmp; output was
# identical and timings were ignored.
./tests/dspark_fast_verifier_soak.sh
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.56 exact Metal router rows added on 2026-07-13:

- Extended the FFN shadow observation through router logits and router
  probabilities. Before changing execution, all six layer-42 observations
  localized the first divergence to the F16 router projection: logits differed
  by roughly `1.9e-6` to `2.38e-6`, while expert IDs still matched.
- Reused the Apple/Metal-only exact F16 decode-row matvec for the router
  projection. This made router logits and probabilities bit-exact and moved the
  first difference to selected router-weight normalization.
- Added a separate `kernel_dsv4_router_weights_decode_rows` across `grid.y`
  rows. Each row gathers six probabilities, sums them left-to-right, clamps the
  denominator, and applies the `1.5` scale exactly as one-token decode does.
  The observer's exact mode overwrites the generic batch weights with this
  row-wise result; normal prefill and runtime batch callers remain unchanged.
  An initial attempt generalized `kernel_dsv4_router_weights_one` itself, but a
  reproducible resumed-chat soak mismatch showed that even its signature/grid
  change could disturb serial behavior. The original one-token kernel was
  restored byte-for-byte, and the distinct row kernel passed both the resumed
  fast-to-exact transition and the layer-42 correctness matrix.
  **Phase 0.57 correction:** that final matrix checked generated output but did
  not retain/assert internal metrics. Later capture showed the separately
  compiled row kernel still differed from the original by a few ulps. Phase
  0.57 removed it and dispatches the unchanged one-token kernel at row offsets
  instead. Do not restore the separate kernel based on this older conclusion.
- Correctness matrices at layers 0, 30, and 42 made input normalization, HC
  projection/recombination, FFN norm, router logits/probabilities, all selected
  IDs, and all selected weights bit-exact. The first remaining divergence is
  therefore `experts_or_hc_post`. Final HC max drift ranged from about
  `2.98e-7` at layer 0 to `2.29e-5` at layer 42 in the short matrix.
- A correctness-only layer-42 `code_8k` replay kept all observed router stages
  bit-exact and moved the first difference to `experts_or_hc_post`; timings were
  ignored. Its generated text differed from the older Phase 0.55 reference,
  but a same-binary run with the observer disabled was byte-identical to the
  observer run. The observer does not alter authoritative output; the retained
  older file predates another accepted output change and is no longer a valid
  cross-commit oracle.
- This remains Metal-only observer work. No CUDA or ROCm implementation was
  added, and no production batch caller enables either exact-row option.

Phase 0.56 checks:

```sh
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
# A separate code_8k layer-42 observer replay used -n 32. Its output was
# byte-identical to a current same-binary no-observer control; timings ignored.
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./tests/dspark_fast_verifier_soak.sh
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.57 expert-output boundary capture added on 2026-07-13:

- Extended the exact FFN observer with the actual batch/serial shared-expert
  output, routed-expert output, their CPU-derived combined FFN vector, and the
  existing final HC output. The optimized batch shared-down path writes F16 to
  `batch_q_half`, so the observer records that real buffer and converts it to
  F32 for comparison instead of reading stale `batch_shared_out` data.
- The new capture exposed the Phase 0.56 row-kernel issue above: router weights
  were again off by `2.98e-8` to `5.96e-8`. The separate Metal kernel was
  removed. `ds4_gpu_dsv4_router_weights_decode_rows_tensor` now dispatches the
  original `kernel_dsv4_router_weights_one` once per row at row-specific buffer
  offsets inside one command encoder. Router weights became bit-exact again
  without changing serial decode.
- Strengthened `dspark_gpu_candidates_correctness.sh` so an enabled exact FFN
  observer must report zero drift through input normalization, HC projection
  and recombination, FFN norm, router projection/probabilities, and router
  weights. Generated-output parity alone can no longer hide a regression in an
  already-proven internal boundary.
- Layers 0, 30, and 42 consistently localized the remaining error. Shared
  expert output drifted in every 2-5 row observation. Routed output was exact
  for 2-4 rows and differed only for the five-row observation. Short-matrix
  shared maxima were about `3.58e-7`, `3.58e-7`, and `3.05e-5`; routed five-row
  maxima were about `1.19e-7`, `2.38e-7`, and `1.19e-6`. Shared output therefore
  dominates the combined FFN and HC error, especially at layer 42.
- A correctness-only layer-42 `code_8k` replay kept every boundary through
  router weights exact. Shared max remained about `3.05e-5`; five-row routed
  max reached about `7.63e-6`. Observer and same-binary no-observer output were
  byte-identical, and timings were ignored.
- All changes remain observer-only and Metal-focused. Normal prefill, DSpark
  sidecar stages, and production target decode do not enable the exact FFN
  shadow or its extra readbacks.

Phase 0.57 checks:

```sh
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
# A separate code_8k layer-42 observer/control replay used -n 32 and cmp;
# output was byte-identical and timings were ignored.
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./tests/dspark_fast_verifier_soak.sh
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.58 expert-internal boundary capture added on 2026-07-14:

- Extended the selected-layer FFN observer with shared gate, shared up, shared
  SwiGLU mid, and routed weighted-SwiGLU mid tensors. Dimensions come from the
  observed layer rather than fixed FFN-width assumptions. Routed batch mid is
  read in its actual F16/F32 format and converted only for diagnostics.
- Do not compare routed gate/up or per-expert down scratch between batch and
  serial paths. Normal serial decode uses fused pair-SwiGLU and direct six-way
  down-sum kernels that intentionally do not materialize those buffers. An
  initial observer draft read them and produced large meaningless differences
  despite exact routed mid/output; those comparisons were removed. The valid
  routed boundaries are weighted mid and accumulated output.
- Layers 0, 30, and 42 all show shared gate/up projection as the first
  persistent expert divergence. Shared gate/up max differences reached roughly
  `4.77e-7`, `1.91e-6`, and `2.86e-6`; SwiGLU mid amplified this to roughly
  `9.54e-7`, `1.19e-6`, and `1.14e-5` in the short matrices. The existing F16
  shared-down path then produced the previously observed output drift.
- Routed weighted mid was bit-exact in every 2-5 row observation at all three
  layers. Routed output remained exact for 2-4 rows. Only five-row batches
  differed, proving that their routed error begins after weighted SwiGLU in the
  non-direct down-projection/six-expert-sum path; short-matrix maxima remained
  about `1.19e-7`, `2.38e-7`, and `1.19e-6`.
- A correctness-only layer-42 `code_8k` replay reached shared gate/up maxima of
  about `1.91e-6`, shared mid max `1.91e-5`, and shared output max `3.05e-5`.
  Routed mid stayed exact; five-row routed down/sum reached `7.63e-6` max.
  Observer and same-binary no-observer text were byte-identical; timings were
  ignored.
- This phase only adds observer allocations, copies, readbacks, and reporting.
  Production execution and non-observer numerical paths are unchanged.

Phase 0.58 checks:

```sh
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
# A separate code_8k layer-42 observer/control replay used -n 32 and cmp;
# output was byte-identical and timings were ignored.
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./tests/dspark_fast_verifier_soak.sh
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.59 exact shared gate/up rows added on 2026-07-14:

- Added an internal observer-only `exact_shared_gate_up` selection to the
  batched FFN. Normal prefill, layer-major prefill, DSpark sidecar stages, and
  every production caller pass `false`; only the selected-layer exact FFN
  shadow passes `true`.
- The exact path creates row views over batch shared gate/up/mid and FFN-norm
  tensors, then calls the existing one-token
  `ds4_gpu_shared_gate_up_swiglu_q8_0_tensor` once per row while the same Metal
  command batch remains open. The original Metal kernel and backend API are
  unchanged, and no duplicated row kernel was introduced.
- Shared gate, shared up, and shared SwiGLU mid became bit-exact in every 2-5
  row observation at layers 0, 30, and 42. The correctness harness now asserts
  those three zero-drift metrics in addition to all previously proven
  boundaries through router weights.
- At layers 0 and 30, shared-output max drift fell to roughly one F32 ulp
  (`1.49e-7` to `2.38e-7`). Layer 42 still reached `3.05e-5`, so shared down and
  its F16 batch output are now the first persistent shared-expert boundary.
  Routed mid remains exact; the independent five-row down/sum drift is
  unchanged.
- A correctness-only layer-42 `code_8k` replay kept shared gate/up/mid exact in
  every observation. Shared output reached `3.05e-5` max and five-row routed
  down/sum reached `7.63e-6`. Observer and same-binary no-observer text were
  byte-identical; timings were ignored.
- This is still observer-only correctness work, not a throughput claim. The
  row-view allocation and repeated kernel encoding are intentionally outside
  production execution until the complete FFN is exact and can be evaluated as
  one controlled runtime candidate.

Phase 0.59 checks:

```sh
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
# A separate code_8k layer-42 observer/control replay used -n 32 and cmp;
# output was byte-identical and timings were ignored.
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./tests/dspark_fast_verifier_soak.sh
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.60 exact shared-down rows added on 2026-07-14:

- Added an internal observer-only `exact_shared_down` selection to the batched
  FFN. Production prefill, layer-major prefill, and DSpark sidecar callers pass
  `false`; only the selected-layer exact FFN shadow passes `true`.
- The exact path creates row views over the F32 batch shared-mid and
  shared-output tensors, then calls the original one-token
  `ds4_gpu_matmul_q8_0_tensor` Q8 shared-down matvec once per row. It bypasses
  the batch F16-output optimization, so the existing F32 HC expansion consumes
  the shared result without an F16 round trip. No backend API or Metal kernel
  changed, and CUDA/ROCm paths were untouched.
- Shared output became bit-exact in every observation at layers 0, 30, and 42;
  the correctness harness now asserts `shared_max=0` in addition to exact
  shared gate/up/mid and all earlier boundaries.
- A retained layer-42 replay was completely exact through FFN combine and HC
  post for two-, three-, and four-row proposals. The five-row observation had
  `shared_max=0` and only the previously isolated routed down/sum divergence:
  `routed_max=1.19209e-6`, `ffn_max=1.90735e-6`, and
  `hc_max=1.90735e-6`. Generated output remained byte-identical in every
  correctness-matrix case; timings were ignored.
- This remains observer-only arithmetic localization. It does not claim or
  measure a production throughput improvement.

Phase 0.60 checks:

```sh
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
./tests/dspark_fast_verifier_soak.sh
git diff --check
```

Phase 0.61 exact routed MoE rows added on 2026-07-14:

- Added an internal observer-only `exact_routed_moe` selection to the batched
  FFN. Production prefill, layer-major prefill, and DSpark sidecar callers pass
  `false`; only the selected-layer exact FFN shadow passes `true`.
- The exact path creates row views over batch FFN norm, router IDs/weights,
  routed gate/up/mid/down scratch, and routed output, then calls the existing
  `ds4_gpu_routed_moe_one_tensor` once per row while the outer Metal command
  batch remains open. This reuses the one-token fused pair-SwiGLU and direct
  six-expert down/sum arithmetic without a host synchronization, new backend
  API, or new Metal kernel. CUDA/ROCm code was not changed.
- The prior batch path used direct down/sum only through four tokens; five rows
  took separate expert down projections followed by a sum kernel and drifted
  after an otherwise exact routed mid. Reusing the one-token path removes that
  final arithmetic difference.
- Tightened `dspark_gpu_candidates_correctness.sh`: every exact FFN observer
  record in a log must now report `first=none` and `result=exact`. A passing
  short proposal can no longer hide a drifting five-row proposal in the same
  test case.
- Every observed two- through five-row FFN at layers 0, 30, and 42 is now
  bit-exact through HC post. The retained layer-42 five-row record reported
  exact router IDs `30/30`, zero drift for routed mid/output and shared
  gate/up/mid/output, `ffn_max=0`, five exact HC rows, and `hc_max=0`.
  Generated output remained byte-identical throughout the correctness matrix;
  timings were ignored.
- This completes the selected-layer FFN arithmetic proof. It remains an
  observer-only composition and is not yet enabled as target verification
  authority or presented as a throughput improvement.

Phase 0.61 checks:

```sh
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
./tests/dspark_fast_verifier_soak.sh
bash -n tests/dspark_gpu_candidates_correctness.sh
git diff --check
```

Phase 0.62 opt-in all-layer exact FFN runtime added on 2026-07-14:

- Added `DS4_DSPARK_EXACT_FFN_BATCH=1`, gated by GPU DSpark runtime. Default
  exact verification is unchanged when the variable is absent or zero.
- In the candidate path, every target layer preserves one-token attention and
  autoregressive cache-update order for each proposal row, then authoritatively
  runs the proven exact FFN composition across the contiguous rows in the same
  Metal command stream. It records target HC capture rows before alternating
  HC storage is reused by the next layer. The candidate does not add a
  per-layer host synchronization.
- A selected-layer exact FFN observer disables runtime authority for that call,
  so observer readback/replay semantics and the candidate cannot overlap.
  Recoverable candidate failure still follows the existing exact-verifier
  frontier restore and serial fallback contract.
- Existing runtime diagnostics now emit an explicit all-layer candidate
  pass/fail record. Resumed sync emits a diagnostics-only retention marker;
  correctness tests require a later successful candidate call, proving the
  option remained active after state extension rather than merely matching the
  first turn.
- `dspark_gpu_candidates_correctness.sh` gained an explicit candidate test
  switch and forces the new runtime variable to zero in its default mode. The
  fast-verifier soak also forces it off so inherited shell state cannot alter
  exact fallback coverage.
- Added `dspark_exact_ffn_batch_runtime_soak.sh`. It compares baseline and
  candidate output for 64-token generation, the rolling DSpark-window prompt,
  and a resumed two-turn chat. It requires repeated candidate and exact-batch
  success records, rejects target-capture/proposal fallback, and proves a
  successful candidate verification after resumed sync.
- Candidate output was byte-identical in the short runtime matrix, both
  64-token cases, and resumed chat. Default exact runtime, fast runtime, and the
  layer-42 strict FFN observer also remained byte-identical. This is correctness
  evidence only; no tok/s benchmark was run by Codex and no performance claim
  is made.
- Existing benchmark runners already clear every inherited
  `DS4_DSPARK_*` variable before selecting their explicit mode, so the new
  candidate cannot contaminate their current baseline/runtime measurements.

Phase 0.62 checks:

```sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_FFN_BATCH_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_exact_ffn_batch_runtime_soak.sh
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_fast_verifier_soak.sh
DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
bash -n tests/dspark_gpu_candidates_correctness.sh \
  tests/dspark_exact_ffn_batch_runtime_soak.sh \
  tests/dspark_fast_verifier_soak.sh
git diff --check
```

Phase 0.63 exact FFN batch benchmark ablation prepared on 2026-07-14:

- Added `run_dspark_comparison.py --exact-ffn-batch-ablation`. It changes the
  paired modes from non-DSpark baseline/default runtime to default exact DSpark
  and the same runtime with `DS4_DSPARK_EXACT_FFN_BATCH=1`.
- Pair order alternates default/candidate then candidate/default. Both modes use
  the same base model, sidecar, prompt, deterministic sampling, acceptance
  policy, context, and generated-token count. Every warmup and measured stdout
  must match the first default-exact warmup byte for byte.
- The ablation is a throughput-only pass: runtime diagnostics and runtime stats
  are absent from both process environments. Inherited `DS4_DSPARK_*` and timing
  instrumentation variables are still cleared before each child. This isolates
  the FFN authority switch without hot-path log or clock-read instrumentation.
- The new option is mutually exclusive with `--fast-verifier`. The ordinary
  baseline/runtime and fast-verifier modes retain their previous commands,
  stats parsing, CSV fields, and report format.
- Ablation summaries report default-exact and candidate medians, ratio of
  medians, median paired ratio, candidate percentage delta, and measured pair
  count. CSV/JSON/Markdown output and machine/process/thermal metadata remain in
  the ignored local-runs directory.
- README commands now show the dry run and user-confirmed execution. Static
  validation covered Python compilation, all three dry-run modes, mutual-option
  rejection, inherited-environment clearing, legacy and ablation paired-summary
  math, and both report formats. No model process or tok/s benchmark was run by
  Codex.

Phase 0.63 checks:

```sh
python3 -m py_compile speed-bench/run_dspark_comparison.py
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --allow-dirty
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --allow-dirty --fast-verifier
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --allow-dirty --exact-ffn-batch-ablation
# A combined --fast-verifier/--exact-ffn-batch-ablation invocation was rejected.
# Inline synthetic assertions covered both comparison summaries, reports,
# blank ablation stats fields, CSV/JSON/Markdown writing, and environment reset.
git diff --check
```

The user ran the Phase 0.63 paired, uninstrumented ablation on 2026-07-14.
Default exact measured 19.57/19.65/19.70 t/s and exact FFN batch measured
20.66/20.81/20.77 t/s in the corresponding alternating-order pairs. The
per-pair ratios were approximately 1.0557x, 1.0590x, and 1.0543x; the ratio of
medians was 1.0570x and the median paired ratio was 1.0557x. All six measured
outputs had the same SHA-256. This is a consistent +5.7% generation-throughput
win for the opt-in exact FFN batch runtime under this workload, with no observed
output change. Raw results are in the ignored local run
`speed-bench/local-runs/20260714-011657/results.csv`.

Phase 0.64 exact FFN attribution profile prepared on 2026-07-14:

- Added exact-FFN attempt and completion counters to the existing
  end-of-session DSpark runtime stats record. A candidate attempt is counted
  only for a multi-row exact verification that selects the all-layer FFN path;
  completion requires the full 43-layer phase to finish successfully. There is
  still no per-call logging when runtime diagnostics are disabled.
- Added `speed-bench/run_dspark_exact_ffn_batch_profile.py`, a separate
  instrumented diagnostic comparing default exact against the same runtime with
  `DS4_DSPARK_EXACT_FFN_BATCH=1`. It explicitly selects Metal, clears inherited
  DSpark and instrumentation variables, alternates pair order, and requires all
  output to match byte for byte.
- The profile attributes target milliseconds per emitted token into the exact
  layer phase, output head, and residual, and reports generation-sidecar time.
  It also reports exact-FFN completions/attempts, general verifier fallbacks,
  acceptance depth, and target positions per evaluation. Instrumented t/s is
  labeled as context only and must not be compared with the Phase 0.63
  uninstrumented result.
- The default profile is one pair with no warmup and requires
  `--confirm-ready`. The user-run command is:

  ```sh
  python3 speed-bench/run_dspark_exact_ffn_batch_profile.py --confirm-ready
  ```

- Extended the existing correctness matrix with
  `DS4_TEST_DSPARK_RUNTIME_STATS=1`. In default exact mode it requires zero FFN
  candidate outcomes; in candidate mode it requires at least one attempt and
  equal attempt/completion totals in every retained process.
- Existing comparison parsing now retains the two new integer fields. Static
  checks covered all old comparison dry-run modes, complete synthetic stats
  parsing, profile summary/report math, inherited-environment clearing, and the
  `--confirm-ready` refusal. Codex did not run the timed profile or any tok/s
  benchmark; timings printed incidentally by correctness runs were ignored.

Phase 0.64 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_FFN_BATCH_RUNTIME=1 \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_exact_ffn_batch_runtime_soak.sh
./tests/dspark_fast_verifier_soak.sh
python3 -m py_compile speed-bench/run_dspark_comparison.py \
  speed-bench/run_dspark_exact_ffn_batch_profile.py
python3 speed-bench/run_dspark_exact_ffn_batch_profile.py \
  --dry-run --allow-dirty
bash -n tests/dspark_gpu_candidates_correctness.sh
git diff --check
```

The user ran the Phase 0.64 instrumented attribution profile on 2026-07-14.
Default exact used 46.728 target ms/emitted, of which 44.746 ms was the layer
phase. Exact FFN used 44.360 target ms/emitted, of which 42.400 ms was the layer
phase. It therefore saved 2.368 target ms/emitted, with 2.346 ms (99.1%)
accounted for by the layer phase. Head time was effectively unchanged
(1.669/1.673 ms), as were residual target time (0.313/0.287 ms) and sidecar time
(3.663/3.673 ms).

Both modes emitted the same 64 tokens with the same stdout SHA-256, 14 target
evaluations, 69 target positions, 4.923 average accepted depth, and 4.929 target
positions per evaluation. Exact FFN completed 14/14 attempts with zero verifier
fallbacks. Together with the Phase 0.63 uninstrumented +5.7% result, this
supports promoting exact FFN batching to the normal exact DSpark verifier path.
The profile's instrumented 19.77/20.72 t/s values remain context only. Raw
results are in the ignored local run
`speed-bench/local-runs/ffn-profile-20260714-013402/results.csv`.

Phase 0.65 exact FFN default promotion completed on 2026-07-14:

- Promoted all-layer exact FFN batching to the normal multi-row exact DSpark
  verifier. With `DS4_DSPARK_GPU_RUNTIME=1`, an absent, empty, or nonzero
  `DS4_DSPARK_EXACT_FFN_BATCH` now selects the proven batched path.
  `DS4_DSPARK_EXACT_FFN_BATCH=0` preserves the former serial FFN
  implementation as an explicit diagnostic control.
- The selected-layer exact FFN observer still disables batched runtime
  authority for its observed call. No FFN kernel or target-state restoration
  logic changed, so a recoverable whole-verifier failure continues through the
  existing frontier restore and serial token-evaluation fallback.
- Renamed comparison modes around the promoted behavior.
  `run_dspark_comparison.py --serial-ffn-ablation` now compares serial exact
  against default exact FFN, alternates serial/default order, and reports the
  default's delta. The old `--exact-ffn-batch-ablation` spelling remains an
  alias. Ordinary baseline/runtime comparisons now measure the promoted
  default automatically.
- Updated the attribution profile to compare `serial_exact` against
  `default_exact`; serial sets `DS4_DSPARK_EXACT_FFN_BATCH=0`, while default
  leaves the override absent. Reports and metadata use the promoted names.
- Reworked `dspark_gpu_candidates_correctness.sh` so normal runtime requires
  exact FFN batching and `DS4_TEST_DSPARK_SERIAL_FFN_RUNTIME=1` requires zero
  FFN-batch attempts. Mixed fast/exact sessions allow zero attempts while fast
  authority is active but require every attempt that occurs to complete.
- The exact-FFN and fast-verifier soaks explicitly clear inherited FFN
  overrides. The fast resumed-session check now proves that after fast authority
  is suspended, a later exact verification uses the promoted FFN path.
- Default exact, explicit serial, the layer-42 observer, and stats-enabled mixed
  fast/exact matrices all produced byte-identical output. Both long soaks
  passed. Codex ran no tok/s benchmark; incidental correctness timings were
  ignored.

Phase 0.65 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_SERIAL_FFN_RUNTIME=1 \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_FAST_VERIFY_RUNTIME=1 \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_exact_ffn_batch_runtime_soak.sh
./tests/dspark_fast_verifier_soak.sh
python3 -m py_compile speed-bench/run_dspark_comparison.py \
  speed-bench/run_dspark_exact_ffn_batch_profile.py
bash -n tests/dspark_gpu_candidates_correctness.sh \
  tests/dspark_exact_ffn_batch_runtime_soak.sh \
  tests/dspark_fast_verifier_soak.sh
# Dry runs covered ordinary, fast, serial-control, legacy-alias, and profile
# commands. Synthetic checks covered both renamed summaries and reports.
git diff --check
```

Phase 0.66 exact attention-pre batch observer added on 2026-07-14:

- Added `DS4_DSPARK_EXACT_ATTN_PRE_BATCH_OBSERVER_LAYER=<layer>` for exact
  multi-row DSpark verification. It is observer-only and inactive for one-row
  calls, invalid layers, or an unset environment variable.
- At the selected layer, the observer fences prior target work, shadows only
  the cache-independent attention prefix over the authoritative contiguous HC
  rows, reads its intermediates, and then runs the unchanged one-token serial
  attention path in its original autoregressive cache-update order.
- The shadow covers plain HC input normalization, the attention HC projection,
  Sinkhorn/split weights, HC recombination, and the following weighted attention
  norm. It stops before Q/KV projection, RoPE, raw/compressed cache writes,
  indexer work, or attention itself.
- The attention HC projection uses the Metal-only exact F16 decode-row primitive
  already proven by exact FFN batching. HC split/recombination and normalization
  select the same fused or reference sequence as one-token decode. No generic
  prefill projection is made authoritative.
- Observer reports compare normalized HC input, projected HC mix, split weights,
  recombined attention input, and attention norm. They name the first divergent
  boundary and require every final norm row to match byte for byte before
  reporting `first=none ... result=exact`.
- The observer's extra command fences, tensor copies, host allocations,
  readbacks, and logs occur only when its layer variable is active. Default
  exact verification remains uninstrumented. After observation, promoted exact
  FFN batching continues normally; explicit serial FFN control also has a
  tested split-layer continuation.
- Strict correctness matrices at layers 0, 30, and 42 found every observed
  boundary bit-exact across proposal widths and all five prompt/session cases,
  including rolling-window and resumed sessions. The layer-42 serial-FFN
  control matrix was also exact. Default runtime, the existing layer-42 exact
  FFN observer, the exact-FFN soak, and the fast-verifier soak remained
  byte-identical and passed.
- This establishes a safe batch boundary through attention normalization, not
  an authoritative attention runtime or speed claim. Codex ran no tok/s
  benchmark; incidental correctness timings were ignored.

Phase 0.66 checks:

```sh
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_ATTN_PRE_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
DS4_DSPARK_EXACT_ATTN_PRE_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_SERIAL_FFN_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
./tests/dspark_exact_ffn_batch_runtime_soak.sh
./tests/dspark_fast_verifier_soak.sh
bash -n tests/dspark_gpu_candidates_correctness.sh
git diff --check
```

Phase 0.67 exact Q/KV-preparation observer added on 2026-07-14:

- Extended `DS4_DSPARK_EXACT_ATTN_PRE_BATCH_OBSERVER_LAYER=<layer>` from the
  proven attention-norm boundary through every cache-independent Q/KV stage:
  Q-LoRA projection, Q-LoRA RMS norm, KV projection and RMS norm, final Q
  projection, per-head Q norm, position-correct Q RoPE, and position-correct KV
  RoPE. The shadow still stops before FP8 KV quantization, raw/compressed cache
  writes, indexer work, or attention.
- The first generic multi-row Q8 attempt produced real arithmetic drift at the
  first new boundary on layer 42 with two proposal rows: `q_lora_max=6.33299e-08`.
  It propagated through `kv_projection_max=7.45058e-08`,
  `q_head_norm_max=1.43051e-06`, `q_rope_max=1.43051e-06`, and
  `kv_rope_max=7.15256e-07`. This localized the mismatch to the generic
  multi-row Q8 projection rather than normalization or RoPE.
- Added an observer-local Metal exact Q8 decode-row wrapper. It creates row
  views and submits the existing one-token Q8 decode kernel once per proposal
  row in the same command stream. This preserves one-token reduction arithmetic
  without a host fence per row. No CUDA or ROCm path was added.
- With exact-row Q8 projection, the multi-row Q/KV RMS and RoPE kernels are
  bit-identical to one-row decode. Reports now name all new boundaries and
  require both final Q-RoPE and KV-RoPE rows to match byte for byte. The
  correctness harness explicitly requires `q_rope_max=0`, `kv_rope_max=0`,
  `first=none`, and `result=exact` on every observer record.
- Strict matrices at layers 0, 30, and 42 passed all five prompt/session cases,
  including medium context, rolling-window positions, and resumed chat. Layer
  42 also passed with explicit serial FFN. Default runtime, the existing exact
  FFN observer, the exact-FFN runtime soak, and the fast-verifier soak remained
  byte-identical and passed.
- This establishes an exact batched command-stream boundary immediately before
  KV quantization/cache mutation. It remains observer-only and makes no runtime
  or speed claim. Codex ran no tok/s benchmark; incidental correctness timings
  were ignored.

Phase 0.67 checks:

```sh
for layer in 0 30 42; do
  DS4_DSPARK_EXACT_ATTN_PRE_BATCH_OBSERVER_LAYER=$layer \
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
done
DS4_DSPARK_EXACT_ATTN_PRE_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_SERIAL_FFN_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
./tests/dspark_exact_ffn_batch_runtime_soak.sh
./tests/dspark_fast_verifier_soak.sh
bash -n tests/dspark_gpu_candidates_correctness.sh
git diff --check
```

Phase 0.68 gated exact attention-pre runtime added on 2026-07-14:

- Added opt-in `DS4_DSPARK_EXACT_ATTN_PRE_BATCH=1` for exact multi-row target
  verification. The gate requires DSpark GPU runtime, applies only when more
  than one proposal row is verified, and remains disabled by default.
- Split single-token Metal attention decode at the boundary proven in phases
  0.66/0.67. Full and attention-only decode still enter at the original start;
  a new tail-only entry joins immediately before KV quantization/raw-cache
  storage. Every persistent cache mutation remains below that join point.
- The gated path prepares HC mixing, attention norm, normalized Q-LoRA state,
  Q RoPE, and KV RoPE over all proposal rows in one command stream. It then
  binds row views for those exact artifacts and invokes the unchanged serial
  tail in original autoregressive row order. Compression, indexer state,
  attention, output projection, HC expansion, and all raw/compressed cache
  writes therefore retain their one-token implementation and ordering.
- Runtime preparation omits observer-only Q-raw/Q-norm copies and all host
  fences/readbacks. The observer uses the same preparation helper with boundary
  capture enabled; production uses it with capture disabled.
- If cache-free preparation cannot encode, the verifier fences the partial
  scratch-only command stream and resumes the original full serial attention
  path for that layer. Diagnostics report per-verification layer attempts and
  successes; strict tests require `layers=43 attempts=43 successes=43
  result=pass`, so a silent fallback cannot satisfy correctness coverage.
- A serial-FFN control initially exposed that per-row `after_attn` views were
  allocated only when FFN splitting was active. Attention batching now requests
  those views independently. The corrected gated runtime passes with both
  promoted exact FFN batching and explicit serial FFN.
- Cache-state equivalence is structural rather than a bulk-cache readback: the
  prepared Q/KV, attention norm, Q-LoRA norm, and HC weights were already proven
  byte-identical, and the exact same tail consumes them and performs every
  persistent mutation in the same row order. Long-generation, rolling-window,
  and resumed-chat output soaks provide end-to-end coverage of that invariant.
- Default runtime, the extended attention observer, the existing exact FFN
  observer, the exact-FFN soak, and the fast-verifier soak all remained exact.
  This phase makes no speed claim. Codex ran no tok/s benchmark; incidental
  correctness timings were ignored.

Phase 0.68 checks:

```sh
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_ATTN_PRE_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_ATTN_PRE_RUNTIME=1 \
DS4_TEST_DSPARK_SERIAL_FFN_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_PRE_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
./tests/dspark_exact_attention_pre_batch_runtime_soak.sh
./tests/dspark_exact_ffn_batch_runtime_soak.sh
./tests/dspark_fast_verifier_soak.sh
bash -n tests/dspark_gpu_candidates_correctness.sh \
  tests/dspark_exact_attention_pre_batch_runtime_soak.sh
git diff --check
```

Phase 0.69 exact attention-pre throughput ablation prepared on 2026-07-14:

- Added `--attention-pre-ablation` to
  `speed-bench/run_dspark_comparison.py`. It compares two DSpark exact-verifier
  modes: `default_exact` with the Phase 0.68 gate unset and
  `attention_pre_exact` with `DS4_DSPARK_EXACT_ATTN_PRE_BATCH=1`.
- Both modes use Metal, the same base model and DSpark sidecar, default exact
  all-layer FFN batching, identical prompt/seed/temperature/token limits, and
  the same acceptance policy. The candidate differs by exactly one environment
  variable.
- The ablation is an uninstrumented throughput pass. Runtime diagnostics and
  end-of-session stats are disabled in both modes, inherited `DS4_DSPARK_*` and
  profiling/logging variables are cleared, and every stdout stream must remain
  byte-identical to the first warmup/reference stream.
- Pair order alternates default/attention-pre then attention-pre/default. The
  existing runner retains warmups, cooldowns, process/thermal metadata, raw
  stdout/stderr, hashes, CSV/JSON output, and dirty-worktree protection. The
  summary reports both medians, ratio of medians, median paired ratio, candidate
  percentage delta, and measured pair count.
- `--attention-pre-ablation` is mutually exclusive with `--fast-verifier` and
  `--serial-ffn-ablation`; the fast verifier does not use this exact attention
  preparation path.
- Updated `speed-bench/README.md` with dry-run and real user-run commands, the
  one-variable comparison contract, and the dedicated Phase 0.68 correctness
  soak.
- Codex did not execute the benchmark. Python compilation, all comparison-mode
  dry runs, environment isolation checks, mutual-exclusion rejection, and a
  synthetic two-pair summary/report check passed without model execution.

Phase 0.69 checks:

```sh
python3 -m py_compile speed-bench/run_dspark_comparison.py
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --allow-dirty --attention-pre-ablation
python3 speed-bench/run_dspark_comparison.py --dry-run --allow-dirty
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --allow-dirty --fast-verifier
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --allow-dirty --serial-ffn-ablation
# Synthetic checks covered mode environments, medians, paired ratios, report
# generation, and mutual-exclusion rejection.
git diff --check
```

Phase 0.70 exact attention-pre default promotion completed on 2026-07-14:

- The user ran the uninstrumented paired Metal ablation from Phase 0.69. The
  serial-attention exact control median was `20.42 t/s`; the attention-pre
  candidate median was `21.66 t/s`, for a `1.0607x` ratio of medians and
  `+6.1%` delta.
  The three paired ratios were `1.0597453x`, `1.0606951x`, and `1.0607248x`.
- Every measured stdout stream had SHA-256
  `86f4851c044b82fffe568644343670a83ea6815b70c160b5a28a0fb357c52998`.
  Prefill was approximately neutral, pair order alternated, and macOS reported
  no thermal or performance warning. The process snapshot still contained
  background activity, including `syspolicyd` and Stats; that caveat remains,
  but the relative result was exceptionally stable across all three pairs.
- Raw benchmark artifacts are in
  `speed-bench/local-runs/20260714-110640/results.csv`. These local artifacts
  remain untracked; the measurements and decision are preserved here.
- Promoted exact attention preparation to the default DSpark exact-verifier
  path. `DS4_DSPARK_EXACT_ATTN_PRE_BATCH=0` now explicitly selects the retained
  fully serial attention implementation for diagnostics and ablations.
- Added aggregate attention-pre attempts/successes to the end-of-session DSpark
  runtime stats. This proves the default path completed without enabling
  per-verification diagnostics in throughput runs. Session counters are updated
  only when end-of-session runtime stats are requested.
- Inverted `--attention-pre-ablation` to compare the serial control (`=0`)
  against the promoted default. Both sides remain uninstrumented and
  byte-identical, and pair order now alternates serial/default then
  default/serial.
- Updated the correctness matrix to require complete 43-layer attention
  preparation in ordinary exact runtime. Added
  `DS4_TEST_DSPARK_SERIAL_ATTN_PRE_RUNTIME=1` for the explicit serial control;
  the old `DS4_TEST_DSPARK_ATTN_PRE_RUNTIME=1` test switch remains as a
  compatibility assertion for Phase 0.68 commands.
- Updated the long-generation, rolling-window, and resumed-chat attention-pre
  soak to test the promoted default with no enabling environment variable.
- Resumed-sync diagnostics now suppress FFN and attention-pre "retained" status
  lines while the corresponding selected-layer observer has made that runtime
  path non-authoritative.
- Codex did not execute a tok/s benchmark. All validation below is build,
  correctness, soak, parser, dry-run, or synthetic summary coverage.

Phase 0.70 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_SERIAL_ATTN_PRE_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_SERIAL_ATTN_PRE_RUNTIME=1 \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_PRE_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
./ds4_test --dspark-validation --dspark-shape-binding
./tests/dspark_exact_attention_pre_batch_runtime_soak.sh
./tests/dspark_exact_ffn_batch_runtime_soak.sh
./tests/dspark_fast_verifier_soak.sh
python3 -m py_compile speed-bench/run_dspark_comparison.py
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --allow-dirty --attention-pre-ablation
# Synthetic checks cover promoted-default statistics and the inverted ablation.
bash -n tests/dspark_gpu_candidates_correctness.sh \
  tests/dspark_exact_attention_pre_batch_runtime_soak.sh
git diff --check
```

Phase 0.71 post-promotion exact-runtime profiling prepared on 2026-07-14:

- The user ran the ordinary paired Metal baseline/runtime comparison after the
  Phase 0.70 promotion. Baseline median was `23.54 t/s`; promoted exact DSpark
  was `21.60 t/s`, for `0.9176x` ratio of medians and an `8.2%` deficit. The
  three paired ratios were `0.9171623x`, `0.9171975x`, and `0.9196087x`.
- Every measured output retained SHA-256
  `86f4851c044b82fffe568644343670a83ea6815b70c160b5a28a0fb357c52998`.
  macOS reported no thermal or performance warning; WindowServer and Stats
  remained visible background activity, but the narrow paired spread makes the
  remaining deficit unlikely to be interference.
- Runtime completed all `602/602` exact attention preparations, accepted an
  average depth of `4.923`, avoided 50 target calls, and had zero verifier
  fallbacks. Generation sidecar cost was `3.600 ms/emitted`; exact target work
  was `42.501 ms/emitted`; residual runtime overhead was approximately
  `0.2 ms/emitted` relative to the observed `46.30 ms/token` runtime total.
- The result composes with the earlier isolated improvements: the pre-FFN exact
  ratio `0.8176x`, exact-FFN gain `1.057x`, and attention-pre gain `1.061x`
  predict about `0.917x`, matching the observed `0.9176x`. Promotion therefore
  retained both gains; it did not introduce a regression.
- Raw benchmark artifacts are in
  `speed-bench/local-runs/20260714-113028/results.csv` and remain ignored. Codex
  did not run this tok/s benchmark.
- Audited `run_dspark_exact_layer_profile.py` against the promoted verifier.
  Its old decode-stage contract expected `attn_hc_pre` plus serial FFN records,
  but authoritative execution now prepares the attention prefix in one batch,
  executes only the cache-mutating attention tail serially, and runs exact FFN
  in one batch. The old parser would therefore fail or misattribute preparation
  work to the first serial-tail boundary.
- Added diagnostic-only `DS4_DSPARK_EXACT_LAYER_PROFILE=1` with
  `DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER=<layer>`. For multi-row exact calls at
  the selected layer, it fences before the layer and records three synchronized
  authoritative components: `attention_pre_batch`, `attention_tail_serial`,
  and `ffn_batch`. One-token verifier calls are deliberately ignored.
- The new gate requires both promoted runtime components and adds no command
  fence, timer, or log unless explicitly enabled. Its disabled path performs
  one environment lookup per exact verifier call, outside the 43-layer loop.
- Reworked the existing profile harness around those three records. Every batch
  must contain an identical `(start, width)` triplet across components; missing
  or fallback work is rejected. Component milliseconds are divided by proposal
  rows before median aggregation, and the report ranks preparation, serial
  attention tail, and exact FFN for layers `0,21,42` by default.
- Resume validation now includes the three-component contract, so an older
  Phase 0.52 decode-stage directory is rejected before its incompatible layer
  logs are considered for reuse.
- The diagnostic remains synchronized and changes scheduling. Its component
  ordering is only evidence for choosing the next implementation target; its
  printed generation rate and summed component values are not throughput
  claims. Codex prepared and dry-ran the harness but did not execute the timed
  profile.

Phase 0.71 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
./ds4_test --dspark-validation --dspark-shape-binding
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
python3 -m py_compile speed-bench/run_dspark_exact_layer_profile.py
python3 speed-bench/run_dspark_exact_layer_profile.py \
  --dry-run --allow-dirty
# Synthetic records cover normalized medians, report generation, exact batch
# signatures, resume compatibility, and rejection of an incomplete component
# triplet.
git diff --check
```

Phase 0.72 exact attention-suffix observer completed on 2026-07-14:

- The user-run Phase 0.71 synchronized profile measured the promoted exact
  runtime at layers `0,21,42`. Attention preparation stayed nearly flat at
  `0.255/0.263/0.267 ms/row`, exact FFN stayed nearly flat at
  `0.389/0.398/0.396 ms/row`, while the serial attention tail grew from
  `0.319` to `0.433` to `0.614 ms/row`. At layer 42 it represented about 48%
  of the synchronized layer total. These values are diagnostic component
  timings, not throughput measurements.
- Added the selected-layer diagnostic
  `DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=<layer>`. It leaves cache
  writes, compressor/indexer state, attention, and the normal serial suffix
  authoritative. After each serial row it captures inverse-RoPE heads plus the
  exact low-rank and output projections. It then shadows the existing generic
  batched output projection and HC expansion over the same rows, reads the
  candidate boundaries, restores exact HC bytes, and continues through the
  default exact FFN batch.
- The observer does not split or alter the production decoder and performs no
  work when disabled. Its readbacks, synchronization, host buffers, and HC
  restore are diagnostic only and must never be used for throughput claims.
- Extended the DSpark correctness harness to make all three selected-layer
  observers mutually exclusive, require exact-verifier runtime, reject shadow
  fallback or malformed records, and require projection A to remain byte-exact
  (`low_max=0 low_rms=0`). The final batch suffix may report `drift`; generated
  stdout must still be byte-identical because exact HC is restored before FFN.
- Correctness matrices passed at layers `0`, `21`, and `42`, including short,
  medium-context, rolling-window, and resumed-chat cases. The ordinary exact
  runtime matrix without an observer also passed.
- At layer 21 the observed projection-B maximum was `1.2e-6` to `2.1e-6`, and
  final-HC maximum was `4.8e-7` to `9.5e-7`. At layer 42 projection-B maximum
  was `1.9e-6` to `7.6e-6`, and final-HC maximum was `7.6e-6` to `1.5e-5`.
  Projection A was byte-exact in every checked record. This localizes the first
  numerical difference to the batched Q8 projection-B implementation, not the
  grouped low-rank projection or stateful attention core.
- `make test` did not pass overall: `think-tool-recovery`,
  `logprob-vectors`, and `metal-ssd-streaming-cache-pressure` failed in model
  golden/tool parsing paths outside the changed DSpark exact-verifier path.
  Long-context, tool-call quality, local golden vectors, short prefill, Metal
  kernels, Metal tensor equivalence, DSpark validation/shape binding, and
  server checks passed. The targeted standalone DSpark validation and shape
  binding test also passed.
- Codex did not execute a tok/s benchmark.

Phase 0.72 checks:

```sh
make -j4
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=0 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=21 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
./ds4_test --dspark-validation --dspark-shape-binding
make test
# Overall failure was limited to the three model-golden/tool checks recorded
# above; the DSpark-specific and Metal equivalence coverage passed.
git diff --check
```

Phase 0.73 exact fused attention-suffix batch candidate completed on
2026-07-14:

- Audited the serial fused Metal projection-B/HC kernel. Its Q8_0 reduction is
  the ordinary one-token matvec reduction copied into the fusion, and its HC
  arithmetic consumes explicit token strides. The only missing batch contract
  was a proposal-row threadgroup dimension.
- Generalized `kernel_dsv4_q8_hc_expand4_q8_0` over `tgpig.y`. Every row keeps
  the serial kernel's SIMD reduction shape and HC accumulation order; only
  activation addresses gain the row stride. The existing serial caller still
  dispatches exactly one row.
- Refactored the Metal host wrapper through a checked internal rows helper and
  added the Apple-only
  `ds4_gpu_matmul_q8_0_hc_expand_batch_tensor`. The batch wrapper validates all
  row-scaled input, output, residual-HC, and split buffers before dispatching.
  CUDA and ROCm implementations were deliberately untouched.
- The selected-layer suffix observer still obtains projection A from the
  existing generic batch function. It then overwrites that function's
  approximate projection-B/HC outputs with the exact fused row-batched kernel
  before comparison. This duplicate observer work is intentional and means
  the current path is not a valid performance candidate yet.
- Tightened the correctness harness: every observed record must now report
  `first=none`, zero max/RMS differences for projection A, projection B, and
  final HC, all rows exact, and `result=exact`.
- Exact observer matrices passed at layers `0`, `21`, and `42`. Layer 42 logs
  covered proposal widths `2`, `3`, `4`, and `5`; every recorded boundary and
  HC row was byte-identical. The observer-disabled exact runtime matrix also
  remained byte-identical to baseline, covering the serial one-row use of the
  generalized shader.
- The 64-token long-generation and rolling-window exact-runtime soaks passed,
  as did resumed chat. DSpark validation and shape binding passed. The normal
  Metal build and `DS4_NO_GPU` object build completed; the CPU build retained
  only its existing unused-function/parameter warnings.
- Codex did not execute a tok/s benchmark.

Phase 0.73 checks:

```sh
make -j4
make ds4_cpu.o ds4_test
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=0 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=21 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_exact_attention_pre_batch_runtime_soak.sh
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.74 opt-in exact attention-suffix batch runtime completed on
2026-07-14:

- Refactored the Metal attention-output batch implementation through a shared
  internal helper and exported
  `ds4_gpu_attention_output_low_q8_batch_tensor`. The new entry point performs
  projection A only; it does not redundantly execute generic projection B.
  The existing full projection API retains its command-buffer composition and
  behavior.
- Split the exact serial attention decoder at the post-inverse-RoPE boundary.
  The cache-mutating attention core remains serial and autoregressive. A new
  output-only part can replay projection A, projection B, and HC expansion from
  captured exact head rows without touching KV, compressor, or indexer state.
- Added the opt-in `DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH=1` runtime. With default
  exact attention preparation enabled, each layer runs its stateful core
  serially, captures the exact head rows, runs row-batched projection A, and
  completes exact fused projection-B/HC expansion with the Phase 0.73 kernel.
  The promoted default exact runtime is unchanged.
- If the batch suffix is unavailable or fails, the verifier closes the current
  command batch and replays only the stateless suffix with the ordinary serial
  kernels. A forced reference-HC test exercised all 43 fallbacks and remained
  byte-identical to the exact control.
- Added attempt/success diagnostics and end-of-session stats for the suffix
  candidate. The correctness harness can require the opt-in runtime and checks
  that every reported verifier call completes all 43 layers without fallback.
- Added a dedicated long-generation, rolling-window, resumed-chat, and forced-
  fallback soak. The selected-layer observer still reports byte-exact
  projection A, projection B, and HC output at layers 0, 21, and 42 after the
  projection-A refactor.
- Added `--attention-suffix-ablation` to the user-run benchmark harness. It is
  an uninstrumented paired comparison of default exact DSpark against the
  opt-in suffix candidate, forces Metal, clears inherited DSpark diagnostics
  and stats, and requires byte-identical output. Codex only exercised dry-run
  command generation and synthetic summary formatting.
- Codex did not execute a tok/s benchmark.

Phase 0.74 checks:

```sh
make -j4
make ds4_cpu.o ds4_test
python3 -m py_compile speed-bench/run_dspark_comparison.py
bash -n tests/dspark_gpu_candidates_correctness.sh \
  tests/dspark_exact_attention_suffix_batch_runtime_soak.sh
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --allow-dirty --attention-suffix-ablation
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_ATTN_SUFFIX_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_ATTN_SUFFIX_RUNTIME=1 \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=0 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=21 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_exact_attention_suffix_batch_runtime_soak.sh
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.74 user-run throughput result on 2026-07-14:

- Run: `speed-bench/local-runs/20260714-124010/results.csv`, clean commit
  `be7798277d96b97d9e4798b201c7f456db3b452c`, Apple M1 Ultra, Metal,
  uninstrumented, one warmup per mode and three alternating measured pairs.
- Default exact median: `21.56 t/s`; exact attention-suffix batch median:
  `16.98 t/s`; ratio of medians: `0.7876x`; median paired ratio: `0.7850x`.
  The candidate regressed generation throughput by `21.2%`.
- All three measured candidate/reference ratios were tightly grouped at about
  `0.782x`, `0.788x`, and `0.785x`. Prefill remained effectively unchanged
  (`45.90` to `46.50 t/s` across both modes), and every output SHA-256 matched.
  This is a generation-path cost, not startup noise or semantic drift.
- Do not promote `DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH`. The result rejects the
  current small-proposal batching schedule as a Metal throughput optimization;
  it does not invalidate the exact fused kernel's correctness.
- Before changing this path again, add a synchronized diagnostic profile that
  separates serial attention core/head capture, batched projection A, and
  batched fused projection-B/HC. Compare those components against the default
  serial suffix at selected early/middle/late layers. Do not run another
  uninstrumented ablation until the regression has a measured owner.

Phase 0.75 attention-suffix attribution harness completed on 2026-07-14:

- Extended the existing opt-in exact-layer synchronized profiler boundaries.
  Default exact execution retains its combined `attention_tail_serial` record.
  The suffix candidate now emits `attention_core_capture_serial`,
  `attention_projection_a_batch`, and
  `attention_projection_b_hc_batch` records at the selected layer. These
  boundaries exist only when `DS4_DSPARK_EXACT_LAYER_PROFILE` is enabled;
  normal candidate scheduling is unchanged.
- Added `speed-bench/run_dspark_exact_attention_suffix_profile.py`. It runs one
  unprofiled exact reference and paired default/candidate synchronized profiles
  at layers `0`, `21`, and `42`, alternating pair order by layer. Every output
  must be byte-identical to the reference.
- The runner strictly validates each mode's stage contract, rejects unknown or
  missing records, and requires identical `(position, proposal width)`
  signatures between default and candidate. Its report compares default serial
  attention-tail milliseconds per proposal row with candidate serial
  core/capture, projection A, fused projection-B/HC, their sum, ratio, and
  delta. Attention-pre and FFN medians are reported as scheduling controls.
- A profile-enabled candidate correctness matrix passed reasoning, Italian,
  medium-context, rolling-window, and resumed-chat cases. Real layer-42 logs
  contained all five expected candidate records. The parser also accepted the
  three established default records from the prior exact-layer profile.
- The runner's dry-run command generation, synthetic aggregation/reporting,
  normal Metal build, and CPU object build passed. The CPU build retained only
  its existing unused-function/parameter warnings.
- Codex did not run the timed attribution profile or any tok/s benchmark.

Phase 0.75 checks:

```sh
make -j4
make ds4_cpu.o
python3 -m py_compile \
  speed-bench/run_dspark_exact_attention_suffix_profile.py
python3 speed-bench/run_dspark_exact_attention_suffix_profile.py \
  --dry-run --allow-dirty
# Synthetic records covered both stage contracts, layers 0/21/42, proposal
# schedule checks, summary aggregation, and Markdown report generation.
DS4_DSPARK_EXACT_LAYER_PROFILE=1 \
DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_ATTN_SUFFIX_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
# Timings ignored; all outputs matched and every candidate stage was present.
git diff --check
```

Phase 0.75 user-run attribution result on 2026-07-14:

- Run: `speed-bench/local-runs/suffix-profile-20260714-130259/stages.csv`.
  All selected layers contained nine matched default/candidate proposal batches
  and 41 proposal rows. Attention-pre and FFN control medians agreed closely,
  so the regression is localized to the suffix transformation rather than a
  different proposal schedule or broad residency shift.
- Layer 0: default serial tail `0.323 ms/row`; candidate core/capture `0.489`,
  projection A `0.112`, projection-B/HC `0.108`, total `0.709 ms/row`
  (`2.192x`, `+119.2%`).
- Layer 21: default serial tail `0.441 ms/row`; candidate core/capture `0.586`,
  projection A `0.118`, projection-B/HC `0.111`, total `0.815 ms/row`
  (`1.848x`, `+84.8%`).
- Layer 42: default serial tail `0.621 ms/row`; candidate core/capture `0.759`,
  projection A `0.119`, projection-B/HC `0.111`, total `0.989 ms/row`
  (`1.594x`, `+59.4%`).
- The candidate core plus head capture is already `0.138` to `0.166 ms/row`
  slower than the entire default tail, even though the default tail also
  includes both serial output projections and HC expansion. The two candidate
  batch projections add another nearly depth-independent `0.220` to
  `0.230 ms/row`. Total candidate overhead is almost constant at `0.368` to
  `0.386 ms/row`; its percentage shrinks only because deeper attention cores
  cost more.
- At the dominant proposal width five, projection A and fused projection-B/HC
  together cost about `0.22 ms/row`. The profile's median overhead, scaled by
  43 layers, `41/9` rows per target evaluation, and the previously measured
  `0.2188` target evaluations per emitted token, predicts about `16 ms` extra
  per emitted token. The uninstrumented ablation observed about `12.5 ms`; the
  agreement is sufficient for a synchronized attribution profile.
- The next narrow experiment is zero-copy head capture: while executing each
  serial attention core, bind `g->heads` directly to that proposal's row in
  `batch_heads`, then remove the separate `ds4_gpu_tensor_copy`. Re-run
  correctness and this attribution profile before another uninstrumented tok/s
  ablation. If core/direct-write plus the two batch projections still cannot
  match the default tail, retire this suffix batching schedule.

Phase 0.76 zero-copy attention-head capture completed on 2026-07-14:

- Changed only the opt-in exact attention-suffix candidate. During each
  `core_only` prepared attention row, `g->heads` is now bound directly to that
  row's view in `batch_heads`. Attention output and inverse RoPE therefore land
  in their final batch storage without an intermediate tensor.
- Removed the verifier's separate per-row `ds4_gpu_tensor_copy` from the serial
  `g->heads` scratch into `batch_heads`. The batch projection and serial
  fallback both continue consuming the same captured rows; normal default exact
  execution is unchanged.
- Renamed the synchronized candidate component from
  `attention_core_capture_serial` to `attention_core_direct_serial` and updated
  the attribution parser/report contract. Old Phase 0.75 raw logs remain valid
  evidence for the pre-change copy path but intentionally do not satisfy the
  new stage contract.
- The normal candidate correctness matrix passed reasoning, Italian,
  medium-context, rolling-window, and resumed-chat cases. The dedicated
  64-token long-generation, rolling-window, resumed-chat, and forced serial-
  fallback soak passed. A profile-enabled layer-42 matrix also passed and real
  logs contained every renamed candidate component.
- The normal Metal build, CPU object build, DSpark validation/shape binding,
  Python compile, dry-run command generation, real-log parsing, and synthetic
  aggregation/reporting passed. The CPU build retained only its existing
  unused-function/parameter warnings.
- Codex did not run the timed attribution profile or any tok/s benchmark.

Phase 0.76 checks:

```sh
make -j4
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_ATTN_SUFFIX_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
./tests/dspark_exact_attention_suffix_batch_runtime_soak.sh
python3 -m py_compile \
  speed-bench/run_dspark_exact_attention_suffix_profile.py
python3 speed-bench/run_dspark_exact_attention_suffix_profile.py \
  --dry-run --allow-dirty
DS4_DSPARK_EXACT_LAYER_PROFILE=1 \
DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER=42 \
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_ATTN_SUFFIX_RUNTIME=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
# Timings ignored; output matched and all renamed candidate stages were parsed.
make ds4_cpu.o ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.76 user-run zero-copy attribution result on 2026-07-14:

- Run: `speed-bench/local-runs/suffix-profile-20260714-131411/stages.csv`.
  As before, every selected layer contained nine matched proposal batches and
  41 rows. Attention-pre and FFN controls remained closely matched, and the
  default serial-tail medians were stable against Phase 0.75.
- Layer 0: default tail `0.324 ms/row`; candidate direct core `0.425`,
  projection A `0.113`, projection-B/HC `0.108`, total `0.646 ms/row`
  (`1.996x`, `+99.6%`). Direct-write saved `0.064 ms/row` from the core stage
  and `0.063 ms/row` from the candidate total.
- Layer 21: default tail `0.432 ms/row`; candidate direct core `0.537`,
  projection A `0.128`, projection-B/HC `0.108`, total `0.773 ms/row`
  (`1.792x`, `+79.2%`). Direct-write saved `0.049 ms/row` from the core stage
  and `0.042 ms/row` from the candidate total.
- Layer 42: default tail `0.619 ms/row`; candidate direct core `0.723`,
  projection A `0.116`, projection-B/HC `0.111`, total `0.950 ms/row`
  (`1.535x`, `+53.5%`). Direct-write saved `0.036 ms/row` from the core stage
  and `0.039 ms/row` from the candidate total.
- The zero-copy change worked, but the remaining candidate overhead is still
  nearly depth-independent at `0.322` to `0.341 ms/row` (mean `0.331`). Scaled
  by 43 layers, `41/9` rows per target evaluation, and the prior `0.2188`
  target evaluations per emitted token, it predicts about `14.2 ms` extra per
  emitted token. This remains consistent with the original uninstrumented
  `12.5 ms/token` regression.
- Retirement decision: do not run another tok/s ablation and do not pursue
  further versions of this deferred small-width suffix schedule. The exact
  batch primitives and observer evidence remain useful research artifacts, but
  `DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH` must remain opt-in and non-production.
  Future attention work should optimize the retained serial tail in place,
  preserving its interleaved row schedule, rather than separating all proposal
  cores from all output projections.

Phase 0.77 retained serial-tail attribution harness completed on 2026-07-14:

- Added an exact-tail specialization to the existing selected decode-stage
  profiler. It activates only when `DS4_METAL_DECODE_STAGE_PROFILE` and
  `DS4_DSPARK_EXACT_TAIL_PROFILE` are both enabled for a prepared serial
  attention tail. Ordinary default execution keeps the existing disabled
  profiler check and does not read the new environment variable.
- The diagnostic preserves each proposal row's operation order and emits six
  synchronized `part=tail` records: `kv_cache_update`,
  `compressor_indexer`, `attention`, `inverse_rope`, `projection_a`, and
  `projection_b_hc`. It does not defer or regroup work across proposal rows.
- When this component profile is active, the exact-layer profiler omits its
  redundant combined `attention_tail_serial` boundary while retaining
  `attention_pre_batch` and `ffn_batch` records as controls. Its FFN timer is
  restarted after the per-row tail boundaries so it does not absorb tail time.
- Added `speed-bench/run_dspark_exact_attention_tail_profile.py`. It runs one
  unprofiled exact reference and selected layer `0/21/42` profiles, requires
  byte-identical output, validates the exact/tail stage contracts, and expands
  every `(start, proposal width)` control record into the expected multiset of
  one-row positions. This handles partial-accept reruns with duplicate starts
  and rejects missing, duplicated, or misplaced tail rows.
- The report gives per-layer median tail totals, all six component medians and
  shares, ranked components, and attention-pre/FFN control medians. Boundaries
  preserve operation order but still synchronize and change scheduling, so the
  numbers are attribution data rather than throughput.
- A profile-enabled layer-42 correctness matrix passed reasoning, Italian,
  medium-context, rolling-window, and resumed-chat cases. A real retained log
  contained every expected stage; its rows matched the proposal expansion and
  the parser/report completed. Synthetic records separately covered duplicate
  proposal starts and layers `0/21/42`.
- The normal Metal build, CPU object build, DSpark validation/shape binding,
  Python compile, and dry-run command generation passed. The CPU build retained
  only its existing unused-function/parameter warnings.
- Codex did not run the selected 8K timed profile or any tok/s benchmark.

Phase 0.77 checks:

```sh
make -j4
python3 -m py_compile \
  speed-bench/run_dspark_exact_attention_tail_profile.py
python3 speed-bench/run_dspark_exact_attention_tail_profile.py \
  --dry-run --allow-dirty
# Synthetic stages covered duplicate proposal starts, all component/control
# contracts, layers 0/21/42, aggregation, ranking, and Markdown reporting.
DS4_DSPARK_EXACT_LAYER_PROFILE=1 \
DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER=42 \
DS4_METAL_DECODE_STAGE_PROFILE=1 \
DS4_METAL_DECODE_STAGE_PROFILE_LAYER=42 \
DS4_DSPARK_EXACT_TAIL_PROFILE=1 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
# Timings ignored; output matched and real exact/tail records passed the parser.
make ds4_cpu.o ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

Phase 0.77 user-run retained-tail attribution result on 2026-07-14:

- Run: `speed-bench/local-runs/tail-profile-20260714-132931/stages.csv`.
  Each selected layer contained nine exact proposal batches and 41 matching
  one-row tail records per component. Attention-pre and FFN controls remained
  stable at about `0.26` and `0.40 ms/row` respectively.
- Layer 0 tail components: KV/cache `0.293`, compressor/indexer `0.030`,
  attention `0.411`, inverse RoPE `0.274`, projection A `0.349`, and fused
  projection-B/HC `0.363 ms/row`. With no attention compression in layers 0-1,
  the near-empty compressor stage provides a useful synchronized floor.
- Layer 21 (ratio-128 compression): KV/cache `0.301`, compressor/indexer
  `0.313`, attention `0.453`, inverse RoPE `0.271`, projection A `0.360`, and
  projection-B/HC `0.358 ms/row`.
- Layer 42 (ratio-4 compression/indexer): KV/cache `0.294`,
  compressor/indexer `0.366`, attention `0.533`, inverse RoPE `0.266`,
  projection A `0.361`, and projection-B/HC `0.363 ms/row`.
- The flat KV, inverse-RoPE, and output-projection medians are not responsible
  for the tail's depth growth. Attention rises by `0.122 ms/row` from layer 0
  to 42. Compression adds roughly `0.28-0.34 ms/row` in every compressed layer
  and therefore deserves separate attribution despite attention being the
  largest single synchronized component.
- Raw position grouping sharpens the compressor result. Layer 21 had 41
  non-emitting rows with compressor median `0.313 ms/row`. Layer 42 had 31
  non-emitting rows at `0.349 ms/row` and 10 ratio-4 emit rows at
  `0.575 ms/row`. Most cost is recurrent compressor projection/update work on
  every token; periodic quantize/commit and indexer work add about
  `0.226 ms/row` on emit rows.
- Next profile compressed layers in place, separating main compressor pair
  projection, recurrent update/emit, ratio-4 indexer compressor projection and
  update, indexer query/weight preparation, score, and top-k. Preserve the full
  512-row indexer selection contract. Choose a kernel optimization only after
  that split; do not use synchronized component totals as throughput values.

Phase 0.78 compressed-tail attribution harness completed on 2026-07-14:

- Added `DS4_DSPARK_EXACT_COMPRESSOR_PROFILE` as a selected-layer diagnostic
  specialization of the prepared serial attention tail. It is read only after
  the existing decode-stage profiler selects a layer and is mutually exclusive
  with the broader retained-tail records. Normal Metal inference is unchanged.
- The diagnostic synchronizes immediately before main compressor projection
  and preserves the existing row-interleaved execution order. It emits
  `main_projection` and `main_update` for ratio-128 and ratio-4 layers. Ratio-4
  layers also emit `indexer_projection` and `indexer_update`; once sparse
  indexed attention is active, they additionally emit `indexer_prepare`,
  `indexer_score`, and `indexer_topk`.
- The `main_update` and `indexer_update` stages include periodic emit work on
  rows where `(position + 1) % ratio == 0`, so the report splits their emit and
  non-emit medians. The score/top-k path retains `DS4_N_INDEXER_TOP_K=512`.
- The exact-layer profiler suppresses its aggregate serial-tail boundary while
  this component mode is active. Because compressor profiling ends before the
  remainder of each row's attention tail, the verifier explicitly flushes the
  final unreported tail before starting its FFN control timer.
- Added `speed-bench/run_dspark_exact_compressor_profile.py`. Its default 8K
  run profiles layer 21 (ratio 128) and layer 42 (ratio 4), requires
  byte-identical output, expands exact proposal signatures to one-row position
  multisets, rejects unknown/missing/duplicated component streams, and reports
  recurrent emit/non-emit groups. Dense ratio-4 rows may omit sparse indexer
  stages; once any query/weight prepare, score, or top-k record appears, all
  three must cover the same valid proposal-row subset. The default generation
  length is 128 so the code fixture crosses from dense to sparse near target
  position 4099. The harness clears inherited sparse-threshold overrides so
  this contract is evaluated against the normal 1024-row Metal threshold.
- The Metal and no-GPU object builds, DSpark validation/shape-binding tests,
  and Python compile/dry run passed. The no-GPU build retained only its existing
  unused-function/parameter warnings. Synthetic records covered
  ratio inference, duplicate proposal positions, emit grouping, all component
  stages, aggregation, and Markdown reporting. A real short layer-42 smoke
  retained the expected main/indexer projection/update records for all three
  proposal rows and matched baseline output byte-for-byte. Its 120-token
  context correctly remained below the 1024-row sparse threshold, so it was
  not used to validate or time prepare/score/top-k.
- The profile-enabled correctness matrix passed reasoning, Italian,
  medium-context, rolling-window, and resumed-chat output identity. Timings
  were ignored. Codex did not run the selected 8K attribution profile or any
  tok/s benchmark.

Phase 0.78 user command:

```sh
python3 speed-bench/run_dspark_exact_compressor_profile.py --dry-run
python3 speed-bench/run_dspark_exact_compressor_profile.py --confirm-ready
```

Phase 0.78 preliminary user-run dense-compressor result on 2026-07-14:

- Retained run:
  `speed-bench/local-runs/compressor-profile-20260714-134837`. Every profiled
  output matched the exact reference. Both layers contained nine proposal
  batches and 41 matching component rows; controls remained stable at about
  `0.27 ms/row` for attention preparation and `0.40 ms/row` for FFN.
- Layer 21 (ratio 128): main projection median `0.281 ms/row`; non-emitting
  recurrent update `0.245 ms/row`. This proposal sample contained no ratio-128
  emit row.
- Layer 42 (ratio 4, dense mode): main projection `0.296 ms/row`; main update
  `0.236` non-emit and `0.417 ms/row` emit. Indexer compressor projection was
  `0.281 ms/row`; indexer update was `0.247` non-emit and `0.372 ms/row` emit.
- These synchronized components must not be added into a production timing,
  but projection and recurrent update are comparable within each compressor
  pair. Ratio-4 layers execute both the main and indexer pairs on every row;
  periodic emit work adds a visible but smaller surcharge.
- The original harness stopped after layer 42 because it incorrectly required
  sparse stages on every row. The prompt begins near target position 3997 with
  this tokenizer, and 32 generated tokens do not cross the strict
  `n_comp > 1024` threshold. No engine stage was missing. The parser now accepts
  valid dense output and validates sparse prepare/score/top-k as a shared row
  subset. The default token count was raised to 128, which should cross near
  position 4099.
- The remaining user-run follow-up can reuse the layer-21 result and profile
  only layer 42:

```sh
python3 speed-bench/run_dspark_exact_compressor_profile.py \
  --layers 42 --confirm-ready
```

Phase 0.78 user-run dense-to-sparse compressor result on 2026-07-14:

- Run: `speed-bench/local-runs/compressor-profile-20260714-141408/stages.csv`.
  Layer 42 contained 57 matched proposal batches and 238 component rows. Sparse
  indexed attention began at target position 4099 and covered 37 row
  occurrences across 24 unique positions through 4123. All sparse prepare,
  score, and top-k signatures matched exactly, and profiled output matched the
  exact reference byte-for-byte.
- Controls remained stable: attention preparation `0.262 ms/row` and FFN
  `0.395 ms/row`. Main and indexer recurrent components also remained close
  across the transition: main projection was `0.303/0.318`, main update
  `0.243/0.255`, indexer projection `0.276/0.295`, and indexer update
  `0.248/0.261 ms/row` for dense/sparse rows respectively.
- Overall component medians were main projection `0.304`, main update `0.238`
  non-emit and `0.412` emit, indexer projection `0.278`, and indexer update
  `0.241` non-emit and `0.376 ms/row` emit. These agree with the preliminary
  32-token dense run and show no attribution discontinuity at sparse entry.
- Sparse-only medians were query/weight preparation `0.405`, score `0.333`,
  and full 512-row top-k `0.329 ms/row`. Their synchronized costs are close;
  there is no evidence for weakening the 512-row selection contract or for
  treating one sparse kernel as the sole bottleneck.
- The best broad next candidate is the existing opt-in
  `DS4_METAL_COMPRESSOR_PAIR_NR4`. It changes the paired F16 projection from
  two to four output rows per threadgroup for 512/1024-wide outputs. Those are
  exactly the ratio-128 and ratio-4 main compressor widths, so it applies to
  every compressed layer in dense and sparse contexts. The ratio-4 indexer
  compressor is 256-wide and remains unchanged, as do score, full top-k, and
  attention semantics.
- Next validate `DS4_METAL_COMPRESSOR_PAIR_NR4=1` against the exact correctness
  matrix, then add a direct uninstrumented paired ablation against default
  exact DSpark. Do not infer a throughput win from this synchronized profile;
  the user must run the ablation.

Phase 0.79 NR4 correctness and throughput gate prepared on 2026-07-14:

- Extended `tests/dspark_gpu_candidates_correctness.sh` with
  `DS4_TEST_DSPARK_COMPRESSOR_PAIR_NR4=1`. The script captures and unsets any
  inherited NR4 variable before launching models, leaves every baseline on the
  default paired projection, and enables NR4 only in the exact DSpark candidate
  environment. The existing reasoning, Italian, medium-context,
  rolling-window, and resumed-chat byte comparisons therefore form a direct
  default-versus-NR4 correctness gate.
- Added `--compressor-pair-nr4-ablation` to
  `speed-bench/run_dspark_comparison.py`. It compares `default_exact` against
  `compressor_pair_nr4`, forces Metal explicitly, scrubs inherited NR4 from
  both modes, enables it only for the candidate, and disables all diagnostics
  and runtime stats. Warmup/measured outputs must remain byte-identical, pair
  order alternates, and the report gives both median ratios and candidate
  delta.
- Python syntax, shell syntax, dry-run command generation, environment
  isolation, mode selection, synthetic paired aggregation/reporting, malformed
  correctness-option rejection, and `git diff --check` passed. Existing
  comparison modes remain available and mutually exclusive with NR4.
- Codex could not execute the Metal correctness matrix in this managed session:
  even the control process reported `Metal device not available`. No timed
  benchmark was run. The user must run correctness first; run throughput only
  if all five cases pass.

Phase 0.79 user gates:

```sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_COMPRESSOR_PAIR_NR4=1 \
  ./tests/dspark_gpu_candidates_correctness.sh

python3 speed-bench/run_dspark_comparison.py \
  --dry-run --compressor-pair-nr4-ablation
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --compressor-pair-nr4-ablation
```

Phase 0.79 user-run NR4 correctness result on 2026-07-14:

- The candidate-only matrix passed reasoning, Italian, medium-context,
  rolling-window, and resumed two-turn chat. The final status was
  `DSpark GPU candidate correctness matrix (runtime): PASS`.
- Each baseline used the default paired projection while each exact DSpark
  candidate used `DS4_METAL_COMPRESSOR_PAIR_NR4=1`, so this establishes output
  identity across short, medium, rolling-window, and resumed-state execution.
- NR4 remains experimental until the user-run uninstrumented paired ablation
  demonstrates a repeatable throughput improvement. Correctness alone is not
  grounds for promoting it to the default.

Phase 0.79 user-run NR4 throughput result on 2026-07-14:

- Run: `speed-bench/local-runs/20260714-231153/results.csv`.
  Default exact median was `21.62 t/s`; compressor-pair NR4 median was
  `21.53 t/s`. The ratio of medians was `0.9958x`, median paired ratio was
  `0.9944x`, and the reported candidate delta was `-0.4%` across three pairs.
- Pair ratios were `0.9935x`, `0.9944x`, and `1.0051x`. Two pairs favored the
  default by about `0.6%`, while one favored NR4 by about `0.5%`; alternating
  order did not reveal a hidden candidate advantage.
- Retirement decision: keep `DS4_METAL_COMPRESSOR_PAIR_NR4` opt-in as a correct
  research control, but do not promote or benchmark it again. Widening the main
  paired projection is not a throughput improvement on this Metal system.
- The synchronized compressor split was useful for ruling out easy targets,
  but its per-boundary projection costs did not predict production sensitivity.
  Return to the retained serial attention operation, the largest tail component
  in Phase 0.77, and distinguish dense versus sparse indexed-attention cost
  before selecting another kernel change.

Phase 0.80 retained-attention transition profile prepared on 2026-07-14:

- Added a selected-layer Metal boundary around only the authoritative retained
  attention call. Each one-row record is labeled from the branch actually used:
  `raw`, `dense_mixed`, or `sparse_indexed`. The boundary starts after all
  compressor/indexer work and ends before inverse RoPE and output projection,
  so every mode has the same immediate before/after scope.
- The new `DS4_DSPARK_EXACT_ATTENTION_PROFILE` diagnostic shares the existing
  exact-layer and Metal-layer selectors. It is mutually exclusive with the
  broader exact-tail and compressor component profiles; compressor has first
  precedence, followed by attention mode, then full tail. Normal execution
  does not read the new variable unless selected-layer stage profiling is
  already active.
- The exact verifier suppresses its aggregate serial-tail boundary while this
  component profile is active and flushes the unreported inverse-RoPE/output
  remainder before starting the FFN control. This prevents partial tail work
  from leaking into `ffn_batch` without changing row-interleaved operation
  order or model state.
- Added `speed-bench/run_dspark_exact_attention_transition_profile.py`. Its
  default profiles layer 42 for 128 generated tokens on the 8K code fixture,
  clears any inherited sparse-threshold override, preserves the default full
  512-row top-k contract, and requires byte-identical stdout against an
  unprofiled exact DSpark reference.
- The parser rejects unknown modes, non-row attention records, mismatched
  attention-pre/FFN schedules, and any combined attention-mode multiset that
  differs from the exact proposal rows. It deliberately uses multisets so
  repeated positions from speculative retries remain valid. By default it also
  fails unless at least one selected layer contains both dense and sparse rows;
  `--allow-single-mode` exists for explicit short diagnostic runs.
- `make -j4 ds4 ds4_test ds4_cpu.o`, Python syntax, dry-run command generation,
  synthetic duplicate-position/mode parsing, unknown-mode rejection,
  `./ds4_test --dspark-validation --dspark-shape-binding`, and
  `git diff --check` passed. A correctness-only runtime matrix with the new
  profile enabled passed reasoning, Italian, medium-context, rolling-window,
  and resumed two-turn chat.
- A separate six-token Metal check compared unprofiled exact DSpark against the
  selected-layer profile byte-for-byte. The real parser matched ten layer-42
  proposal rows, all `dense_mixed`, as expected for the short run. Those timing
  values were ignored. Codex did not run the 128-token transition attribution
  or make any performance claim.

Phase 0.80 user-run gate:

```sh
python3 speed-bench/run_dspark_exact_attention_transition_profile.py \
  --dry-run
python3 speed-bench/run_dspark_exact_attention_transition_profile.py \
  --confirm-ready
```

The result should contain both `dense_mixed` and `sparse_indexed` rows. Compare
their synchronized medians only within this run; do not compare its t/s or
absolute component values with uninstrumented generation.

Phase 0.80 user-run retained-attention result on 2026-07-14:

- Run: `speed-bench/local-runs/attention-transition-20260714-234148/stages.csv`.
  Layer 42 contained 57 exact proposal batches and 238 proposal-row
  occurrences: 201 dense mixed rows and 37 sparse indexed rows. Sparse entry
  began at position 4099, matching the earlier compressor/indexer transition.
- Dense mixed attention had median `0.504 ms/row`; sparse indexed attention had
  median `0.722 ms/row`. The sparse/dense ratio was `1.433x`, a `43.3%`
  synchronized increase, or about `0.218 ms` additional attention time per
  sparse proposal row at this context length.
- This was not an outlier-driven result. Dense p10/p90 was `0.482/0.531 ms`
  with range `0.417..0.637`; sparse p10/p90 was `0.706/0.756 ms` with range
  `0.681..0.840`. The slowest dense row was still faster than the fastest
  sparse row. Repeated speculative positions were represented in both modes
  and did not blur the separation.
- Attention-pre and FFN controls were `0.262` and `0.391 ms/row`, nearly
  identical to the Phase 0.78 transition controls (`0.262/0.395`). The current
  dense attention median also agrees with the earlier layer-42 tail attribution
  (`0.533 ms/row`) closely enough to support the profile's localization.
- Combined only as a rough synchronized attribution, Phase 0.78's sparse-only
  prepare/score/top-k medians sum to `1.067 ms/row`, and sparse attention adds
  another `0.218 ms/row` over dense attention. Do not treat the `1.285 ms` sum
  as production latency, but it shows sparse entry has both an indexer-pipeline
  cost and a separate indexed-attention penalty.
- The indexed path preserves all 512 selected rows but scans only those rows;
  at this transition it is nevertheless slower than dense attention over about
  1025 compressed rows. The penalty is therefore implementation overhead from
  indexed gathering/staging/scheduling, not excess selected-row count.
- This kernel is shared by every ratio-4 layer (the Flash layout uses ratio 4
  on even layers 2 through 42), so a real decode-kernel improvement would be a
  general long-context improvement rather than a layer-42 benchmark special.
- Source/history audit: current one-token indexed decode already uses
  `kernel_dsv4_indexed_mixed_attention_heads8_rb16`, which stages 16 rows per
  block. The earlier RB4 candidate was removed in upstream commit `18c2d4b` as
  part of rejected experimental-path cleanup. Do not re-run RB4. The next
  candidate should improve the retained RB16 decode specialization while
  preserving row order, online-softmax arithmetic, and all 512 selected rows.

Phase 0.81 RB16-direct indexed-attention candidate prepared on 2026-07-14:

- Added opt-in Metal kernel
  `kernel_dsv4_indexed_mixed_attention_heads8_rb16_direct`, selected only by
  `DS4_METAL_INDEXED_ATTN_RB16_DIRECT=1`. The existing RB16 kernel and default
  host route are unchanged.
- The candidate retains 16-row staging, raw-window traversal, all 512 selected
  rows, selected-row order, F16 Q/K/V rounding, and the exact online-softmax
  update sequence. It removes the per-thread `uint rows[16]` scan/array and
  loads each selected row id at its cooperative K/V copy site.
- The host uses the candidate only for one-token decode, exactly 512 selected
  rows, and `visible_rows >= n_comp`. Together with `top_k <= n_comp` and the
  argsort contract, that proves every selected id is a valid visible cache-row
  id. If any guard fails, execution remains on current RB16.
- Added opt-in trace `DS4_METAL_INDEXED_ATTN_RB16_DIRECT_TRACE=1` for
  correctness only. It emits one route record per process and is never enabled
  by the timed runner.
- Extended `tests/dspark_gpu_candidates_correctness.sh` with
  `DS4_TEST_DSPARK_INDEXED_ATTN_RB16_DIRECT=1`. Besides the normal five cases,
  it generates a temporary long prompt, lowers the implementation threshold to
  64 while preserving the immutable 512-row selector frontier, and compares
  base, default-RB16 exact DSpark, and RB16-direct exact DSpark output. The
  candidate route was observed at position 2679 with 670 compressed rows and
  512 selected rows. All three outputs were byte-identical; the complete
  reasoning, Italian, medium-context, rolling-window, resumed-chat, and forced
  sparse matrix passed.
- Added `--indexed-attention-rb16-direct-ablation` to
  `speed-bench/run_dspark_comparison.py`. It runs paired uninstrumented default
  exact versus candidate execution, clears inherited candidate/trace/threshold
  settings, disables runtime stats and diagnostics, alternates order, and
  requires every output to remain byte-identical. The benchmark must explicitly
  use the 8K transition fixture because the runner's normal short fixture never
  enters sparse attention.
- Build, Metal runtime kernel compilation, shell/Python syntax, dry-run command
  generation, forced-sparse route selection, candidate/default output identity,
  malformed correctness-option rejection, and `git diff --check` passed. Codex
  did not run a tok/s ablation or make a performance claim.

Phase 0.81 user-run throughput gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --indexed-attention-rb16-direct-ablation \
  --prompt-file speed-bench/issue468/code_8k.txt --ctx 16384 --tokens 128
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --indexed-attention-rb16-direct-ablation \
  --prompt-file speed-bench/issue468/code_8k.txt --ctx 16384 --tokens 128
```

Do not benchmark the default short fixture: it cannot exercise this candidate.

Phase 0.81 user-run RB16-direct throughput result on 2026-07-15:

- Run: `speed-bench/local-runs/20260715-001221/results.csv`. Default exact
  median was `10.31 t/s`; RB16-direct median was `10.43 t/s`. The ratio of
  medians and median paired ratio were both `1.0116x`, for a reported `+1.2%`
  candidate delta across three pairs.
- Pair ratios were `1.0116x`, `1.0116x`, and `1.0136x`. Every pair favored the
  candidate, including the reversed-order second pair. Default/candidate values
  were `10.31/10.43`, `10.31/10.43`, and `10.27/10.41 t/s`; all measured
  outputs had the same SHA-256.
- The process snapshot showed substantial unrelated activity, including Codex
  renderer processes and other desktop services. Absolute t/s remains noisy,
  but the alternating order and narrow `1.16..1.36%` paired range make the
  direction more credible than the retired NR4 result, whose pairs crossed
  both sides of parity.
- Gate decision: RB16-direct is a successful candidate, but keep it opt-in for
  one more attribution check before promotion. The next phase should add an
  explicit candidate mode to the existing synchronized attention-transition
  profile and verify that the sparse indexed median falls from Phase 0.80's
  `0.722 ms/row` while dense/control values remain stable. If that localization
  passes, promote the guarded route to default and retain current RB16 as the
  fallback/control.

Phase 0.82 RB16-direct synchronized attribution prepared on 2026-07-15:

- Extended `run_dspark_exact_attention_transition_profile.py` with an explicit
  `--rb16-direct-comparison` mode. It runs an unprofiled exact reference, the
  default RB16 transition profile, and the opt-in RB16-direct transition
  profile. `DS4_METAL_INDEXED_ATTN_RB16_DIRECT=1` is set only for the candidate
  process; inherited candidate and sparse-threshold overrides remain cleared.
- Every profiled stdout must match the unprofiled reference byte-for-byte. The
  comparison additionally requires default and candidate to have identical
  proposal schedules and identical per-position dense/sparse branch labels.
  This prevents a changed speculative schedule or transition point from being
  mistaken for a kernel timing improvement.
- The synchronized report compares default/direct medians for dense mixed and
  sparse indexed attention separately. It also reports attention-pre and FFN
  controls in both processes. Only sparse indexed attention uses RB16-direct;
  dense and control values are noise/scheduling controls for the attribution.
- The original single-variant profiler invocation and report remain available
  unchanged. Comparison CSV records carry an explicit `variant` column, and
  metadata records both fully expanded commands.
- Python syntax, the original and comparison dry-run command paths, synthetic
  matched-profile summary/report generation, deliberate branch-label mismatch
  rejection, `make -j4 ds4 ds4_test ds4_cpu.o`,
  `./ds4_test --dspark-validation --dspark-shape-binding`, and
  `git diff --check` passed. Codex did not execute the synchronized profile or
  use its timing output; that remains the user-run gate below.

Phase 0.82 user-run gate:

```sh
python3 speed-bench/run_dspark_exact_attention_transition_profile.py \
  --dry-run --rb16-direct-comparison
python3 speed-bench/run_dspark_exact_attention_transition_profile.py \
  --confirm-ready --rb16-direct-comparison
```

The useful result is the candidate/default sparse indexed ratio. Compare it
with the dense ratio and attention-pre/FFN controls from this same run, not
with absolute synchronized values from Phase 0.80. If sparse attention alone
improves coherently with the `+1.2%` uninstrumented result, promote the guarded
RB16-direct route to default while retaining current RB16 as fallback/control.

Phase 0.82 user-run synchronized attribution result on 2026-07-15:

- Run: `speed-bench/local-runs/attention-transition-rb16-direct-20260715-003733`.
  Layer 42 had identical default/direct schedules with 57 proposal batches,
  238 proposal rows, 201 dense rows, and 37 sparse rows. The unprofiled
  reference and both profiled variants had the same stdout SHA-256.
- Default/direct dense medians were `0.502/0.504 ms/row` (`1.004x`, `+0.4%`),
  while sparse indexed medians were `0.727/0.624 ms/row` (`0.858x`, `-14.2%`).
  This localizes the candidate improvement to the branch that actually uses
  RB16-direct.
- The sparse distributions separated cleanly: default p10/p90 was
  `0.710/0.733 ms/row`, while direct p10/p90 was `0.607/0.632 ms/row`. The
  candidate p90 remained below the default p10. Full ranges overlapped only at
  outliers (`0.701..0.871` default, `0.597..0.723` direct).
- Attention-pre was `0.2632/0.2644 ms/row` and FFN was
  `0.3906/0.3926 ms/row`, both about `+0.5%` in the candidate process. Their
  matched movement, together with the stable dense path, is a scheduling/noise
  control rather than evidence of a broader runtime change.
- Metadata recorded clean commit `601bfcd`, no inherited cleared environment
  variables, and the candidate switch only in the RB16-direct process. The
  output hashes and runner checks confirm identical output, proposal schedule,
  and per-position dense/sparse routing.
- Gate decision: Phase 0.82 passes. Combined with the prior correct candidate
  matrix and consistent `+1.2%` uninstrumented paired result, promote the
  guarded RB16-direct route to the default one-token full-visibility path.
  Retain current RB16 behind an explicit control/fallback route so promotion
  can be validated and reversed without deleting the known-correct kernel.

Phase 0.83 RB16-direct default promotion prepared on 2026-07-15:

- Promoted `kernel_dsv4_indexed_mixed_attention_heads8_rb16_direct` to the
  environment-free default only when all proven guards hold: one-token decode,
  exactly 512 selected compressed rows, and `visible_rows >= n_comp`. The
  direct pipeline is now created and required during normal Metal
  initialization.
- Legacy RB16 remains the automatic one-token fallback whenever any direct
  guard fails. `DS4_METAL_INDEXED_ATTN_RB16_LEGACY=1` explicitly forces that
  legacy route even when the direct guard passes, providing a reversible
  correctness/performance control without deleting the known-correct kernel.
  Multi-token indexed attention remains on the general kernel.
- Updated the forced-sparse correctness case to compare base output, explicit
  legacy RB16, and the environment-free promoted default. The candidate trace
  is now a promoted-route trace; the same generated long prompt and lowered
  implementation threshold still preserve the immutable 512-row selector
  frontier. `DS4_TEST_DSPARK_INDEXED_ATTN_RB16_PROMOTION=1` is the new switch;
  the old `...RB16_DIRECT` test spelling remains a compatibility alias.
- Reoriented the uninstrumented comparison runner around promotion semantics.
  `--indexed-attention-rb16-promotion-ablation` compares explicit legacy RB16
  against promoted default, alternates order, disables diagnostics/stats, and
  requires byte-identical output. The old direct-ablation spelling remains an
  alias but no longer implies that direct is opt-in.
- Reoriented the synchronized transition runner in the same way. Its normal
  single-mode profile now records the promoted default; promotion comparison
  runs explicit legacy first and default direct second, reporting
  promoted/legacy ratios. The old direct-comparison spelling remains an alias.
- `make -j4 ds4 ds4_test ds4_cpu.o`, Metal pipeline compilation, shell/Python
  syntax, new and compatibility dry-run command generation, synthetic
  promotion summary/report checks, mismatched transition rejection,
  malformed promotion-option rejection, and
  `./ds4_test --dspark-validation --dspark-shape-binding` passed.
- The full exact-runtime correctness matrix with the promotion switch passed
  reasoning, Italian, medium-context, rolling-window, and resumed-chat cases.
  Its forced-sparse extension passed base/legacy/promoted byte identity and
  observed the direct route only in the promoted process. Timings were ignored;
  Codex did not run the final throughput confirmation.

Phase 0.83 user-run final confirmation:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --indexed-attention-rb16-promotion-ablation \
  --prompt-file speed-bench/issue468/code_8k.txt --ctx 16384 --tokens 128
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --indexed-attention-rb16-promotion-ablation \
  --prompt-file speed-bench/issue468/code_8k.txt --ctx 16384 --tokens 128
```

This should reproduce the earlier direct-over-legacy `+1.2%` direction with
the candidate now represented by the environment-free default. It is the last
throughput gate for this promotion, not a request for more synchronized
attribution.

Phase 0.83 user-run final confirmation on 2026-07-15:

- Run: `speed-bench/local-runs/20260715-005708/results.csv`, on clean promotion
  commit `e185e3f`. Explicit legacy RB16 had median `10.31 t/s`; the promoted
  environment-free RB16-direct default had median `10.40 t/s`. The ratio of
  medians was `1.0087x`, median paired ratio was `1.0097x`, and the reported
  promotion delta was `+0.9%` across three pairs.
- Every pair favored the promoted route: `1.0097x`, `1.0107x`, and `1.0087x`.
  The second pair ran promoted first and legacy second, so alternating order
  did not expose a hidden ordering effect. The narrow paired range was
  `+0.87..+1.07%`.
- All six measured stdout streams had the same SHA-256. Metadata recorded no
  inherited DS4 environment, no thermal warning before or after, explicit
  `DS4_METAL_INDEXED_ATTN_RB16_LEGACY=1` only on the control command, and no
  route environment on promoted default.
- The process snapshot still contained unavoidable desktop activity, so the
  absolute t/s values are not treated as machine-isolated measurements. The
  consistent paired direction nevertheless agrees with Phase 0.81's `+1.2%`
  opt-in result and Phase 0.82's `-14.2%` synchronized sparse-kernel result.
- Final decision: the promotion is accepted. Keep RB16-direct as the guarded
  default, keep legacy RB16 only as automatic fallback and explicit research
  control, and do not benchmark this optimization again unless its guard or
  kernel semantics change.

Phase 0.84 current-default Issue 468 rebaseline prepared on 2026-07-15:

- Before rerunning the three long prompts, audited
  `run_dspark_issue468_comparison.py` and found its environment filter narrower
  than its documentation: it removed `DS4_DSPARK_*` and diagnostic names but
  could inherit Metal route controls such as legacy RB16, NR4 compressor
  pairing, or a sparse-threshold override.
- Strengthened this authoritative corpus runner to remove every inherited
  `DS4_*` key from every child process. Baseline children receive no DS4
  environment; ordinary exact-runtime children then receive only
  `DS4_DSPARK_GPU_RUNTIME=1` and `DS4_DSPARK_MULTI_COMMIT=1`. Optional fast,
  exact-head, and stats keys are added only when their explicit runner options
  request them.
- Metadata still captures the inherited DS4 environment and actual removed
  keys, and now records a machine-readable child-environment policy with
  baseline, ordinary runtime, and optional runtime key lists.
- Python syntax, ordinary and stats dry runs, and `git diff --check` passed.
  Synthetic environment tests injected legacy
  RB16, NR4, sparse-threshold, diagnostic, unrelated Metal, and future unknown
  DS4 keys. Baseline retained none; ordinary runtime retained exactly its two
  required keys; an explicitly instrumented synthetic mode retained only the
  five requested runtime/fast/head/stats keys.
- A poisoned-environment dry run printed all three baseline/runtime command
  pairs without exposing any injected key. Codex did not execute the long
  throughput benchmark or a stats pass.

Phase 0.84 user-run throughput gate:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py --dry-run
python3 speed-bench/run_dspark_issue468_comparison.py --confirm-ready
```

Run the paired uninstrumented comparison without `--stats-pass` first. This
refreshes the current default across `code_8k`, `synthesis_8k`, and
`grounded_8k` after the promoted exact FFN, attention preparation, and
RB16-direct changes. Use the result to choose the next bottleneck; do not add
another local optimization based on the stale Phase 0.47 throughput table.

Phase 0.84 user-run current-default rebaseline on 2026-07-15:

- Run: `speed-bench/local-runs/issue468-20260715-011255/throughput.csv`, on
  clean harness commit `326f43c`. The metadata confirmed no inherited DS4
  environment and the intended child policy: no DS4 keys for baseline, only
  GPU runtime and multi-commit for exact DSpark. No throughput instrumentation
  or stats were enabled.
- `code_8k` measured `21.54/9.95 t/s` baseline/DSpark with paired median
  `0.4619x` (`-53.8%`). Its three paired ratios were
  `0.4617/0.4621/0.4619x`.
- `synthesis_8k` measured `22.06/12.27 t/s` with paired median `0.5558x`
  (`-44.4%`). Its pairs were `0.5547/0.5558/0.5576x`.
- `grounded_8k` measured `20.25/10.32 t/s` with paired median `0.5096x`
  (`-49.0%`). Its pairs were `0.5096/0.5079/0.5102x`.
- Every prompt's reversed-order second pair agreed with the other two, and all
  six measured outputs per prompt had one SHA-256. Unavoidable desktop activity
  remained substantial, but the narrow within-prompt ranges make the relative
  result stable.
- The aggregate paired median improved from Phase 0.47's `0.4762x` to
  `0.5096x`: about `+3.34` ratio points, a `7.0%` relative improvement in
  DSpark/baseline throughput ratio. Per-prompt ratio improvements versus the
  old run were about `6.9%` code, `7.7%` synthesis, and `7.0%` grounded.
- The result is still not competitive: aggregate DSpark remains `49.0%` below
  baseline, `43.4` percentage points behind published MTP K=2, and `6.0`
  points behind published MTP K=5. The promoted optimizations are real but do
  not resolve the long-context exact-verifier cost.
- Decision: collect one separate instrumented exact-runtime sample per prompt
  before choosing another optimization. Do not rerun the full throughput table
  just to obtain stats. Add a stats-only runner mode that executes one fresh
  baseline reference and one stats-enabled exact DSpark process per prompt,
  requires byte-identical output, and reports the existing accepted-depth,
  target-eval, sidecar, and fallback fields.

Phase 0.85 Issue 468 stats-only attribution prepared on 2026-07-15:

- Added `--stats-only` to `run_dspark_issue468_comparison.py`, mutually
  exclusive with the existing combined `--stats-pass`. It executes exactly six
  child processes: one fresh uninstrumented baseline reference followed by one
  stats-enabled exact DSpark runtime for each of the three pinned prompts. It
  uses no warmups or throughput pairs.
- Every instrumented runtime stdout must match its fresh prompt-specific
  baseline byte-for-byte through the existing executor contract. Stats-only
  child environments retain the Phase 0.84 isolation policy: baseline receives
  no DS4 keys; runtime receives GPU runtime, multi-commit, and runtime stats.
- Stats-only output goes to `issue468-stats-<timestamp>` and contains
  `runs.csv`, `stats.csv`, `summary.json`, `summary.md`, start/final metadata,
  and raw stdout/stderr. It deliberately does not create `throughput.csv` or
  report instrumented t/s as performance evidence.
- Expanded the stats summary for bottleneck selection. Per prompt it reports
  accepted depth, target evals per emitted token, positions per eval, target
  milliseconds per eval and emitted token, generation sidecar milliseconds per
  emitted token, their accounted sum, sidecar bridge/stage/head/chain breakdown,
  prefill sidecar time, avoided target evals, batch outcomes, source fallbacks,
  fast-verifier outcomes, and accepted-depth counts.
- Metadata records `execution_mode=stats_only`, zero warmups/pairs, and the
  actual baseline-reference/stats-runtime commands. The old `--stats-pass`
  path remains available and unchanged for combined runs.
- Python syntax, stats-only dry-run command generation, poisoned-environment
  isolation, stats-only/stats-pass mutual-exclusion rejection,
  `git diff --check`, synthetic summary/report arithmetic, exact
  baseline/runtime call
  order, output-reference propagation, CSV/JSON/Markdown creation, and absence
  of `throughput.csv` passed. Codex did not run the real stats attribution.

Phase 0.85 user-run stats gate:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py \
  --confirm-ready --stats-only
```

Use this diagnostic to compare target-verifier and sidecar milliseconds per
emitted token across all three long prompts. Do not interpret its printed
instrumented t/s as throughput and do not begin another optimization until the
three attribution rows are available.

Phase 0.86 Issue 468 stats-only attribution on 2026-07-15:

- The user ran the Phase 0.85 gate. Raw results are in
  `speed-bench/local-runs/issue468-stats-20260715-100900`. The checkout was the
  clean Phase 0.85 commit, all child environments matched policy, and each
  instrumented runtime output matched its fresh uninstrumented baseline.
- Long-prompt accepted depth was much lower than the earlier short-context
  result: `2.246` for `code_8k`, `2.909` for `synthesis_8k`, and `2.667` for
  `grounded_8k`, versus `4.923` in the 64-token short prompt. Full/partial batch
  outcomes were `10/28`, `14/19`, and `10/24`; there were no verifier, source,
  or fast-path fallbacks. This is genuine proposal rejection, not fallback
  overhead.
- Exact target verification dominated accounted generation cost. Target and
  sidecar milliseconds per emitted token were `92.662 + 7.827` for code,
  `75.249 + 6.029` for synthesis, and `90.234 + 6.601` for grounded. Target
  work was therefore `92.2%`, `92.6%`, and `93.2%` of accounted time. The
  accounted totals imply `9.95/12.30/10.33 t/s`, effectively reconciling the
  separate uninstrumented `9.95/12.27/10.32 t/s` results.
- Exact verification processed `2.078/1.734/1.922` target positions per emitted
  token (`266/222/246` positions for 128 outputs). Each target position cost
  `44.589/43.387/46.951 ms`. The corresponding uninstrumented baseline costs
  were `46.425/45.331/49.383 ms` per token, so batching multiple verifier
  positions currently provides almost no per-position efficiency over normal
  one-token target decode.
- This establishes two independent requirements. Better acceptance is needed
  to reduce positions per emitted token, but even the theoretical floor of
  roughly one target position per emitted token would not break even at the
  current target-position and sidecar costs. Break-even positions per emitted
  token are only `0.866/0.906/0.911`. Conversely, at current acceptance the
  target cost per position would need to fall by about `58%/48%/53%`. A useful
  runtime will need both better long-context proposal acceptance and materially
  more efficient exact target verification (or a new correctness-preserving
  verifier architecture).
- Decision: stop optimizing the sidecar for now; its entire `6.0-7.8 ms/token`
  is too small to explain or recover the regression. The next phase is a
  correctness-preserving acceptance-quality audit. It should determine whether
  the long-context depth collapse is expected model behavior or a DSpark
  integration error by attributing rejection depth and proposal/target token
  disagreement without changing token authority. Use that evidence before
  choosing between proposal-path fixes and another exact-verifier architecture.

Phase 0.87 paper-aligned acceptance audit prepared on 2026-07-15:

- Re-read the official DSpark paper (`arXiv:2607.05147v1`), current DeepSpec
  evaluator, and released DeepSeek-V4-Flash-DSpark inference/config artifacts.
  The paper defines accepted length as accepted draft tokens plus one
  target-generated bonus token per verification round. DeepSpec computes the
  same quantity as `acceptance_length_sum / proposal_count`. The existing ds4
  `avg_depth` is emitted tokens per internal runtime cycle and is not the same
  metric because ds4 emits the target bonus on the following loop iteration.
- Added opt-in `DS4_DSPARK_ACCEPTANCE_AUDIT=1` aggregation to the exact runtime.
  It adds no model evaluations and emits one session-end record containing
  paper-aligned proposal, proposed-draft, accepted-draft, accepted-length, and
  full-accept totals. It also records each position's proposed, reached,
  accepted, and first-rejected counts; conditional and cumulative confidence
  sums/Brier terms; and non-finite confidence counts.
- Position-wise conditional acceptance is
  `P(position accepted | all earlier positions accepted)`. Prefix survival is
  `P(all positions through this one accepted)` and matches the current DeepSpec
  evaluator's `accept_rate@position` calculation. Both are reported because
  the paper's Figure 2 prose discusses the former while the released evaluator
  directly emits the latter.
- Capacity- or EOS-truncated final proposals are counted separately and
  excluded from paper-aligned aggregates. This prevents the 128-token output
  boundary from making a five-position proposal look like a quality failure.
  The audit refuses the known-unsafe fast verifier and the runner clears all
  inherited `DS4_*` state.
- Added `--acceptance-audit` to the Issue 468 runner. It executes one fresh
  uninstrumented baseline and one exact audited runtime per prompt, requires
  byte-identical output, runs no warmups or throughput pairs, and deliberately
  omits throughput results. Outputs are `runs.csv`, `acceptance.csv`,
  `positions.csv`, `summary.json`, `summary.md`, metadata, and raw child logs.
- Embedded DSpark Table 1 as the official directional reference. Across the
  released Qwen3/Gemma4 checkpoints, individual benchmark-cell ranges are
  `4.89-6.21` math, `4.51-5.64` code, and `2.92-3.72` chat. The per-target code
  domain macros form a tighter `5.09-5.28` accepted-length target; with seven
  draft tokens plus one bonus, its normalized verify-rate range is
  `0.636-0.660`. `code_8k` is mapped directionally to code; synthesis and
  grounded remain deliberately unmapped.
- This is not a matched Table 1 reproduction. The paper used Qwen3/Gemma4,
  seven draft tokens, temperature `1.0` rejection sampling, non-thinking mode,
  nine named public datasets, and a disabled confidence scheduler. This audit
  uses V4-Flash, five draft tokens, temperature `0`, and custom 8K prompts.
  Report the official values as targets with this caveat, not as a strict
  implementation pass/fail threshold.
- Confidence columns are raw sigmoid outputs from the checkpoint. The paper
  applies post-hoc Sequential Temperature Scaling for scheduling, but no STS
  calibration parameters are present in the released V4 inference config or
  applied by ds4. Use confidence here to diagnose ranking/calibration shape,
  not to validate the paper's calibrated production scheduler.
- C build, Python syntax, shell syntax, poisoned-environment isolation,
  argument incompatibility rejection, synthetic parser/invariant/report and
  end-to-end artifact tests passed. The exact Metal correctness matrix passed
  all five prompt/session cases with the audit enabled. No throughput benchmark
  or timed profile was run by Codex.

Phase 0.87 user-run acceptance gate:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py \
  --confirm-ready --acceptance-audit
```

Use `paper accept_len`, per-position conditional acceptance, prefix survival,
and first-rejection counts to choose the next branch. A weak first position
suggests target-context/bridge or checkpoint alignment; a strong first position
with suffix decay suggests Markov-chain behavior. Compare `code_8k` only
directionally with Table 1's code range. Do not optimize or enable the
confidence scheduler until the raw acceptance result is understood.

Phase 0.87 user-run acceptance result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/issue468-acceptance-20260715-105449`. Metadata records
  clean commit `a6e94a1`, the intended isolated acceptance-audit environment,
  exact verification, and byte-identical baseline/runtime output for all three
  prompts.
- Paper-aligned accepted length / verify rate / full-accept rate were:
  `3.000 / 0.500 / 18.5%` for code, `3.690 / 0.615 / 31.0%` for synthesis,
  and `3.568 / 0.595 / 22.7%` for grounded. Excluded truncated-final-proposal
  counts were `3/2/4` respectively.
- `code_8k` is directionally far below Table 1's per-target code macro target of
  `5.09-5.28` accepted length and `0.636-0.660` verify rate. This is not yet
  evidence of an implementation defect because the target model, quantization,
  sampling, generation mode, context length, and corpus do not match Table 1.
  Synthesis and grounded are deliberately unmapped, but their `3.69/3.57`
  values are at or above Table 1's `3.25-3.50` chat macro band; the weakness is
  therefore not universal across this integration.
- First-position conditional acceptance was only `68.5%/73.8%/75.0%` for
  code/synthesis/grounded. Once a prefix survived, later conditional acceptance
  was generally healthy rather than monotonically collapsing: code positions
  2-5 were `78.4/65.5/68.4/76.9%`, synthesis `87.1/88.9/75.0/72.2%`, and
  grounded `84.8/89.3/68.0/58.8%`. Code is weak at the initial target-context
  boundary and remains uneven, but this does not look like a simple broken
  Markov suffix or rolling-window collapse.
- Raw confidence behavior is coherent enough to argue against gross chain
  wiring corruption. Position-1 confidence overestimated observed acceptance
  by about `9.3/6.1/9.4` percentage points. At later reached positions it was
  usually near observed conditional acceptance, with small-sample deviations.
  Prefix confidence and actual survival also tracked the same general shape.
- The CLI does render each raw `--prompt-file` as an official V4 chat message,
  so missing chat encoding is not the issue. However, the Issue 468 runner uses
  the CLI default thinking-high mode (`nothink=false` in metadata), whereas the
  paper's Table 1 protocol explicitly uses non-thinking mode. The base target
  is also an aggressive `IQ2XXS` quantization while the sidecar was trained
  against the official target checkpoint, and the custom 8K tasks differ
  substantially from MBPP/HumanEval/LiveCodeBench.
- Decision: do not label this a DSpark integration bug yet. The next smallest
  diagnostic is a paired acceptance-mode control on the same three prompts:
  add a runner option that applies `--nothink` to both fresh baseline and exact
  audited runtime, then compare accepted length and every position's
  conditional acceptance with this thinking-high result. This is an acceptance
  diagnostic, not a tok/s benchmark. If non-thinking does not materially close
  the gap, move next to a small officially sourced code-prompt corpus before
  investigating quantization or proposal-path arithmetic.

Phase 0.88 thinking-mode acceptance control prepared on 2026-07-15:

- Added acceptance-audit-only `--nothink`, applying the CLI option to both the
  fresh baseline reference and exact DSpark runtime. Ordinary Issue 468
  throughput and stats protocols remain unchanged.
- Added `--acceptance-reference SUMMARY_JSON`. The runner requires an
  acceptance-audit artifact made with the opposite thinking mode and validates
  its sibling metadata against the current binary, base and sidecar paths,
  context, token count, temperature, seed, and all three prompt hashes. This
  prevents unlike runs from being presented as a controlled mode comparison.
- The acceptance report now names the generation mode and, when a reference is
  supplied, reports per-prompt deltas for paper accepted length, verify rate,
  and full acceptance. It also reports every draft position's conditional
  acceptance and prefix-survival delta. Metadata records both reference files
  and their thinking mode. Non-thinking outputs have the distinct
  `issue468-acceptance-nothink-<timestamp>` prefix.
- Python syntax, default acceptance and throughput dry runs, the real prior
  artifact's provenance validation, same-mode and out-of-scope option
  rejection, and synthetic aggregate/position report generation passed. No
  model execution, throughput benchmark, or timed profile was run by Codex.

Phase 0.88 user-run non-thinking acceptance gate:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py \
  --confirm-ready \
  --acceptance-audit \
  --nothink \
  --acceptance-reference \
  speed-bench/local-runs/issue468-acceptance-20260715-105449/summary.json
```

This is six correctness-diagnostic processes, not a tok/s benchmark. The
generated `Thinking-Mode Control` section is the decision surface. A broad
accepted-length improvement, especially at position 1 for `code_8k`, would
identify thinking mode as a material source of the earlier gap. Little or no
improvement would move the next phase to a small officially sourced code-prompt
corpus before investigating target quantization or proposal arithmetic.

Phase 0.88 user-run non-thinking result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/issue468-acceptance-nothink-20260715-111403`.
  Metadata records clean commit `b1261f4`, the validated Phase 0.87 reference,
  non-thinking mode on both processes, exact verification, no inherited
  `DS4_*` environment, and byte-identical baseline/runtime output for every
  prompt.
- Non-thinking accepted length / verify rate / full-accept rate were
  `3.196 / 0.533 / 21.6%` for code, `2.629 / 0.438 / 9.7%` for synthesis, and
  `3.078 / 0.513 / 19.6%` for grounded. Relative to thinking-high, accepted
  length changed by `+0.196/-1.061/-0.490`; the three-prompt macro mean changed
  from `3.419` to `2.968` (`-0.452`).
- Non-thinking modestly helped code across positions 1-4: conditional
  acceptance changed by `+2.1/+2.2/+6.9/+3.0` percentage points, then fell
  `3.6` points at position 5. This raised code verify rate only `3.3` points,
  from `0.500` to `0.533`, still `10.3-12.7` points below Table 1's unmatched
  code macro range of `0.636-0.660`. With only 51 versus 54 proposal rounds and
  different generated trajectories, the small code gain should not be treated
  as a precise or statistically established effect.
- Synthesis degraded from the first position onward: position-1 conditional
  acceptance fell `15.7` points, prefix survival was down `20-26.5` points at
  positions 2-5, and full acceptance fell `21.3` points. Grounded also lost
  `10.3` points at position 1 and `3.1-13.7` points of prefix survival. These
  large, directionally coherent declines show that non-thinking is not a
  generally better alignment mode for these custom prompts.
- Confidence remains coherent rather than indicating gross candidate-record
  corruption. Code position-1 confidence exceeded observed acceptance by
  `7.6` points; synthesis became more overconfident at position 1 (`14.1`
  points), while grounded was close (`3.0` points). No confidence values were
  non-finite.
- Decision: generation mode is prompt-dependent and does not explain the main
  acceptance gap. Retain thinking-high for the Issue 468 throughput
  reproduction because that matches its source harness; use non-thinking only
  when matching the paper protocol. Do not investigate mode plumbing or enable
  confidence scheduling next. The next diagnostic should use a small,
  officially sourced code corpus and the paper's non-thinking mode, while
  preserving the current five-draft greedy protocol initially so corpus/domain
  is the only new variable. Only after that result should we choose between a
  target-quantization control and proposal-path arithmetic validation.

Phase 0.89 official-source HumanEval pilot prepared on 2026-07-15:

- Inspected DeepSpec commit
  `005e03b81cec38b7da6399833d609ee89a2587f2` and the DSpark paper source.
  DeepSpec's HumanEval adapter wraps the original `openai/openai_humaneval`
  prompt as `Write a solution ...` plus the Python code fence, stores that exact
  string in `eval_datasets/humaneval.jsonl`, and passes `turns[0]` as one user
  message with thinking disabled. Its Table 1 evaluator uses all 164 rows in
  file order, up to 2048 output tokens, seed `980406`, temperature `1.0`, and
  standard rejection sampling. The Qwen3/Gemma4 checkpoints use seven draft
  tokens and no confidence scheduler.
- Table 1's four DSpark HumanEval cells are `5.38/5.52/5.43/5.64` accepted
  tokens including the target bonus. Their directional accepted-length range is
  `5.38-5.64`; normalized by seven drafts plus one bonus, verify rate is
  `0.672-0.705`. These are not V4-Flash pass/fail thresholds. The paper's V4
  deployment section independently confirms that the released V4 design uses
  a five-token maximum block, matching this sidecar.
- Added `speed-bench/humaneval-acceptance/` with eight evenly spaced source rows
  `0/23/47/70/93/116/140/163`. The frozen JSONL preserves the exact DeepSpec
  `turns[0]` strings. Provenance pins repository, commit, source file, selection
  formula, source-line hashes, and exact prompt byte hashes. The runner validates
  all of it and materializes prompts without adding trailing newlines.
- Added `run_dspark_humaneval_acceptance.py`. It reuses the Issue 468 runner's
  exact execution, inherited-environment clearing, acceptance parser, and
  baseline byte-equality checks. It runs eight baseline/runtime pairs in
  non-thinking mode with the exact verifier and no default cooldown. It keeps
  the current V4 isolation protocol fixed at five drafts, temperature zero, and
  128 output tokens; no timing values enter its report or result CSV.
- The report contains pooled proposal metrics, every sample's accepted length,
  aggregate conditional/prefix acceptance and confidence by position, and the
  directional official HumanEval range. It explicitly records the remaining
  mismatches: 8 versus 164 samples, V4-Flash IQ2XXS versus Qwen3/Gemma4, five
  versus seven drafts, 128 versus 2048 output tokens, and greedy versus
  temperature-1.0 rejection sampling. It does not execute functional HumanEval
  tests because this gate measures draft acceptance only.
- Python syntax, corpus provenance, byte-exact materialization, dry-run command
  construction, pooled aggregation using real saved audit records, official
  HumanEval reference extraction, report rendering, and a synthetic end-to-end
  16-process artifact flow passed. No model execution, throughput benchmark, or
  timed profile was run by Codex.

Phase 0.89 user-run gate:

```sh
python3 speed-bench/run_dspark_humaneval_acceptance.py --confirm-ready
```

This executes 16 child processes and should take several minutes; it has no
throughput conclusion and does not require an idle machine. Compare the pooled
verify rate first with the custom `code_8k` non-thinking result of `0.533`, then
only directionally with Table 1's unmatched `0.672-0.705` HumanEval range. A
clear increase would justify expanding the official corpus before changing the
model path. A result near `0.533` or below would indicate that corpus/domain is
not the main gap and make an unquantized or less aggressively quantized target
control the next diagnostic.

Phase 0.89 user-run HumanEval result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/humaneval-acceptance-20260715-113734`. Metadata
  records clean commit `42e355a`, the pinned DeepSpec commit and eight-row
  selection, non-thinking mode, exact verification, no inherited `DS4_*`
  environment, and byte-identical baseline/runtime output for all samples.
- Across 254 valid proposal rounds, pooled accepted length was `4.028` for the
  V4 sidecar's five-draft block and pooled verify rate was `0.6713`; 110 rounds
  fully accepted all five drafts (`43.3%`). This is a large corpus effect versus
  non-thinking `code_8k`: accepted length rose from `3.196` to `4.028`, verify
  rate from `0.533` to `0.671` (`+13.8` percentage points), and full acceptance
  from `21.6%` to `43.3%` (`+21.7` points).
- The pooled verify rate is only `0.0012` below Table 1's unmatched HumanEval
  lower bound of `0.6725`; the eight-task unweighted mean was `0.6971`, inside
  the official `0.6725-0.7050` range. Accepted length itself is not directly
  comparable because V4 has five drafts plus one bonus while Table 1 has seven
  plus one. At the measured V4 verify rate, an eight-position normalization
  would be `5.370`, essentially Table 1's `5.38` lower endpoint, but this is an
  explanatory normalization rather than a claimed block-seven result.
- Per-sample verify rates ranged widely from `0.534` to `0.944`, as expected for
  only eight code tasks. The result is not driven by one high-acceptance sample:
  leave-one-sample-out pooled rates span `0.658-0.692`. The proposal-weighted
  aggregate is lower than the unweighted task mean because shorter/easier
  generations contribute fewer proposal rounds.
- Aggregate conditional acceptance was `0.791/0.876/0.881/0.819/0.866` across
  positions 1-5. Relative to non-thinking `code_8k`, every position improved by
  `8.5/7.0/15.7/10.5/13.3` percentage points. There is no suffix collapse;
  once position 1 survives, all later conditional rates remain above `0.819`.
  This is strong evidence against a general bridge, Markov-chain, rolling-window,
  or proposal-record wiring defect.
- Raw confidence is coherent on the official-source prompts. Conditional
  confidence differs from observed acceptance by `+3.6/-3.1/-5.5/-3.3/-8.0`
  points across positions, and prefix confidence differs from survival by
  `+3.6/+0.9/-2.7/-3.8/-6.0` points. No value was non-finite. STS calibration
  remains unnecessary for validating raw proposal quality.
- Decision: the custom long `code_8k` fixture, not a universal DSpark arithmetic
  failure, explains most of the earlier acceptance gap. Deprioritize target
  quantization and proposal-path arithmetic controls. This eight-sample pilot is
  strong directional validation, not yet a publishable Table 1 reproduction.
  The next acceptance step should pin the complete 164-row DeepSpec HumanEval
  corpus and support a deterministic 32-sample expansion first; if that remains
  near the paper's normalized range, use the same corpus for a user-run paired
  throughput study before considering a full 164-sample acceptance run.

Phase 0.90 HumanEval scale ladder prepared on 2026-07-15:

- Replaced the eight-row frozen subset with all 164 byte-exact turns from the
  same pinned DeepSpec commit. Provenance now records and validates the complete
  upstream file size/hash, every source index and line hash, every prompt's byte
  size/hash, and the local 164-row JSONL size/hash. A direct checkout comparison
  passed for every row.
- Added `--sample-count`, defaulting to `32`. Selection uses the recorded
  inclusive spacing formula over the complete source order. Count `8` exactly
  reproduces indices `0/23/47/70/93/116/140/163`; count `32` selects
  `0/5/11/16/21/26/32/37/42/47/53/58/63/68/74/79` and
  `84/89/95/100/105/110/116/121/126/131/137/142/147/152/158/163`;
  count `164` preserves every source row in order. Counts above the corpus size
  are rejected.
- Selection is stored separately in `experiment_selection` metadata and the
  summary protocol, so every run identifies both the immutable full corpus and
  the chosen scale. Run directories now include the count as
  `humaneval-acceptance-<count>-<timestamp>`.
- Updated report wording and corpus/benchmark documentation for dynamic sample
  counts. The default 32-sample run executes 64 child processes. Based on the
  user's 137-second eight-sample run, it should take roughly 9-12 minutes on the
  same machine; there is no cooldown and no throughput conclusion.
- Full-corpus provenance, 8/32/164 deterministic selection, 8- and 32-sample
  dry runs, over-capacity rejection, Python syntax, and a synthetic 32-sample
  end-to-end artifact flow passed. The synthetic flow covered 64 process rows,
  exact prompt materialization, pooled sample/aggregate CSVs, all position rows,
  metadata/summary selection identity, and report rendering. No model
  execution, throughput benchmark, or timed profile was run by Codex.

Phase 0.90 user-run gate:

```sh
python3 speed-bench/run_dspark_humaneval_acceptance.py --confirm-ready
```

Interpret pooled verify rate and the per-sample distribution first. If the
32-sample result remains near the official directional normalized range, stop
expanding acceptance immediately and prepare a same-32-sample paired throughput
study. Run all 164 acceptance samples only if the 32-sample result materially
changes the eight-sample conclusion or shows unstable task-level behavior.

Phase 0.90 user-run 32-sample result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/humaneval-acceptance-32-20260715-121045`. Metadata
  records clean commit `8334c06`, the expected 32-row inclusive selection from
  the pinned 164-row corpus, exact non-thinking verification, no inherited
  `DS4_*` environment, and byte-identical baseline/runtime output for every
  task.
- Across 1,023 valid proposal rounds, accepted length was `4.202` for the V4
  five-draft block, pooled verify rate was `0.7004`, and 479 rounds fully
  accepted all five drafts (`46.8%`). Verify rate is inside Table 1's unmatched
  HumanEval directional range of `0.6725-0.7050`, near its upper endpoint. The
  explanatory eight-position normalization is `5.603`, also inside Table 1's
  `5.38-5.64` accepted-length range, but remains a normalization rather than a
  block-seven measurement.
- Task-level verify rates ranged from `0.510` to `0.861`; the unweighted mean
  was `0.708`, median `0.728`, and 21 of 32 tasks individually met or exceeded
  the paper range's lower endpoint. No task dominates the pooled result:
  leave-one-task-out rates stay within `0.696-0.711`.
- Four tasks overlap the earlier eight-sample pilot (`000/047/116/163`). Every
  recorded scalar acceptance metric, including proposal/truncation counts,
  reproduced exactly across the independent runs. This supports deterministic
  corpus materialization, chat rendering, proposal generation, and audit
  aggregation.
- Aggregate conditional acceptance was `0.813/0.893/0.879/0.871/0.842` across
  positions 1-5. Later positions remain uniformly strong; there is no suffix
  decay or hidden failure that the eight-sample pilot missed. Raw conditional
  confidence differs from observed acceptance by `+4.9/-1.4/-2.2/-4.3/-4.3`
  points, while prefix confidence differs from survival by
  `+4.9/+3.5/+1.5/-1.5/-2.6` points. Calibration is coherent without STS.
- Decision: acceptance validation is complete at the current protocol. Do not
  spend machine time on all 164 rows and do not reopen bridge, Markov,
  rolling-window, quantization, or confidence-scheduler investigations without
  new contradictory evidence. The next phase is a user-run, uninstrumented,
  paired baseline-versus-exact-DSpark throughput study on this exact 32-task
  selection. It must retain non-thinking, temperature zero, seed one, 128 output
  tokens, byte-equality checks, and isolated child environments so acceptance
  quality and performance refer to the same workload.

Phase 0.91 same-32 HumanEval throughput gate prepared on 2026-07-15:

- Added `speed-bench/run_dspark_humaneval_throughput.py`. It reuses the pinned
  complete HumanEval corpus and the exact deterministic 32-row selection from
  Phase 0.90. The measured workload remains non-thinking, temperature zero,
  seed one, 128 output tokens, context 16,384, and the exact verifier.
- A measured run requires the Phase 0.90 `summary.json`. The runner validates
  its source commit, complete selection object and sample labels, protocol,
  binary/base/sidecar paths, context, token count, seed, exact verifier, and
  non-thinking mode before starting. Per-task acceptance verify rates are
  joined from that separate artifact; acceptance instrumentation is never
  enabled in throughput children.
- Two global prompt pairs warm the relevant process-local paths with reversed
  baseline/runtime order and are excluded from results. The 32 measured tasks
  contribute one pair each, alternating baseline-first and runtime-first, with
  a three-second cooldown after every child. Every second output in a pair must
  match the first byte-for-byte. Baseline children have no `DS4_*` variables;
  runtime children set only `DS4_DSPARK_GPU_RUNTIME=1` and
  `DS4_DSPARK_MULTI_COMMIT=1`.
- The median of the 32 within-task speed ratios is the primary metric. The
  report also records ratio of medians, geometric/arithmetic means, inclusive
  interquartile and full ranges, faster/equal/slower task counts, every task's
  prior acceptance and paired ratio, and a descriptive Pearson correlation
  between acceptance and speed. This avoids averaging absolute t/s across
  generations with unlike output lengths.
- Python compilation, the real Phase 0.90 acceptance-reference validator, the
  default dry schedule, the minimum-sample guard, and a synthetic end-to-end
  68-execution run all passed. The synthetic run covered both warmup orders,
  every measured order, output-reference equality, CSV row counts, metadata,
  summary/report rendering, and acceptance/speed correlation. No model process
  or performance benchmark was run by Codex.

Phase 0.91 user-run command:

```sh
python3 speed-bench/run_dspark_humaneval_throughput.py \
  --confirm-ready \
  --acceptance-reference \
  speed-bench/local-runs/humaneval-acceptance-32-20260715-121045/summary.json
```

Allow roughly 13-15 minutes based on the preceding 32-sample audit plus four
warmup children and 204 seconds of cooldown. Run it with the machine as quiet
and thermally stable as practical. Do not compare its absolute t/s with the
instrumented acceptance audit; interpret the median paired ratio first, then
the task distribution and acceptance/speed relationship.

Phase 0.91 user-run throughput result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/humaneval-throughput-32-20260715-124729`. Metadata
  records clean commit `f35f4cc`, no inherited `DS4_*` environment, exact
  non-thinking verification, the validated Phase 0.90 acceptance reference,
  both reversed warmup pairs, and the intended alternating measured order.
  The run lasted 13 minutes 29 seconds. Every baseline/runtime pair produced
  byte-identical output.
- Baseline and DSpark medians were `23.00` and `14.94 t/s`. The median paired
  ratio was `0.6456x`, geometric mean was `0.6431x`, and ratio of medians was
  `0.6496x`: a `35.4%` median throughput loss. The interquartile range was
  `0.6020x-0.7044x`, the full range was `0.4269x-0.8063x`, and all 32 tasks
  were slower with DSpark. A deterministic task bootstrap placed the median's
  descriptive 95% resampling interval at `0.6214x-0.6831x`.
- Task acceptance strongly predicted speed: Pearson `r=0.839` (`R^2=0.704`
  for a one-variable linear fit). Acceptance quartiles averaged verify rates
  `0.580/0.686/0.744/0.823`, with median paired ratios
  `0.581/0.617/0.692/0.710`. This is coherent runtime behavior, not evidence
  of a broken proposal stream: better acceptance avoids substantial verifier
  work, but does not make the current exact verifier competitive.
- The descriptive fit was `ratio = 0.106 + 0.767 * verify_rate`. Extrapolating
  beyond the observed range predicts only `0.873x` at perfect acceptance and a
  physically impossible `1.166` verify rate for break-even. Do not treat that
  linear extrapolation as a model law, but it reinforces Phase 0.86's direct
  cost accounting: acceptance improvements alone cannot recover the fixed
  exact-verifier and sidecar cost.
- Background interference existed. The initial/final snapshots showed a
  Logitech updater at `88-97%` CPU, and an acceptance/order-controlled fit
  estimated about a `5.7` percentage-point ratio decline over the run. There
  were no macOS thermal/performance warnings. Baseline first-half/second-half
  medians remained `23.02/22.98 t/s`, baseline coefficient of variation was
  `1.77%`, the controlled mode-order effect was only about `-1.0` point, and
  every task lost by at least `19.4%`. Interference limits fine task-level
  comparisons but cannot change the aggregate decision.
- Decision: the current exact DSpark runtime is not throughput-competitive on
  this Metal system even on the official-domain workload where proposal
  quality matches the paper directionally. Acceptance validation is complete;
  do not run all 164 tasks, add a confidence scheduler, or optimize the
  sidecar. The next phase must return to target verification and pursue a
  numerically authoritative compute-batched verifier that preserves exact
  autoregressive cache/state updates while amortizing target weight access.
  The unsafe generic-prefill fast verifier remains only a useful parity and
  speed reference, never production authority.

Phase 0.92 exact Q8 proposal-row microbatch prepared on 2026-07-15:

- Revisited the precise arithmetic boundary established in Phase 0.67. The
  retained exact attention-pre runtime batches HC/norm/RoPE work but deliberately
  launches the one-token Q8 decode matvec separately for every proposal row.
  The generic 2-5-row prefill matvec was previously rejected because its
  different lane/reduction shape introduced immediate Q-LoRA drift.
- Added a Metal-only 2-5-row Q8 kernel and
  `ds4_gpu_matmul_q8_0_exact_rows_tensor`. For every output weight row, it loads
  each Q8 block once and applies it to all proposal activations. Each activation
  row retains the one-token kernel's block stride, multiply/add sequence,
  per-lane accumulator, SIMD reduction, cross-SIMD threadgroup reduction, and
  row-major destination. This amortizes weight reads and dispatches without
  substituting prefill arithmetic.
- Added opt-in `DS4_DSPARK_EXACT_Q8_ROWS=1`. It affects only the three
  cache-independent Q/KV projections inside exact attention preparation and
  only for proposal widths 2-5. It does not touch KV quantization, raw or
  compressed caches, compressor/indexer state, attention, output projection,
  FFN, or target-HC capture. A kernel/setup failure fails the exact batch and
  enters the existing restored serial fallback rather than changing token
  authority.
- Runtime diagnostics require 129/129 successful candidate projections for
  every multi-row verifier call. The correctness matrix gained
  `DS4_TEST_DSPARK_EXACT_Q8_ROWS=1`, clears inherited candidate state, and
  rejects missing, partial, failed, or unexpected candidate records. The
  long-generation attention-pre soak gained the same explicit mode.
- The authoritative candidate matrix passed reasoning, Italian, medium
  context, rolling-window generation, and resumed chat with byte-identical
  output. Selected-layer observer matrices at layers `0/21/30/42` compared all
  attention-pre boundaries against the original one-row projections and
  reported zero drift for every proposal. The 64-token generation,
  rolling-window, and resumed-chat soak also passed with every candidate route
  complete. Timings from these correctness runs were ignored.
- Full `ds4`/test/server/eval/agent/CPU/warm-prefill builds, focused DSpark
  validation and shape tests, the unchanged default exact runtime matrix, and
  the legacy fast-verifier soak passed. Shell syntax, Python compilation, dry
  command generation, synthetic ablation summary/report arithmetic, and
  inherited candidate-environment clearing also passed.
- Added `--exact-q8-rows-ablation` to `run_dspark_comparison.py`. It directly
  compares uninstrumented default exact DSpark with the opt-in candidate on the
  existing high-acceptance 64-token microbenchmark. It clears all inherited
  DSpark/instrumentation variables, alternates order over three pairs after one
  warmup per mode, waits ten seconds between processes, and requires
  byte-identical output. No performance benchmark was run by Codex.

Phase 0.92 user-run gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --exact-q8-rows-ablation
```

Interpret the median paired candidate/default ratio first. This is a component
ablation, not a baseline-versus-DSpark result. Promote the kernel only if all
three pairs are positive by a useful margin; otherwise retain it as an exact
research route and use the result to choose the next verifier slice.

Phase 0.92 user-run result on 2026-07-15:

- Default exact median: `22.00 t/s`; exact Q8 proposal-row median:
  `23.30 t/s`; ratio of medians: `1.0591x`; median paired ratio:
  `1.0625x` (`+5.9%` by medians).
- All three paired ratios were positive: `1.0655x`, `1.0516x`, and
  `1.0625x`. Every measured stdout SHA-256 was identical. This clears the
  explicit promotion gate with a useful margin despite the machine's accepted
  background interference.
- Raw result:
  `speed-bench/local-runs/20260715-135037/results.csv`.

Phase 0.93 exact Q8 proposal-row promotion prepared on 2026-07-15:

- Promoted the 2-5-row Metal Q8 kernel to the environment-free exact runtime
  default. `DS4_DSPARK_EXACT_Q8_ROWS=0` now selects the legacy sequence of
  one-token Q8 projections as a diagnostic and performance control. Explicit
  nonzero values remain enabled for compatibility.
- Converted `--exact-q8-rows-ablation` into a promotion confirmation comparing
  explicit legacy control against promoted default. Both sides remain
  uninstrumented, alternate order, clear inherited DSpark settings, and require
  byte-identical output.
- The environment-free promoted correctness matrix and explicit legacy-control
  matrix both passed reasoning, Italian, medium-context, rolling-window, and
  resumed-chat cases. The promoted and legacy attention-pre soaks both passed
  64-token generation, rolling-window generation, and resumed chat. Timings
  from all correctness runs were ignored.
- Full `ds4`/test/server/eval/agent/CPU/warm-prefill builds, focused DSpark
  validation/shape tests, shell syntax, Python compilation, synthetic
  promotion-summary/report arithmetic, inherited environment clearing, command
  generation, and `git diff --check` passed. The CPU-only object emitted the
  same eight known unused-symbol warnings. No timed promotion confirmation was
  run by Codex.

Phase 0.93 user-run confirmation:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --exact-q8-rows-ablation
```

This repeats the same underlying component comparison after promotion, but now
proves that the faster route is selected without an environment variable and
that `DS4_DSPARK_EXACT_Q8_ROWS=0` faithfully retains the old control.

Phase 0.93 user-run confirmation result on 2026-07-15:

- Legacy one-row Q8 median: `22.01 t/s`; promoted exact Q8-row median:
  `23.34 t/s`; ratio of medians: `1.0604x`; median paired ratio:
  `1.0609x` (`+6.0%` by medians).
- All three promotion-confirmation ratios were positive: `1.0613x`,
  `1.0609x`, and `1.0568x`. Every measured stdout SHA-256 was identical.
  This independently confirms both environment-free default selection and the
  legacy opt-out after promotion.
- Raw result:
  `speed-bench/local-runs/20260715-140157/results.csv`.
- Phase 0.93 is complete. Retain `DS4_DSPARK_EXACT_Q8_ROWS=0` only as a
  regression control; subsequent exact-runtime work should build on the
  promoted Q8 proposal-row path.

Phase 0.94 promoted-default HumanEval throughput rerun prepared on 2026-07-15:

- Reused `run_dspark_humaneval_throughput.py` unchanged so the workload remains
  directly comparable with Phase 0.91: the same pinned 32 HumanEval tasks,
  acceptance reference, non-thinking template, temperature zero, seed one,
  128 output tokens, 16,384 context, two reversed global warmup pairs,
  alternating measured order, three-second cooldown, and byte-equality gate.
- The runner's child-environment isolation clears inherited `DS4_DSPARK_*`
  settings. The runtime command contains only `DS4_DSPARK_GPU_RUNTIME=1` and
  `DS4_DSPARK_MULTI_COMMIT=1`, so the promoted exact Q8 proposal-row path is
  selected by default. No stats, acceptance audit, route diagnostics, profiler,
  or legacy-control variable is enabled.
- Dry-run validation passed against
  `humaneval-acceptance-32-20260715-121045/summary.json`; no prompts were
  materialized and no model process or performance benchmark was run by Codex.
- Interpret the new within-run median paired ratio first. Then compare it with
  Phase 0.91's `0.6456x` median paired ratio and task-level distribution. This
  before/after comparison is useful because each run normalizes DSpark against
  a fresh paired baseline, but cross-run differences remain more exposed to
  machine interference than each run's own ratios.

Phase 0.94 user-run command:

```sh
python3 speed-bench/run_dspark_humaneval_throughput.py \
  --confirm-ready \
  --acceptance-reference \
  speed-bench/local-runs/humaneval-acceptance-32-20260715-121045/summary.json
```

The previous run took about 13.5 minutes; expect roughly the same duration.

Phase 0.94 user-run result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/humaneval-throughput-32-20260715-140901`. Metadata
  records clean commit `a78878b`, no inherited `DS4_*` environment, the same
  acceptance reference and selection, and the intended uninstrumented
  alternating schedule. Every baseline/runtime pair produced byte-identical
  output.
- Baseline median was `22.93 t/s`; promoted-default DSpark median was
  `15.68 t/s`; ratio of medians and median paired ratio were both `0.6840x`;
  geometric mean was `0.6794x`. All 32 tasks remained slower with DSpark, with
  a median delta of `-31.6%` and paired range `0.4541x-0.8134x`.
- Relative to Phase 0.91, the median paired ratio rose from `0.6456x` to
  `0.6840x`: `+3.84` percentage points and a `1.0594x` relative improvement.
  The geometric mean rose from `0.6431x` to `0.6794x` (`1.0564x`), closely
  matching the exact Q8 component ablation and promotion confirmation.
- Absolute DSpark generation throughput improved on all 32 tasks, with a
  `1.0530x` median task-level increase. Paired ratios improved on 31/32 tasks.
  The sole ratio exception, `humaneval_058`, still improved from `16.94` to
  `18.07 t/s`; its old baseline was unusually low (`21.01` versus `22.89 t/s`
  now), which inflated the old paired ratio. The cross-task baseline median
  stayed stable (`23.00` to `22.93 t/s`).
- Acceptance/speed Pearson correlation remained high (`0.839` to `0.846`).
  The Q8 optimization therefore generalizes broadly and removes real exact
  verifier cost, but acceptance-dependent target verification remains the
  dominant end-to-end problem. Phase 0.94 is complete.

Phase 0.95 HumanEval exact-runtime attribution prepared on 2026-07-15:

- Added `run_dspark_humaneval_exact_profile.py` around the retained
  synchronized exact-layer profiler. Its default tasks are
  `humaneval_152`, the selected workload's low-acceptance/worst-ratio case
  (`0.528`, `0.4541x`), and `humaneval_079`, a high-acceptance/best-ratio case
  (`0.839`, `0.8134x`). Default layers remain first/middle/last: `0/21/42`.
- The runner requires Phase 0.94's uninstrumented `summary.json`, validates its
  protocol, model/binary paths, task pairs and equal hashes, then revalidates
  prompt bytes against the frozen DeepSpec corpus. Every fresh reference and
  synchronized profile must match the prior uninstrumented runtime stdout
  byte-for-byte.
- Extended `run_dspark_exact_layer_profile.py` with backward-compatible
  `--nothink` command support and a private stats-only reference option. The
  HumanEval reference enables runtime stats only to recover emitted tokens,
  target evaluations, evaluated positions, avoided evaluations, multi-attempts
  and accepted depth. The selected-layer processes disable stats and preserve
  the existing synchronized attention-pre/serial-tail/exact-FFN boundaries.
- Reports retain median synchronized cost per proposal row and add actual
  synchronized component totals per emitted token. They also expose target
  evals/emitted, positions/eval, profiled rows/emitted, profile coverage, and
  low/high ratios for row cost, invocation amplification and emitted-token
  cost. This separates arithmetic cost from low-acceptance repetition without
  presenting profiler timings as throughput.
- Python compilation, retained real runtime-stats parsing, synthetic
  multi-layer/task summaries and reports, inherited-environment isolation, old
  profiler dry-run compatibility, new artifact/protocol validation, command
  generation, and `git diff --check` passed. In particular, an inherited
  `DS4_DSPARK_EXACT_Q8_ROWS=0` is cleared from both tasks; the profile exercises
  the promoted default. No model profile or performance run was executed by
  Codex.

Phase 0.95 user-run command:

```sh
python3 speed-bench/run_dspark_humaneval_exact_profile.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-throughput-32-20260715-140901/summary.json
```

Eight child processes run: one stats-only exact reference and three
synchronized layer profiles per task, with five-second cooldowns. Treat every
reported time as attribution context only.

Phase 0.95 user-run result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/humaneval-exact-profile-20260715-144116`. Every
  reference and synchronized profile output matched the prior uninstrumented
  HumanEval runtime artifact byte-for-byte.
- The low-acceptance task, `humaneval_152`, had acceptance `0.528`, prior speed
  ratio `0.4541x`, `0.6667` target evals/emitted, `3.333` positions/eval and
  `2.044` profiled proposal rows/emitted. The high-acceptance task,
  `humaneval_079`, had acceptance `0.839`, prior speed ratio `0.8134x`, `0.2969`
  target evals/emitted, `4.289` positions/eval and `1.250` profiled rows/emitted.
- Per-row exact-verifier cost was nearly identical across tasks. The low/high
  ratios were `1.003x`, `1.005x` and `1.015x` at layers `0`, `21` and `42`.
  Low acceptance instead caused `1.636x` more profiled rows per emitted token,
  producing `1.688x`, `1.677x` and `1.682x` more selected-layer synchronized
  cost per emitted token. Its target-eval invocation rate was also about
  `2.246x` higher.
- Component shares were stable across both tasks. Attention preparation was
  about `18-21%`. Exact FFN was the largest layer-0 component at about `43%`.
  The retained serial attention tail grew from about `36%` at layer 0 to about
  `47%` at layer 42 and was the largest component from layer 21 onward.
- The performance spread is therefore primarily acceptance-driven invocation
  amplification, not task-specific arithmetic cost. High-acceptance tasks are
  still below baseline, so reducing exact verifier cost remains necessary in
  addition to any future scheduling policy.
- Whole attention-suffix batching remains retired: Phases 0.73-0.76 showed it
  regressed throughput by `21.2%`, because deferring the serial core and adding
  batched projections cost more than the retained interleaved tail. The next
  implementation boundary must be a narrow in-place tail improvement that
  preserves row order; do not revive that candidate, the NR4 compressor pair,
  or other previously rejected sparse-attention candidates.

Phase 0.96 attention/inverse-RoPE fusion prepared on 2026-07-15:

- Added opt-in `DS4_DSPARK_EXACT_ATTN_INV_ROPE_FUSED=1` for the retained exact
  verifier's serial attention tail. Raw FlashAttention reduction,
  dense-compressed FlashAttention reduction, and sparse indexed attention can
  now apply inverse RoPE directly when writing each final attention head. The
  separate full-head inverse-RoPE dispatch is skipped only for this candidate.
  Cache mutation, compressor/indexer work, output projections, HC expansion,
  and proposal-row order are unchanged.
- The candidate has function-constant-specialized Metal pipelines. The default
  control compiles inverse-RoPE arithmetic and its branch away; it does not run
  a disabled hot-path branch merely because the candidate exists. Normal APIs
  remain the control, while explicit fused APIs carry validated RoPE arguments.
- Added route tracing and a same-dispatch observer. In observer mode the same
  attention dispatch writes its unrotated reduction to diagnostic scratch; the
  existing standalone inverse-RoPE kernel transforms that exact scratch, and
  the observer compares it with the fused output. This avoids confounding the
  comparison with a second attention reduction.
- Raw attention was bit-exact in the selected-layer gate. Compressed YaRN paths
  showed bounded arithmetic drift because the shared interpolation math is
  optimized in a different shader context. At the forced sparse transition,
  the largest observed record was max absolute `2.38419e-7`, RMS
  `5.31978e-9`, and vector-relative L2 `1.53409e-8`; generated output remained
  byte-identical. The observer rejects non-finite values or records above max
  absolute `1e-6`, RMS `1e-8`, or relative L2 `1e-6`. Do not describe this
  candidate as hidden-state bit-exact.
- Extended `tests/dspark_gpu_candidates_correctness.sh` with raw,
  dense-compressed, and forced sparse-indexed observer gates. Reasoning,
  Italian, medium-context, rolling-window, resumed-chat, and all three route
  checks passed with byte-identical output.
- Added `--attention-inverse-rope-fusion-ablation` to the paired benchmark
  runner. It compares default exact DSpark with only the fusion env enabled,
  forces Metal, clears inherited diagnostics/observers/stats, validates output
  equality, alternates order, and emits a dedicated report. Its dry run passed.
- Normal Metal and CPU builds, shell/Python syntax checks, DSpark correctness,
  and `git diff --check` passed. Codex did not run a tok/s benchmark.

Phase 0.96 user-run command:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --attention-inverse-rope-fusion-ablation \
  --confirm-idle
```

Promotion requires a repeatable paired throughput win plus the existing
byte-identical output gate. Even if throughput wins, retain the standalone
inverse-RoPE path as the bit-exact control and document the bounded fused
arithmetic rather than silently calling it exact.

Phase 0.96 user-run result on 2026-07-15:

- Raw results are in `speed-bench/local-runs/20260715-162112/results.csv` at
  clean commit `9811ff0`, with no inherited `DS4_*` environment. The run used
  one warmup per mode and three alternating, uninstrumented measured pairs.
  Every control/candidate output hash matched.
- Default exact median was `23.35 t/s`; fused inverse-RoPE median was
  `23.48 t/s`; ratio of medians was `1.0056x`. The paired ratios were
  `1.0034x`, `1.0177x`, and `0.9991x`, for a `1.0034x` median paired ratio and
  reported `+0.6%` delta.
- This is directionally positive but not a repeatable win: one pair was
  slightly negative, and the median paired effect is only `+0.34%`, well within
  the interference accepted on this machine. The candidate also carries
  bounded YaRN hidden-state drift that the standalone control does not.
- Do not promote `DS4_DSPARK_EXACT_ATTN_INV_ROPE_FUSED`. Keep it opt-in as a
  measured narrow-tail experiment. Do not spend another standard-prompt
  throughput run trying to resolve a sub-percent effect unless a later change
  can remove the arithmetic drift or combine this fusion with a materially
  larger same-write optimization.

Phase 0.97 confidence-scheduler diagnostic prepared on 2026-07-15:

- The Phase 0.90 HumanEval acceptance artifact has aggregate confidence sums,
  Brier terms, and position counts but no per-proposal confidence vectors. It
  cannot honestly simulate a threshold scheduler after the fact.
- Inspected official DeepSpec commit
  `005e03b81cec38b7da6399833d609ee89a2587f2`, file
  `deepspec/eval/dspark/draft_ops.py`. Its released inference path computes the
  complete DSpark block, applies sigmoid to confidence logits, and selects the
  prefix before the first confidence strictly below the configured threshold.
  A non-positive threshold selects the full block; a low first confidence can
  select zero drafts. The V4-Flash release provides no STS calibration values,
  so this study uses raw sigmoid confidence and says so explicitly.
- Added opt-in `DS4_DSPARK_ACCEPTANCE_TRACE=1`. It requires acceptance audit and
  emits one compact record per proposal with the sequential round number,
  proposed and accepted drafts, truncation flag, and confidence vector. It adds
  no model evaluations and is never enabled for throughput runs.
- Added `speed-bench/run_dspark_humaneval_scheduler_trace.py`. It validates and
  reuses the prior 32 baseline stdout artifacts, runs only 32 traced exact
  runtimes, requires byte-identical output, and requires every per-sample and
  aggregate acceptance metric to reproduce the prior audit exactly.
- Added `speed-bench/analyze_dspark_confidence_scheduler.py`. Full proposals are
  analyzed using local progress `min(accepted, K) + 1`; K=0 still costs an
  ordinary one-token target evaluation. It reports progress retention, target
  positions per local progress, target evaluations/sidecar rounds per local
  progress, premature cuts, lost accepted drafts, wasted verified positions,
  exact in-sample threshold frontiers at `99%`, `97.5%`, and `95%` retention,
  and a fixed-grid leave-one-task-out check. Oracle K=accepted is reported only
  as a non-implementable bound.
- Interpretation is deliberately limited: truncating a proposal changes later
  proposal boundaries, target cost is not linear in positions, and the released
  DeepSpec implementation still computes the full sidecar block before cutting
  the prefix. These proxy results are not speed predictions and cannot justify
  a runtime scheduler by themselves.
- Added model-free tests for strict threshold equality, trace validation, K=0,
  fixed K=5, and the oracle. Metal/CPU builds, DSpark validation/shape binding,
  the full traced exact-runtime correctness matrix, Python/shell syntax checks,
  the synthetic end-to-end analyzer, and `git diff --check` passed. Codex did
  not run a tok/s benchmark or the 32-task user diagnostic.

Phase 0.97 user-run command:

```sh
python3 speed-bench/run_dspark_humaneval_scheduler_trace.py \
  --confirm-ready \
  --acceptance-reference \
  speed-bench/local-runs/humaneval-acceptance-32-20260715-121045/summary.json
```

The command runs 32 exact-runtime processes and no fresh baseline processes.
The next decision comes from `scheduler_summary.md`: first check that traced
acceptance reproduced exactly, then compare in-sample and leave-one-task-out
retention/proxy rows. Do not add a runtime scheduler before reviewing that
result together.

Phase 0.97 user-run result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/humaneval-scheduler-trace-32-20260715-165938` at clean
  commit `787921e`. All 32 traced exact-runtime outputs matched their previously
  validated baseline byte-for-byte. The aggregate audit reproduced 1,023 full
  proposals, accepted length `4.202`, and verify rate `0.700` exactly.
- The conservative `99%` in-sample policy selected threshold `0.381954`, mean
  K `4.629`, retained `0.9902x` local progress, reduced the normalized target-
  position proxy to `0.9360x`, and amplified target evaluations/full sidecar
  rounds per local progress to `1.0099x`. It removed 374 raw target positions
  and 338 wasted positions, lost 42 accepted-draft opportunities, cut
  prematurely in `2.8%` of proposals, and selected K=0 in `0.6%`.
- The balanced `97.5%` policy selected threshold `0.455892`, mean K `4.361`,
  retained `0.9763x` progress, reduced the proxy to `0.8981x`, and amplified
  rounds to `1.0243x`. It removed 630 raw target positions and 552 wasted
  positions, lost 102 accepted drafts, cut prematurely in `5.7%`, and selected
  K=0 in `2.3%`.
- The aggressive `95%` policy selected threshold `0.535048`, mean K `4.053`,
  retained `0.9507x` progress, reduced the proxy to `0.8623x`, and amplified
  rounds to `1.0519x`. Its `11.2%` premature-cut rate and 212 lost drafts make
  it a poor first runtime candidate.
- Leave-one-task-out validation was unusually stable. The `99%` policy chose
  only `0.380-0.385`; the `97.5%` policy chose `0.455-0.460`; and held-out
  progress/proxy results matched the in-sample rows to rounding. The confidence
  signal generalizes across this selected corpus rather than depending on one
  fitted floating-point breakpoint.
- The oracle bound is mean K `3.202` and target-position proxy `0.678x` at full
  progress. It confirms substantial theoretical waste but cannot be
  implemented because it uses future target acceptance.
- These results do not establish a speedup. DeepSpec still computes all five
  sidecar drafts, shorter Metal verifier batches may have poorer per-position
  efficiency, and extra target evaluations can erase the position reduction.
  Proceed with an opt-in exact runtime ablation using predeclared thresholds
  `0.38` and `0.455`; do not promote or tune against throughput yet.

Phase 0.98 confidence-prefix runtime ablation prepared on 2026-07-15:

- Added opt-in `DS4_DSPARK_CONFIDENCE_THRESHOLD`. Its value must parse exactly
  as a finite number in `[0,1]`; malformed, non-finite, negative, and above-one
  values fail generation with an explicit error rather than silently reverting
  to fixed K=5. An unset variable leaves the prior hot path unchanged.
- After all five GPU sidecar candidates and raw sigmoid confidences are ready,
  the runtime selects the prefix before the first confidence strictly below the
  threshold, matching pinned DeepSpec semantics. Threshold `0` keeps all five
  drafts. The scheduler runs before output-capacity and EOS limits and changes
  only the verifier's `n_limit`; sidecar generation and candidate values are
  unchanged.
- K=0 uses the existing ordinary one-token target evaluation and emits the
  target-authoritative token. K=1 also follows the existing one-token fallback.
  K>=2 enters the retained exact batch verifier at the selected width. Cache,
  checkpoint, capture, replay, and accepted-prefix logic are unchanged.
- Scheduler route records are emitted only under existing runtime diagnostics.
  Uninstrumented throughput runs do not log or time the scheduler. The
  correctness harness now rejects unexpected scheduling in fixed controls,
  validates route records when enabled, allows intentionally absent verifier
  component records for K<2, and can require an observed K=0 fallback.
- Threshold `0.455` passed reasoning, Italian, medium-context, rolling-window,
  and resumed-chat output equality while exercising mixed prefix lengths.
  Threshold `1` passed the same matrix and exercised K=0. Fixed K=5 passed
  again with no scheduler records. A direct invalid-value check exited with
  `DS4_DSPARK_CONFIDENCE_THRESHOLD must be a finite number in [0,1]`.
- Added `speed-bench/run_dspark_humaneval_scheduler_ablation.py`. Tasks are
  fixed to low-acceptance `humaneval_152` and high-acceptance `humaneval_079`;
  modes are fixed to K=5, threshold `0.38`, and threshold `0.455`; pair count is
  fixed to three. Each task uses a three-period Latin rotation. One excluded
  three-mode warmup plus 18 measured runs produces 21 exact DSpark processes.
- The runner requires Phase 0.94's 32-task throughput artifact, validates model,
  corpus, protocol, prompts, and prior byte-equal outputs, clears all inherited
  `DS4_*` state, enables only GPU runtime/multi-commit plus the selected
  threshold, and rejects stats, audit, trace, diagnostics, or profiler output.
  Every measured output must match the prior exact DSpark output byte-for-byte.
- Added model-free tests for the Latin rotation and fixed environment policy,
  and documented the gate in `speed-bench/README.md`. Metal/CPU builds, DSpark
  validation/shape binding, all three correctness matrices, Python/shell syntax
  checks, dry run, and `git diff --check` passed. CPU warnings were the existing
  unused-code warnings. Codex did not run a tok/s benchmark.

Phase 0.98 user-run command:

```sh
python3 speed-bench/run_dspark_humaneval_scheduler_ablation.py \
  --confirm-idle \
  --throughput-reference \
  speed-bench/local-runs/humaneval-throughput-32-20260715-140901/summary.json
```

Review per-task and aggregate candidate/fixed paired ratios before any full
32-task rerun. A candidate should be directionally positive on both the low-
and high-acceptance tasks and across most of its six measured pairs. Do not tune
the thresholds or tasks after observing this gate.

Phase 0.98 user-run result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/humaneval-scheduler-ablation-20260715-173043` at clean
  commit `546927a`. The run used the frozen two tasks, three fixed modes, one
  excluded three-mode warmup, three Latin-rotated measured periods per task,
  and a five-second cooldown. All 18 measured outputs matched their prior exact
  DSpark artifact byte-for-byte; no instrumentation was enabled.
- Threshold `0.38` won all six candidate/fixed pairs. The low-acceptance task's
  paired ratios were `1.0674x`, `1.0715x`, and `1.0740x`; the high-acceptance
  task's were `1.0674x`, `1.0716x`, and `1.0631x`. Aggregate median was
  `1.0695x` and geometric mean was `1.0692x`.
- Threshold `0.455` also won all six pairs. On `humaneval_152` it raised median
  throughput from `10.23` to `12.27 t/s`, with paired ratios `1.1994x`,
  `1.1988x`, and `1.1947x` and median `1.1988x`. On `humaneval_079` it raised
  median throughput from `18.69` to `20.84 t/s`, with paired ratios `1.1150x`,
  `1.1216x`, and `1.1097x` and median `1.1150x`. Aggregate median was `1.1582x`
  and geometric mean was `1.1558x`.
- This rejects the main Phase 0.97 uncertainty: on these representatives,
  shorter exact Metal verifier batches save substantially more work than the
  extra target-evaluation rounds cost. The offline target-position proxy was
  conservative rather than falsely optimistic.
- Select threshold `0.455` for the next experiment. It has materially larger
  gains than `0.38`, stayed positive on the high-acceptance control, and was
  selected before throughput was observed. Do not test intermediate thresholds
  or tune per task.
- Do not promote to default yet. Two tasks establish mechanism and direction,
  not workload-wide end-to-end performance. The next phase should run the same
  frozen 32 HumanEval tasks as Phase 0.94, pairing ordinary target baseline
  against exact DSpark with fixed threshold `0.455`, one pair per task and no
  instrumentation. Compare that end-to-end result with Phase 0.94's fixed-K
  `0.6840x` median paired ratio, but use the new within-run baseline/scheduled
  ratios as the authoritative measurement.

Phase 0.99 full scheduled HumanEval gate prepared on 2026-07-15:

- `speed-bench/run_dspark_issue468_comparison.py` now accepts an optional
  runtime-only confidence threshold in its shared environment, command, execute,
  and metadata paths. Baselines reject the threshold, invalid/non-finite values
  reject before execution, and the variable remains absent when the option is
  unused.
- `speed-bench/run_dspark_humaneval_throughput.py --confidence-scheduler`
  reuses the exact Phase 0.94 corpus selection, pair order, warmups, cooldown,
  byte-equality gate, and reporting. It hardcodes threshold `0.455`; there is no
  throughput-facing threshold argument to tune.
- `--scheduler-reference` must be the 32-task Phase 0.97 study. The runner
  validates analysis kind, K=5, in-sample threshold `0.455891937`, held-out
  threshold median/range `0.455`/`0.455-0.460`, retention floor `0.975`, trace
  metadata, selection, and derivation from the supplied acceptance reference.
- The default schedule has four excluded warmup children and 64 measured
  children: one ordinary-baseline/scheduled-runtime pair on each of 32 tasks,
  with alternating order and no instrumentation.
- The new within-run scheduled/baseline ratio is authoritative. Compare it
  descriptively with Phase 0.94's fixed-K `0.6840x`, but do not form a synthetic
  fixed-to-scheduled ratio from runs collected at different times.

Phase 0.99 user-run command:

```sh
python3 speed-bench/run_dspark_humaneval_throughput.py \
  --confirm-ready \
  --confidence-scheduler \
  --scheduler-reference \
  speed-bench/local-runs/humaneval-scheduler-trace-32-20260715-165938/scheduler_summary.json \
  --acceptance-reference \
  speed-bench/local-runs/humaneval-acceptance-32-20260715-121045/summary.json
```

This benchmark was deliberately user-run with the machine as idle as practical;
retain the command for exact reproduction.

Phase 0.99 user-run result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/humaneval-scheduler-throughput-32-20260715-182840`
  at clean commit `fb415b5`. The run had no inherited `DS4_*` environment, no
  instrumentation, four excluded warmup children, and 64 measured children.
  All 32 baseline/scheduled output-hash pairs matched.
- Baseline median was `22.97 t/s`; scheduled DSpark median was `17.14 t/s`.
  Ratio of medians was `0.7462x`, median paired ratio was `0.7498x`, and
  geometric mean was `0.7430x`. The paired interquartile range was
  `0.7013x-0.7947x`, with range `0.5396x-0.9047x`.
- No task crossed baseline: scheduled DSpark was slower on all 32 tasks. This
  remains the primary end-user result and prevents any claim that DSpark is
  already a net speedup on this model and machine.
- Compared descriptively with Phase 0.94's separately collected fixed-K run,
  median paired ratio improved from `0.6840x` to `0.7498x` (`1.0962x`
  relative), and geometric mean improved from `0.6794x` to `0.7430x`
  (`1.0935x` relative). Per-task scheduled/fixed ratios improved on 30 of 32
  tasks, with median `1.0789x`; this cross-run comparison is supporting
  evidence, not a substitute for the new within-run baseline comparison.
- Together with Phase 0.98's controlled six-of-six fixed-K wins, this is strong
  evidence that threshold `0.455` reduces exact verification cost broadly on
  HumanEval. Do not spend another benchmark on nearby thresholds.
- Do not promote `0.455` as a universal runtime default yet. HumanEval is code
  only, the released V4 sidecar has raw rather than STS-calibrated confidence,
  and proposal boundaries change under scheduling. The next gate should test
  the already selected threshold on a small, frozen non-code workload with an
  in-run fixed-K/scheduled comparison and byte-exact target output. Its purpose
  is generalization and regression detection, not threshold selection.

Phase 1.00 math/chat generalization gate prepared on 2026-07-15:

- The pinned corpus is in `speed-bench/dspark-generalization/`. It stores exact
  `turns[0]` content, source indices, source-line and prompt hashes, complete
  original-dataset hashes, selection policy, and DeepSpec commit
  `005e03b81cec38b7da6399833d609ee89a2587f2`.
- Selection is restricted to the row caps in DeepSpec `eval.py`: first 500 for
  GSM8K and Alpaca, all 500 MATH-500 rows, all 30 AIME-2025 rows, and all 80
  MT-Bench rows. Math uses two deterministic interior rows from each math
  subset. Chat uses three SHA-seeded, evaluator-subset-tercile-stratified
  long-form non-code Alpaca rows plus MT-Bench first turns selected before
  execution to span writing, reasoning, and humanities. Model output and
  acceptance were not used in selection. MT-Bench second turns are excluded
  because they depend on a generated first response.
- `speed-bench/run_dspark_generalization_gate.py` runs three modes per task:
  ordinary baseline, exact fixed K=5 DSpark, and exact threshold-`0.455`
  DSpark. The 12 measured triples rotate mode order exactly evenly. Three global
  warmups are excluded, for 39 child processes total with five-second cooldowns.
- Every fixed and scheduled output must match the task's ordinary target output
  byte-for-byte. The runner clears inherited `DS4_*` state and rejects stats,
  acceptance, trace, scheduler diagnostics, or profiler leakage.
- Promotion criteria are frozen now: scheduled/fixed median above `1.0x` in
  both math and chat; at least `4/6` scheduled wins in each domain; overall
  scheduled/fixed geometric mean at least `1.03x`; and no individual
  scheduled/fixed ratio below `0.90x`. Scheduled/baseline is reported separately
  as the end-user comparison. A pass permits making `0.455` the DSpark runtime
  default while retaining fixed K=5 as an explicit control. A failure keeps the
  scheduler opt-in and must not trigger threshold retuning on these tasks.

Phase 1.00 user-run command:

```sh
python3 speed-bench/run_dspark_generalization_gate.py --confirm-idle
```

Do not run this timed gate automatically. The user will return its printed
summary or `summary.md`.

Phase 1.00 user-run result on 2026-07-15:

- Raw results are in
  `speed-bench/local-runs/dspark-generalization-20260715-190530` at clean commit
  `d114c31`. The run had no inherited `DS4_*` environment, no instrumentation,
  three excluded warmups, and 36 measured children. Every baseline/fixed/
  scheduled output-hash triple matched.
- Math fixed/baseline median was `0.6154x`; scheduled/baseline was `0.6695x`;
  scheduled/fixed was `1.0831x`, with `5/6` scheduled wins. Math
  scheduled/fixed geometric mean was `1.1019x` and its minimum was `0.9950x`.
- Chat fixed/baseline median was `0.4429x`; scheduled/baseline was `0.6291x`;
  scheduled/fixed was `1.3928x`, with `6/6` scheduled wins. Chat
  scheduled/fixed geometric mean was `1.3558x` and its minimum was `1.0675x`.
- Overall fixed/baseline median was `0.5416x`; scheduled/baseline was `0.6500x`;
  scheduled/fixed was `1.2122x`. Overall scheduled/fixed geometric mean was
  `1.2222x`, with `11/12` wins and range `0.9950x-1.6225x`.
- The gate passed every criterion frozen before measurement: both domain
  medians exceeded `1.0x`; math/chat wins were at least `4/6`; overall geometric
  mean exceeded `1.03x`; and no task fell below `0.90x`. Do not tune `0.455`.
- Product-level performance remains negative: scheduled DSpark was slower than
  ordinary target decoding on all 12 tasks. Promotion means making the proven
  scheduler the better DSpark default, not claiming DSpark is yet an end-to-end
  speedup.
- Next promote `0.455` as the default in
  `dspark_session_confidence_prefix_limit`. An explicit environment value must
  continue to override it, with `DS4_DSPARK_CONFIDENCE_THRESHOLD=0` selecting
  fixed K=5. Audit every fixed-K benchmark and correctness control because an
  absent environment variable will no longer mean fixed K=5 after promotion.

Phase 1.01 confidence-scheduler default promotion on 2026-07-15:

- `dspark_session_confidence_prefix_limit` now starts from the named runtime
  default `DS4_DSPARK_DEFAULT_CONFIDENCE_THRESHOLD=0.455f`. A non-empty
  `DS4_DSPARK_CONFIDENCE_THRESHOLD` parses as the explicit override. Threshold
  `0` leaves all proposed rows selected and is the fixed-K=5 control.
- Shared benchmark code names both policies:
  `DSPARK_DEFAULT_CONFIDENCE_THRESHOLD="0.455"` and
  `DSPARK_FIXED_CONFIDENCE_THRESHOLD="0"`. Metadata records both the explicit
  override and effective runtime threshold.
- Fixed historical protocols now set `0` explicitly: Issue 468 acceptance,
  HumanEval acceptance, the HumanEval confidence trace, the two-task scheduler
  ablation fixed arm, the math/chat generalization fixed arm, and the no-flag
  historical HumanEval throughput runner. `--confidence-scheduler` remains the
  frozen `0.455` reproduction mode for that runner. Ordinary Issue 468 runtime
  and stats paths omit the override and therefore exercise the product default.
- Dry-run command audits confirmed every fixed path renders
  `DS4_DSPARK_CONFIDENCE_THRESHOLD=0`. The default and explicit-zero
  model-backed correctness matrices each passed reasoning, Italian, medium
  context, rolling-window, and resumed-chat byte equality.
- Validation completed without any timed throughput run:

  ```sh
  python3 -m unittest \
    tests/test_dspark_confidence_scheduler.py \
    tests/test_dspark_generalization_gate.py
  bash -n tests/dspark_gpu_candidates_correctness.sh
  make -j4
  make ds4_test
  ./ds4_test --dspark-validation --dspark-shape-binding
  DS4_TEST_DSPARK_MODE=runtime \
    ./tests/dspark_gpu_candidates_correctness.sh
  DS4_TEST_DSPARK_MODE=runtime \
  DS4_TEST_DSPARK_CONFIDENCE_THRESHOLD=0 \
    ./tests/dspark_gpu_candidates_correctness.sh
  git diff --check
  ```

- Do not repeat Phase 0.99 or 1.00 merely to compare an absent override with
  explicit `0.455`: both reach the same `float` threshold in the runtime. The
  next performance phase should first attribute the promoted scheduled runtime
  on a small frozen cross-domain set, then choose the next exact-verifier or
  proposal-cost optimization from measured per-emitted-token cost. The user
  must run that timed diagnostic.

Phase 1.02 promoted cross-domain attribution prepared on 2026-07-16:

- Added `speed-bench/run_dspark_generalization_attribution.py`. It requires the
  completed Phase 1.00 `summary.json`, its metadata and throughput CSV, validates
  the frozen corpus and output hashes, and proves that the selected labels are
  still the scheduled/baseline extrema in their domains.
- The four frozen tasks are:
  `math500_00166` (`math_low`), `gsm8k_00333` (`math_high`),
  `mt_bench_00075` (`chat_low`), and `alpaca_00115` (`chat_high`).
  Selection uses the prior end-user scheduled/baseline ratio, not new model
  output or attribution measurements.
- The runner starts exactly four instrumented promoted-default runtime
  processes. It reuses each prior scheduled output as the byte-exact reference;
  there are no fresh baseline, fixed-K, warmup, acceptance, trace, or layer
  profile children. The command intentionally omits
  `DS4_DSPARK_CONFIDENCE_THRESHOLD`.
- The report omits diagnostic t/s. It records prior uninstrumented ratio and
  latency context, progress and proposal rounds per emitted token, target
  evaluations and positions, synchronized target and sidecar cost, target
  accounted share, sidecar components, and low/high amplification ratios for
  math and chat.
- Model-free validation passed:

  ```sh
  python3 -m py_compile \
    speed-bench/run_dspark_generalization_attribution.py \
    speed-bench/run_dspark_generalization_gate.py \
    speed-bench/run_dspark_issue468_comparison.py \
    tests/test_dspark_generalization_gate.py
  python3 -m unittest \
    tests/test_dspark_generalization_gate.py \
    tests/test_dspark_confidence_scheduler.py
  python3 speed-bench/run_dspark_generalization_attribution.py \
    --dry-run --allow-dirty \
    --throughput-reference \
    speed-bench/local-runs/dspark-generalization-20260715-190530/summary.json
  git diff --check
  ```

Phase 1.02 user-run command:

```sh
python3 speed-bench/run_dspark_generalization_attribution.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/dspark-generalization-20260715-190530/summary.json
```

Do not run this timed attribution automatically. This prepared commit is a
clean stopping point for a community progress survey before interpreting the
result or opening another implementation phase.

Phase 0.52 checks:

```sh
make ds4 ds4_test ds4-server ds4-eval ds4-agent ds4_cpu.o \
  ds4-warm-prefill-bench
python3 -m py_compile speed-bench/run_dspark_exact_layer_profile.py
python3 speed-bench/run_dspark_exact_layer_profile.py \
  --dry-run --allow-dirty
# Synthetic records covered every known decode stage, layers 0/21/42, summary
# grouping, report generation, and rejection without --confirm-ready.
DS4_METAL_DECODE_STAGE_PROFILE=1 \
DS4_METAL_DECODE_STAGE_PROFILE_LAYER=0 \
DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
# All five outputs matched baseline; timings were ignored. The harness parser
# accepted 375 real stage records across the five retained logs.
./ds4_test --dspark-validation --dspark-shape-binding
git diff --check
```

## Phase 1.03: Community Survey and Exact Prefix Checkpoints

Date: 2026-07-16.

This phase began from clean commit `46b66cb`. The community survey reviewed
issue 468, PR 502, and Igor Lobanov's `dspark-research/issue468` branch.

Community findings that must survive compaction:

- The reported full-stack `+4.9%` result is not directly portable into our
  production path. It depends on a committing batched verifier that is not
  output exact: the research report records substantial greedy token
  divergence and small nonzero sampled-distribution divergence. Their exact
  sequential verifier remains slower than baseline.
- Their Metal drafter win is already represented here. Our DSpark proposal
  generation is GPU-resident and the sidecar is no longer the main cost.
- Their confidence scheduler uses STS-calibrated cumulative survival and a
  confident-prefix-plus-one policy. Our promoted raw-confidence threshold
  already produced large local wins versus fixed K, so changing scheduler
  semantics is a separate future experiment, not part of this port.
- Their prefix-checkpoint mechanism is independently useful: a partial accept
  can restore the already-computed compressor/indexer frontier instead of
  restoring the pre-proposal frontier and replaying accepted target rows.
- Their anchor reuse is valuable mainly when combined with their sublinear,
  approximate verifier. Our exact path already verifies and commits the
  target-generated continuation state inside the proposal batch; no anchor
  change was imported in this phase.

Relevant community references:

- `https://github.com/antirez/ds4/issues/468`
- `https://github.com/antirez/ds4/issues/468#issuecomment-4983904315`
- `https://github.com/antirez/ds4/pull/502`
- `https://github.com/lobanov/ds4/tree/dspark-research/issue468`
- `https://github.com/lobanov/ds4/commit/f631fcc184662c7c0416572f59706090e92de861`

Implemented candidate:

- `DS4_DSPARK_EXACT_PREFIX_CHECKPOINT=1` is default-off.
- The graph allocates fixed prefix slots for proposal lengths 1 through 15.
- During exact verification it captures compressed-attention and ratio-4
  indexer frontiers after each non-final proposal row.
- It retains exact per-row logits and the target hidden-state capture already
  produced by the authoritative verifier.
- On partial acceptance it restores the accepted prefix, reads that prefix's
  logits, trims the target capture to the committed context, and skips replay.
- If any checkpoint, logits, or target-capture requirement is unavailable, it
  restores the original frontier and executes the old exact replay path.
- The approximate fast verifier does not use this path.

Prepared user-run throughput gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-prefix-checkpoint-ablation \
  --prompt-file speed-bench/issue468/code_8k.txt \
  --ctx 16384 \
  --tokens 64 \
  --confirm-idle
```

The ablation is paired and uninstrumented. Both modes use the promoted
confidence scheduler and must match byte-for-byte. Codex must not run it.

Phase 1.03 checks:

```sh
make -j4 ds4 ds4_test
python3 -m py_compile \
  speed-bench/run_dspark_comparison.py \
  tests/test_dspark_exact_prefix_checkpoint.py
python3 -m unittest \
  tests/test_dspark_exact_prefix_checkpoint.py \
  tests/test_dspark_confidence_scheduler.py \
  tests/test_dspark_generalization_gate.py
bash -n tests/dspark_gpu_candidates_correctness.sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-prefix-checkpoint-ablation \
  --dry-run --allow-dirty \
  --prompt-file speed-bench/issue468/code_8k.txt \
  --ctx 16384 --tokens 64
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_PREFIX_CHECKPOINT=1 \
DS4_TEST_DSPARK_CONFIDENCE_THRESHOLD=0 \
  ./tests/dspark_gpu_candidates_correctness.sh
# PASS: reasoning, Italian, medium context, rolling window, and resumed chat.
# The fixed-K control forced partial accepts and the checkpoint engagement
# assertion passed. Timings were ignored.
git diff --check
```

The broad `make test` target was also run. Its extractor, agent, long-context,
local-golden, short-prefill, Metal kernel/equivalence, DSpark validation/shape,
and server sections passed. It exited with three ordinary-model vector failures
outside the DSpark path: `think-tool-recovery`, `logprob-vectors`, and the
SSD-streaming cache-pressure copy of the same short-code token mismatch. Those
runs did not open a DSpark sidecar or enable speculative graph state, so this
phase cannot affect their route. The focused
`./ds4_test --dspark-validation --dspark-shape-binding` command passed.

Next decision after the user-run ablation:

- Promote exact prefix checkpoints only if all output hashes match and the
  paired direction is consistently positive.
- If the candidate is neutral or negative, retain it as research and return to
  exact verifier kernel work. Do not import the community's approximate
  committing verifier merely to reproduce its headline speedup.

## Resume Checklist

If continuing from a compacted context, start here:

1. Check current state:

   ```sh
   git status --short --branch
   git diff --stat
   ```

2. Re-scan all engine-option consumers before adding or renaming a runtime
   option:

   ```sh
   rg -n -- "ds4_engine_options|ds4_engine_open\\(|--mtp|ds4_help_print|DSPARK_GPU"
   ```

3. If testing a real DSpark sidecar, start with inspect/validation only:

   ```sh
   ./ds4 --inspect --model /path/to/base.gguf --dspark /path/to/ds4-dspark.gguf
   ```

   If validation fails, record the exact missing tensor/metadata key here before
   changing the contract. Adjust only with evidence from an actual GGUF,
   converter code, DeepSpec, or the ds4 issue discussion.

4. If adding runtime execution, do not reuse the legacy MTP path blindly.
   DSpark input wiring differs:

   - It needs captured target-layer hidden states from base layers `40,41,42`.
   - It uses `main_proj/main_norm` over those states.
   - It uses noise-token block inputs for the parallel block.
   - It shares base embeddings and the base LM head; they are not expected in
     the sidecar.
   - It has Markov and confidence heads on the last stage.

5. Keep benchmark discipline:

   - Do not run tok/s benchmarks automatically.
   - Avoid hot-path logging by default.
   - If adding timing/debug logs, gate them behind an explicit option or env var.
   - For performance comparisons, the user should run no-log baseline first on
     an idle machine, then optional instrumentation runs.

## Open Questions

- The exact attention-suffix candidate is correct but measured `21.2%` slower
  than default exact Metal generation. Zero-copy head writes reduced cost but
  left a profile-predicted `14.2 ms/emitted token` regression. Retire this
  deferred suffix schedule: keep it opt-in as research code, do not benchmark
  it again, and focus future attention work on optimizing the retained serial
  tail in place without changing its interleaved row schedule.
- The retained serial-tail selected-layer profiler is ready for the user:
  its result localizes depth growth to recurrent compression/indexer work and
  attention. The dense compressed-layer split shows comparable projection and
  recurrent-update costs and two compressor pairs per ratio-4 row. Sparse
  preparation, score, and full top-k are also comparable. The existing
  main-compressor `DS4_METAL_COMPRESSOR_PAIR_NR4` candidate was correct but
  measured `0.4%` slower and is retired. Next attribute the retained attention
  operation across dense and sparse rows rather than pursuing another
  compressor projection schedule. The Phase 0.80 mode-aware runner is now
  complete: sparse indexed attention was consistently `43.3%` slower than
  dense mixed attention at transition. Next optimize the retained RB16
  one-token indexed kernel without reviving rejected RB4 or reducing top-k.
  Phase 0.81's guarded RB16-direct candidate is correct and ready for the
  user-run 8K paired throughput gate. That gate passed consistently at `+1.2%`;
  Phase 0.82's matched synchronized gate passed: RB16-direct reduced sparse
  indexed attention by `14.2%`, with stable dense/control medians and clearly
  separated sparse distributions. Phase 0.83 promoted the guarded route to
  default, retained explicit legacy RB16, and passed the full correctness
  matrix. The final legacy-versus-promoted confirmation passed at `+0.9%`
  across three consistently positive pairs. This optimization is complete;
  keep direct as default and legacy as fallback/control. Phase 0.84 hardened
  the Issue 468 runner against all inherited `DS4_*` state and rebaselined the
  accumulated current default. The aggregate ratio improved `7.0%` relative to
  Phase 0.47 but remains only `0.5096x`. Phase 0.85 added the stats-only
  six-process diagnostic. Its Phase 0.86 gate attributed `92-93%` of accounted
  generation time to exact target verification and exposed accepted depth of
  only `2.25-2.91` on the long prompts. Phase 0.87 corrected that internal depth
  statistic to the paper's accepted-drafts-plus-bonus definition and added an
  exact acceptance-quality audit with position-wise survival, conditional
  acceptance, rejection, and raw-confidence attribution. The next gate is the
  user-run `--acceptance-audit` command. Phase 0.87's result measured paper
  accepted length `3.00/3.69/3.57`; code was weak from position 1 while later
  conditional acceptance and the other prompts argued against a universal
  Markov/rolling-window defect. Phase 0.88 prepared a validated same-prompt
  `--nothink` acceptance control against that exact thinking-high artifact.
  That control modestly improved code but substantially hurt synthesis and
  grounded; generation mode is not the general explanation. The next phase is
  Phase 0.89 prepared an eight-sample, byte-exact DeepSpec HumanEval pilot in
  non-thinking mode while keeping the current five-draft greedy protocol fixed.
  Its user-run result reached `0.671` pooled verify rate across 254 rounds,
  essentially the unmatched paper range's `0.6725` lower bound and `13.8`
  points above `code_8k`. Phase 0.90 pinned all 164 official rows and made a
  deterministic 32-sample expansion the default. Its user-run result reached
  `0.700` pooled verify rate across 1,023 rounds, with leave-one-task-out rates
  of `0.696-0.711` and exact repeatability on four overlapping pilot tasks.
  Acceptance validation is complete; do not run all 164. Phase 0.91's same-32
  uninstrumented throughput study then measured a `0.6456x` median paired
  ratio, with all 32 tasks slower and acceptance/speed Pearson `r=0.839`.
  Quantization, proposal arithmetic, confidence scheduling, and sidecar work
  are no longer leading paths. The next phase must make target verification
  materially cheaper with exact autoregressive semantics; the approximate
  generic-prefill fast verifier remains invalid as token authority. Phase 0.92
  added a decode-arithmetic-identical Q8 proposal-row kernel for the three
  cache-independent attention projections; its user-run ablation won all three
  pairs with a `1.0625x` median paired ratio. Phase 0.93 promotes it to the
  Metal exact-runtime default while retaining
  `DS4_DSPARK_EXACT_Q8_ROWS=0` as the legacy control. Its post-promotion
  confirmation won all three pairs with a `1.0609x` median paired ratio and
  identical output hashes. Phase 0.93 is complete. Phase 0.94 now reruns the
  identical 32-task HumanEval throughput workload on the promoted default; use
  its new within-run ratio to measure end-to-end movement before choosing the
  next exact-verifier slice. That rerun improved the median paired ratio from
  `0.6456x` to `0.6840x`, with absolute DSpark throughput higher on all 32
  tasks, but every task remains slower than baseline. Phase 0.94 is complete;
  Phase 0.95 then attributed the promoted exact runtime on representative low-
  and high-acceptance HumanEval tasks. Per-row layer cost differed by only
  `1.003x-1.015x`, while the low-acceptance task paid for `1.636x` more proposal
  rows and about `1.68x` more selected-layer cost per emitted token. Phase 0.95
  is complete. The retained serial attention tail is the largest late-layer
  component, but whole-suffix batching and deferred projection candidates were
  already rejected. Phase 0.96 prepared a narrow in-place fusion that applies
  inverse RoPE in the final attention write and removes the standalone full-head
  dispatch while preserving the interleaved row schedule. Its output was
  byte-identical, but compressed YaRN arithmetic has documented bounded
  hidden-state drift. The user-run gate measured only a `1.0034x` median paired
  ratio, with one slightly negative pair, so the candidate remains opt-in and
  is not promoted. Phase 0.97 now tests whether official-style raw-confidence
  prefix scheduling can reduce low-acceptance proposal-row work without losing
  too much local progress. Its trace result generalized and supports a guarded
  runtime ablation at predeclared thresholds `0.38` and `0.455`. Phase 0.98 has
  now implemented that policy behind an opt-in environment variable. Its
  two-task gate won all six pairs at threshold `0.455`, with aggregate median
  `1.1582x` versus fixed K=5. The next gate is the frozen 32-task end-to-end
  baseline-versus-scheduled run at exactly `0.455`; do not tune further or
  promote before that result. Do not repeat the sub-percent inverse-RoPE
  ablation unchanged.
- The GPU stage path currently borrows target graph transient batch workspace
  and therefore requires `prefill_cap >= block_size` (five). Decide whether to
  allocate sidecar-specific batch scratch before production enablement or keep
  the target scratch reuse with a guaranteed minimum graph capacity.
- What queue-aware policy should server, agent, and eval use before accepting a
  multi-token batch? Their stop, tool, and forced structural tokens can alter
  the stream after the first output, so they cannot simply reuse the CLI loop.
- How should a future acceptance path preserve non-greedy sampling semantics?
  The diagnostic separates `greedy_eligible` from `stream_eligible`; committing
  multi-token prefixes is only straightforward for a greedy target stream.
- Should future CLI UX stay as `--dspark`, or eventually become a broader
  `--draft dspark` once runtime is real?

## Phase 1.04: Promote Exact Prefix Checkpoints

Date: 2026-07-16.

The user-run gate from clean commit `c10bbc8` decisively passed:

```text
Replay partial-accept median: 12.46 t/s
Exact prefix-checkpoint median: 14.67 t/s
Ratio of medians: 1.1774x
Median paired ratio: 1.1774x
Measured pairs: 3
```

Raw artifact:
`speed-bench/local-runs/20260716-101654/results.csv`.

Artifact audit:

- All three paired ratios were positive: `1.1774x`, `1.2098x`, and `1.1733x`.
- Every warmup and measured stdout had the same SHA-256:
  `0657766e6b609a2098b777c1ad49b40b43539cd1731e5fe617498ec7d2dd9b1a`.
- The recorded commit was exactly `c10bbc85ca...` with no tracked changes.
- The run was paired and uninstrumented. The machine was not perfectly idle,
  but the magnitude, consistency, and byte-identical outputs make the
  promotion decision robust.

Promotion:

- Exact prefix checkpoints are now enabled by default.
- `DS4_DSPARK_EXACT_PREFIX_CHECKPOINT=0` or `off` selects legacy partial-accept
  replay.
- The benchmark gate is inverted for future confirmation: legacy replay is
  the explicit reference and the ordinary exact runtime is the candidate.
- The correctness harness accepts `default`, `1`, or `0`, so it can validate
  promoted behavior, forced behavior, and the legacy fallback separately.

Phase 1.04 checks:

```sh
make -j4 ds4 ds4_test
make ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
python3 -m py_compile \
  speed-bench/run_dspark_comparison.py \
  tests/test_dspark_exact_prefix_checkpoint.py
python3 -m unittest \
  tests/test_dspark_exact_prefix_checkpoint.py \
  tests/test_dspark_confidence_scheduler.py \
  tests/test_dspark_generalization_gate.py
bash -n tests/dspark_gpu_candidates_correctness.sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-prefix-checkpoint-ablation \
  --dry-run --allow-dirty \
  --prompt-file speed-bench/issue468/code_8k.txt \
  --ctx 16384 --tokens 64
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_PREFIX_CHECKPOINT=default \
DS4_TEST_DSPARK_CONFIDENCE_THRESHOLD=0 \
  ./tests/dspark_gpu_candidates_correctness.sh
git diff --check
```

Results:

- 24 focused Python tests passed.
- DSpark validation and shape binding passed.
- The default exact checkpoint path engaged and all five retained correctness
  cases passed byte-for-byte.
- The Metal build and CPU-only core object compiled successfully. The CPU
  object emitted only the repository's existing unused-code warnings.
- No timed benchmark was run by Codex.

## Phase 1.05: Frozen HumanEval Checkpoint Rerun

Date: 2026-07-16.

Purpose:

- Measure the promoted scheduler plus exact prefix-checkpoint runtime against
  ordinary target decoding on the same frozen 32 HumanEval tasks.
- Preserve the original task selection, alternating order, 128-token
  generation, 16K context, non-thinking mode, greedy decoding, and threshold
  `0.455`.
- Compare the new task-level results descriptively with Phase 0.99's scheduled
  artifact without treating separate-day runs as a paired ablation.

Preparation:

- `run_dspark_humaneval_throughput.py` accepts
  `--historical-throughput-reference`.
- It validates the prior sample selection, threshold, protocol, model paths,
  acceptance reference, and scheduler reference.
- The report records current/prior task-level movement in DSpark t/s and
  DSpark/baseline ratio, clearly labeled as cross-run context.
- The runtime command does not set
  `DS4_DSPARK_EXACT_PREFIX_CHECKPOINT`, so it exercises the promoted default.
- The benchmark remains uninstrumented and starts 68 child processes: four
  excluded warmups and 64 measured runs.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_throughput.py \
  --confirm-ready \
  --confidence-scheduler \
  --scheduler-reference \
  speed-bench/local-runs/humaneval-scheduler-trace-32-20260715-165938/scheduler_summary.json \
  --acceptance-reference \
  speed-bench/local-runs/humaneval-acceptance-32-20260715-121045/summary.json \
  --historical-throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260715-182840/summary.json
```

Codex must not run this timed command.

Preparation checks:

```sh
python3 -m py_compile \
  speed-bench/run_dspark_humaneval_throughput.py \
  tests/test_dspark_confidence_scheduler.py
python3 -m unittest \
  tests/test_dspark_confidence_scheduler.py \
  tests/test_dspark_exact_prefix_checkpoint.py \
  tests/test_dspark_generalization_gate.py
python3 speed-bench/run_dspark_humaneval_throughput.py \
  --dry-run --allow-dirty \
  --confidence-scheduler \
  --scheduler-reference \
  speed-bench/local-runs/humaneval-scheduler-trace-32-20260715-165938/scheduler_summary.json \
  --acceptance-reference \
  speed-bench/local-runs/humaneval-acceptance-32-20260715-121045/summary.json \
  --historical-throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260715-182840/summary.json
git diff --check
```

All 26 focused model-free tests passed. The real dry-run validated every
reference and printed the frozen schedule without materializing prompts or
executing the model.

Phase 1.05 user-run result:

- Raw artifact:
  `speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542`.
- The run came from clean commit `a853e9f`, inherited no `DS4_*` variables,
  used threshold `0.455`, and enabled no stats or diagnostics.
- All 32 baseline/runtime output hashes matched.
- Baseline median was `23.08 t/s`; DSpark median was `18.67 t/s`.
- Ratio of medians was `0.8089x`, median paired ratio was `0.8081x`, and
  geometric mean was `0.7910x`.
- The paired interquartile range was `0.7490x-0.8493x`, with a full range of
  `0.5775x-0.9250x`.
- DSpark was faster/equal/slower on `0/0/32` tasks.
- Relative to Phase 0.99's separate scheduled run, the median task-level
  paired-ratio movement was `1.0651x`, the geometric movement was `1.0647x`,
  and 26/32 tasks improved. Median task-level DSpark t/s movement was
  `1.0711x`.
- The six cross-run regressions were `humaneval_042`, `063`, `079`, `095`,
  `105`, and `163`. Four were within about one percent; the largest were only
  `-3.5%` and `-3.3%`. The launch snapshot also showed substantial unrelated
  OBS, media, WindowServer, and system activity, so these are not evidence of
  a checkpoint-specific behavioral regression.
- The median gap to baseline fell from `25.0%` to `19.2%`, recovering about
  `23%` of the prior remaining gap. This is a material broad win, but not a net
  speedup.

Next measurement:

- Existing runtime stats count partial batches but do not distinguish exact
  prefix-checkpoint attempts, successful restores, fallback replays, or the
  accepted target rows whose replay was avoided.
- Phase 1.06 should add those counters only under the existing stats gate, so
  uninstrumented throughput remains unaffected.
- Then run a small frozen stats-only attribution on representative HumanEval
  tasks, including low-acceptance `humaneval_152`, high-throughput
  `humaneval_047`, and tasks with large checkpoint movement such as
  `humaneval_131` or `humaneval_137`.
- Do not rerun all 32 tasks before that attribution identifies the remaining
  exact-verifier cost.

## Phase 1.06: Prefix-Checkpoint Stats Attribution

Date: 2026-07-16.

Engine instrumentation:

- Added runtime-stats counters:
  `prefix_checkpoint_attempts`, `prefix_checkpoint_successes`,
  `prefix_checkpoint_fallbacks`, and
  `prefix_checkpoint_rows_avoided`.
- An attempt is counted only for an exact partial batch with prefix
  checkpoints enabled.
- Success requires frontier commit, exact logits retrieval, and target-capture
  trimming to the committed prefix.
- A fallback is an attempted checkpoint that could not commit and therefore
  used the legacy replay path.
- Rows avoided is the sum of committed target rows that legacy replay would
  have reevaluated.
- All increments occur only when the existing runtime-stats gate is enabled;
  uninstrumented throughput has no added counter work.

Frozen attribution:

- Added `speed-bench/run_dspark_humaneval_checkpoint_attribution.py`.
- It requires the completed Phase 1.05 artifact:
  `speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json`.
- Frozen tasks:
  `humaneval_152` low acceptance,
  `humaneval_047` best current paired ratio,
  `humaneval_131` large checkpoint-era gain, and
  `humaneval_137` large gain with lower acceptance.
- It validates the scheduler threshold, model paths, 32-task protocol, prompt
  bytes, pair completeness, and prior output hashes.
- It starts four stats-enabled exact-runtime processes. Each output must match
  its prior uninstrumented runtime artifact byte-for-byte.
- The report omits diagnostic throughput and presents checkpoint coverage,
  fallback count, replay rows avoided per emitted token, current target
  positions, and a structural legacy-position proxy.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_checkpoint_attribution.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json
```

Codex must not run this timed attribution.

Preparation checks:

```sh
make -j4 ds4 ds4_test
make ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
python3 -m py_compile \
  speed-bench/run_dspark_humaneval_checkpoint_attribution.py \
  speed-bench/run_dspark_issue468_comparison.py \
  tests/test_dspark_checkpoint_attribution.py
python3 -m unittest \
  tests/test_dspark_checkpoint_attribution.py \
  tests/test_dspark_confidence_scheduler.py \
  tests/test_dspark_exact_prefix_checkpoint.py
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
DS4_TEST_DSPARK_CONFIDENCE_THRESHOLD=0 \
  ./tests/dspark_gpu_candidates_correctness.sh
python3 speed-bench/run_dspark_humaneval_checkpoint_attribution.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json
git diff --check
```

Results:

- Metal and CPU core builds passed; CPU emitted only existing unused-code
  warnings.
- 33 focused Python tests passed.
- DSpark validation and shape binding passed.
- The stats-enabled fixed-K correctness matrix passed all five retained cases
  and validated the new checkpoint accounting fields.
- The real dry-run validated all four frozen references and printed commands
  with stats and threshold `0.455`, but no checkpoint override.
- No timed attribution was run by Codex.

Phase 1.06 user-run result:

- Raw artifact:
  `speed-bench/local-runs/humaneval-checkpoint-attribution-20260716-110524`.
- The run came from clean commit `a7083b7`, inherited no `DS4_*` variables,
  used threshold `0.455`, and enabled only the runtime-stats gate.
- Every output matched its completed uninstrumented HumanEval runtime artifact
  byte-for-byte.
- Checkpoint outcomes were `29/29` successful with zero fallback replays:
  `8/8` on `humaneval_152`, `3/3` on `047`, `8/8` on `131`, and `10/10` on
  `137`.
- The checkpoints avoided `72` exact target replay rows:
  `14`, `8`, `24`, and `26`, respectively.
- Avoided replay rows per emitted token were `0.311`, `0.062`, `0.255`, and
  `0.203`.
- Structural target-position reductions versus the legacy replay proxy were
  `17.3%`, `5.6%`, `18.2%`, and `14.9%`.

Interpretation:

- Prefix checkpoints are fully covering eligible exact partial batches. There
  is no fallback or missing-state problem to solve.
- On best-current task `humaneval_047`, target verification alone measured
  `41.99 ms/emitted`, slightly below its historical ordinary baseline cost of
  `42.63 ms/token`. The `4.46 ms/emitted` sidecar is what keeps total DSpark
  cost below parity.
- On low-acceptance `humaneval_152`, target verification measured
  `64.85 ms/emitted` against a `43.78 ms/token` baseline, before adding
  `9.31 ms/emitted` of sidecar cost. Rejected target positions remain the main
  problem there.
- `humaneval_131` and `137` sit between those cases: target cost is near but
  still above baseline, and sidecar adds another `5.12-6.53 ms/emitted`.
- Therefore a single next optimization is unlikely to fix both extremes.
  High-acceptance tasks need lower sidecar or target-batch cost; low-acceptance
  tasks need to avoid unprofitable proposal widths and rejected positions.

Next phase:

- Add stats-only histograms for scheduler-selected verifier width, committed
  progress at each width, target evaluation count/time by width, and sidecar
  rounds.
- Reuse the four frozen tasks. Do not run another 32-task throughput study.
- Use the result to decide whether a cost-aware scheduler can skip
  unprofitable short/low-survival proposals while retaining wide,
  high-acceptance batches.
- Do not tune a new threshold or policy before the width economics are
  measured.

## Phase 1.07: Scheduler And Verifier Width Economics

Purpose:

- Measure the economics of confidence-selected proposal widths before changing
  scheduler policy again.
- Separate proposal selection from actual target verifier width. Capacity,
  EOS, partial acceptance, and ordinary one-token evaluation mean these are not
  interchangeable.
- Attribute sidecar timing to the draft block that actually reaches
  multi-commit. Sidecar work consumed elsewhere or left after generation is
  reported as `sidecar_outside_scheduler_ms`.

Runtime stats:

- Added six-bin arrays for V4 Flash block size five:
  `scheduler_width_rounds`, `scheduler_width_committed`,
  `scheduler_width_sidecar_ms`, `verify_width_evals`,
  `verify_width_positions`, and `verify_width_target_ms`.
- Scheduler width is the confidence-prefix result before capacity and EOS
  limits.
- Verifier width is the actual number of target positions submitted in one
  target evaluation.
- Scheduler sidecar timing is captured per generated draft block and assigned
  only when that exact block reaches multi-commit.
- `sidecar_outside_scheduler_ms` preserves all remaining measured sidecar time,
  including proposal work consumed by another path or left after the last
  emitted token.
- The shared stats parser verifies that rounds, committed tokens, verifier
  evals, verifier positions, target time, and total sidecar time reconcile.

Frozen report:

- Extended `speed-bench/run_dspark_humaneval_checkpoint_attribution.py`; no new
  runner or 32-task pass is needed.
- Reuses the four frozen Phase 1.06 tasks and the completed Phase 1.05
  throughput artifact.
- Adds aggregate and per-task scheduler-width economics, verifier-width
  economics, and sidecar-outside-scheduler totals while retaining checkpoint
  attribution.
- Every stats-enabled output must still match its prior uninstrumented runtime
  artifact byte-for-byte.

Preparation validation:

```sh
make -j4 ds4 ds4_test
make ds4_cpu.o
./ds4_test --dspark-validation --dspark-shape-binding
python3 -m py_compile \
  speed-bench/run_dspark_humaneval_checkpoint_attribution.py \
  speed-bench/run_dspark_issue468_comparison.py \
  tests/test_dspark_checkpoint_attribution.py
python3 -m unittest \
  tests/test_dspark_checkpoint_attribution.py \
  tests/test_dspark_confidence_scheduler.py \
  tests/test_dspark_exact_prefix_checkpoint.py
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_KEEP_LOGS=1 \
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
DS4_TEST_DSPARK_CONFIDENCE_THRESHOLD=0 \
  ./tests/dspark_gpu_candidates_correctness.sh
python3 speed-bench/run_dspark_humaneval_checkpoint_attribution.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json
git diff --check
```

Results:

- Metal and CPU builds passed; CPU emitted only the existing unused-code
  warnings.
- DSpark validation and shape binding passed.
- The 26 focused Python tests passed.
- The stats-enabled fixed-K correctness matrix passed all five retained cases
  byte-for-byte.
- All five real Metal stats records parsed and reconciled scheduler rounds,
  committed progress, verifier evaluations and positions, target timing, and
  sidecar timing.
- The retained short cases confirmed that sidecar outside multi-commit is
  material enough to report separately: approximately `19-39 ms` per case.
- The frozen dry-run validated all four references and printed the expected
  stats-only threshold-`0.455` commands.
- No timed attribution or throughput benchmark was run by Codex.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_checkpoint_attribution.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json
```

Expected duration is roughly two to four minutes on the current machine. The
result decides whether the next phase should implement a cost-aware scheduler
or first reduce a particular verifier-width or sidecar cost.

Phase 1.07 user-run result:

- Raw artifact:
  `speed-bench/local-runs/humaneval-checkpoint-attribution-20260716-120149`.
- All four instrumented outputs matched their prior uninstrumented artifacts
  byte-for-byte.
- Prefix checkpoints remained `29/29` successful with zero fallback and
  avoided the same `72` replay rows as Phase 1.06.
- Scheduler-selected width totals:
  - `K=0`: 4 rounds, `1.000` progress/round, `17.392 ms` sidecar/round.
  - `K=1`: 5 rounds, `1.000` progress/round, `17.172 ms` sidecar/round.
  - `K=2`: 10 rounds, `1.600` progress/round, `18.330 ms` sidecar/round.
  - `K=3`: 13 rounds, `2.154` progress/round, `21.920 ms` sidecar/round.
  - `K=4`: 8 rounds, `2.875` progress/round, `17.779 ms` sidecar/round.
  - `K=5`: 87 rounds, `3.667` progress/round, `17.932 ms` sidecar/round.
- Only `2.9%` of measured sidecar time was outside multi-commit. This is
  primarily one residual proposal block per task, not the main overhead.
- Actual verifier-width cost:
  - width 1: `44.399 ms/eval`, `44.399 ms/position`.
  - width 2: `88.154 ms/eval`, `44.077 ms/position`.
  - width 3: `122.574 ms/eval`, `40.858 ms/position`.
  - width 4: `167.681 ms/eval`, `41.920 ms/position`.
  - width 5: `199.392 ms/eval`, `39.878 ms/position`.

Interpretation:

- Exact verifier batching provides only about a `10%` per-position reduction
  from width one to width five. It remains much closer to linear decode than to
  the sublinear legacy fast verifier.
- The sidecar is a nearly fixed per-round tax because all five draft positions
  are computed before raw-confidence prefix selection. Selecting fewer rows
  reduces target work but not draft work.
- `K=0` and `K=1` rounds are necessarily poor, but forcing them wider would add
  roughly linear target positions with weak expected progress. A new
  cost-aware threshold is unlikely to be a structural win.
- High-acceptance `humaneval_047` is already near the exact-target floor:
  `41.922 ms target + 4.445 ms sidecar` per emitted token. Reducing the sidecar
  to the community Metal drafter's reported range would bring it close to
  parity, while low-acceptance tasks would still require cheaper verification.

## Community Sync: 2026-07-16

Sources reviewed:

- `antirez/ds4` issue `#468`.
- `antirez/ds4` PR `#502`, branch
  `stephenlthorn/dspark-integration`.
- `lobanov/ds4` branch `dspark-research`, fetched locally as
  `lobanov/dspark-research` at `d0aae3d`.
- Canonical community report:
  `issue468/summaries/dspark_runtime_milestone_3_progress.md`.

Community result:

- Lobanov reports `+4.9%` over plain ds4 on a 176-entry corpus and `+4.8%` on
  long-context prompts.
- The stack combines:
  - a committing batched verifier,
  - target-anchor/bonus-token reuse,
  - generalized prefix checkpoints,
  - the PR #502 specialized Metal drafter,
  - STS confidence scheduling.
- Their specialized Metal drafter reduced draft time from about `45 ms` to
  `7.6 ms/cycle`.
- Their batched verifier is explicitly not byte-identical:
  about `56.6%` committed-token divergence in one greedy benchmark, mean
  temp>0 total variation about `0.0104`, and argmax flips about `0.64%`.
  They retain sequential verification as the exact fallback and judge the
  batched path score-neutral on their 92-question gate.

Comparison with this branch:

- Already present here:
  - bonus/anchor reuse in the multi-commit architecture. The next target token
    is folded into the existing proposal cycle rather than paid as a separate
    standalone decode.
  - generalized exact prefix checkpoints, with `29/29` eligible partial
    commits and byte-identical output.
  - a promoted confidence scheduler, empirically validated across code, math,
    and chat.
- Not directly portable under the current contract:
  - the community committing batched verifier. Its throughput win accepts
    numerical and transcript drift, while this branch currently requires every
    tested exact DSpark output to match baseline byte-for-byte.
- Potentially portable:
  - the PR #502 specialized Metal drafter and persistent DSpark KV lifecycle.
    It attacks the fixed sidecar tax identified in Phase 1.07 and can be tested
    behind an environment gate while retaining our exact verifier.
  - STS temperatures/cumulative survival as an offline scheduler comparison,
    but community threshold recalibration was only about a `+1%` term after the
    larger runtime changes. It is not the first priority.

Recommended next phase:

1. Isolate the PR #502 Metal drafter implementation and lifecycle requirements:
   direct target-layer hidden capture, persistent three-stage draft KV,
   verified-row refresh, compact output-head/Markov/confidence handling.
2. Map it against this branch's existing GPU bridge/stage/head/chain path.
3. Add it as a default-off candidate path rather than replacing the exact
   runtime.
4. Require draft/candidate diagnostics, the existing five-case byte-identical
   correctness matrix, and exact-verifier output identity before any benchmark.
5. If correctness passes, prepare a user-run sidecar ablation. The decisive
   question is whether proposal cost moves materially from about `18 ms/cycle`
   toward the community's `7.6 ms/cycle`.

Do not port the divergent committing batched verifier without an explicit
decision to relax the byte-identical output contract.

## Phase 1.08: Default-Off Persistent-KV Metal Drafter

Purpose:

- Port the safe portion of community PR #502 that attacks the fixed DSpark
  sidecar tax.
- Preserve this branch's existing GPU bridge, stage FFN, output head,
  Markov/confidence chain, confidence scheduler, exact prefix checkpoints, and
  byte-identical exact verifier.
- Keep the new proposal route default-off until a user-run paired throughput
  ablation establishes that it is worthwhile.

Implementation:

- Added `DS4_DSPARK_METAL_DRAFTER=1`, effective only with the DSpark GPU
  runtime on Apple Metal.
- Added a noncausal raw-batch FlashAttention entry point. Existing callers
  retain the causal mask; the new entry point uses an all-visible mask for the
  parallel DSpark proposal block.
- Reused the existing device-resident `main_x` bridge instead of importing the
  community branch's separate hidden-flow lifecycle.
- Added persistent projected context KV for all three DSpark stages:
  - the initial target window is projected once;
  - later captures retain the absolute-position overlap;
  - a rolling 128-token window shifts the retained prefix through one shared
    device scratch tensor;
  - only missing prefix/suffix rows are projected.
- The specialized stage route skips full context-KV reprojection and replaces
  five serial attention calls with one noncausal batch call. Existing attention
  preparation, exact FFN batching, output head, compact candidate chain, and
  target verifier remain unchanged.
- Cache alignment is checked before every proposal. A refresh or proposal
  failure disables the specialized round and retries through the legacy GPU
  proposal path instead of aborting generation.
- Runtime stats now report `metal_drafter_attempts`,
  `metal_drafter_successes`, and `metal_drafter_fallbacks`.
- The correctness matrix accepts
  `DS4_TEST_DSPARK_METAL_DRAFTER=1`, requires successful cache/proposal route
  records, and rejects specialized fallback.
- Tightened the matrix's prefix-checkpoint engagement assertion: a partial
  checkpoint is now mandatory only when explicitly requested with
  `DS4_TEST_DSPARK_EXACT_PREFIX_CHECKPOINT=1`. Dedicated model-free tests still
  own the checkpoint feature gate.

Correctness and build validation:

```sh
make -j4
make ds4_cpu.o
python3 tests/test_dspark_metal_drafter.py
python3 tests/test_dspark_exact_prefix_checkpoint.py
python3 tests/test_dspark_checkpoint_attribution.py
python3 tests/test_dspark_confidence_scheduler.py
python3 tests/test_dspark_generalization_gate.py
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_METAL_DRAFTER=1 \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
  tests/dspark_gpu_candidates_correctness.sh
python3 speed-bench/run_dspark_comparison.py \
  --metal-drafter-ablation \
  --dry-run \
  --allow-dirty
git diff --check
```

Results:

- Metal release binaries built cleanly.
- The CPU object built with only the existing unused-code warnings.
- All 38 focused model-free Python tests passed.
- The specialized five-case runtime matrix passed reasoning, Italian, medium
  context, rolling-window, and resumed-chat cases byte-for-byte against
  ordinary baseline output.
- The retained specialized logs recorded `17/17` successful proposal rounds
  with zero fallback. Rolling-window refresh retained 125-127 context rows.
- The default-off control matrix also passed all five cases and recorded zero
  Metal-drafter attempts.
- No timed throughput benchmark was run by Codex.

User-run ablation:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --metal-drafter-ablation \
  --confirm-idle
```

The runner alternates default exact DSpark and the persistent-KV Metal
drafter, disables diagnostics and runtime stats in both modes, requires every
output to match byte-for-byte, and reports paired generation throughput. The
next decision is based on this result:

- A material positive gain justifies a stats-only attribution and broader
  HumanEval/generalization confirmation.
- A flat or negative result means the retained KV lifecycle works, but command
  boundaries or the batch-attention implementation still consume the saved
  projection work.

Phase 1.08 user-run throughput result:

- Raw artifact:
  `speed-bench/local-runs/20260716-123750`.
- Default exact median: `23.72 t/s`.
- Persistent-KV Metal drafter median: `23.99 t/s`.
- Median paired ratio: `1.0114x`, or `+1.1%`.
- All three pairs favored the Metal drafter:
  `1.0164x`, `1.0114x`, and `1.0017x`.
- Every output hash matched. Diagnostics and runtime stats were disabled.
- The direction is consistent, but the effect is too small to promote the
  route or justify a broad HumanEval/generalization rerun.

Stats-only attribution:

- Extended `speed-bench/run_dspark_comparison.py` with
  `--stats-only --confirm-ready` for the Metal-drafter ablation.
- Stats-only reports intentionally omit throughput and require paired
  byte-identical output.
- Runtime parsing now includes Metal-drafter attempt, success, and fallback
  counters.
- Raw artifact:
  `speed-bench/local-runs/20260716-124558`.
- Exact schedules were identical:
  - `14` prepared proposal blocks;
  - `13` multi-commit/target evaluations;
  - `64` target positions and emitted tokens;
  - accepted depth `4.923`;
  - no verifier or source fallback.
- Metal drafter outcomes were `14/14` successful with zero fallback.

Per-emitted-token sidecar attribution:

| mode | bridge | stages | head | chain | sidecar |
|:---|---:|---:|---:|---:|---:|
| default exact | 0.371 ms | 2.888 ms | 0.218 ms | 1.086 ms | 4.562 ms |
| persistent-KV Metal | 0.591 ms | 2.374 ms | 0.149 ms | 1.097 ms | 4.212 ms |

Interpretation:

- Persistent KV plus noncausal batch attention saves `0.514 ms/emitted` in
  the three stages.
- Rolling cache refresh adds `0.220 ms/emitted` to the bridge.
- Net sidecar savings are `0.351 ms/emitted`, a `0.9231x` sidecar ratio.
- At proposal granularity, total sidecar falls from about `20.86 ms` to
  `19.25 ms`; stage work saves about `2.35 ms/proposal`, while bridge refresh
  adds about `1.01 ms/proposal`.
- The observed `+1.1%` throughput gain is therefore consistent with the
  measured sidecar savings. It is not acceptance or verifier noise.
- The candidate is still far from the community report near `7.6 ms/cycle`.
  This branch still pays separate synchronized command submissions for each of
  the three stages and the output head; the compact candidate chain alone is
  about `5 ms/proposal`.

Recommended next phase:

1. Keep `DS4_DSPARK_METAL_DRAFTER` default-off.
2. Refactor only the specialized proposal route so stage 0, stage 1, stage 2,
   and the output head encode into one Metal command batch.
3. Preserve the existing legacy path and retry fallback.
4. Require the same five-case byte-identity matrix and zero fallback.
5. Prepare a user-run paired ablation only after a stats-only profile shows a
   material reduction from the current `19.25 ms/proposal`.

Do not run broader workload throughput yet; the local gain is not large enough
to justify that cost.

## Phase 1.09: Consolidated Metal Drafter Command Batch

Goal:

- Replace the four synchronized command submissions that remained inside the
  default-off persistent-KV Metal drafter with one submission.
- Keep the legacy GPU proposal path, exact target verification, scheduler,
  candidate chain, and retry fallback unchanged.

Implementation:

- The specialized proposal route now begins one Metal command batch before
  stage 0, encodes stage 0, stage 1, stage 2, and the output head into that
  batch, then synchronizes once.
- Stage and head helpers recognize the externally owned command batch and
  skip their local begin/end pairs.
- The command-batch-active flag is cleared on success, failure, target-context
  reset, and full session reset.
- A failed consolidated command batch still records a specialized fallback
  and retries the ordinary GPU proposal route.
- Runtime stats now expose aggregate synchronized proposal timing as
  `metal_drafter_ms`, `prefill_metal_drafter_ms`, and
  `generation_metal_drafter_ms`.
- Specialized stage/head timings are intentionally zero while the aggregate
  timer owns that work, avoiding double accounting.
- The stats-only report calls this aggregate `proposal core`; the default
  comparison remains the sum of its three stage timers plus the head timer.

Validation:

```sh
make -j4
make ds4_cpu.o
python3 tests/test_dspark_metal_drafter.py
python3 tests/test_dspark_exact_prefix_checkpoint.py
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_METAL_DRAFTER=1 \
DS4_TEST_DSPARK_RUNTIME_STATS=1 \
DS4_TEST_KEEP_LOGS=1 \
  tests/dspark_gpu_candidates_correctness.sh
python3 speed-bench/run_dspark_comparison.py \
  --metal-drafter-ablation \
  --stats-only \
  --pairs 1 \
  --warmups 0 \
  --cooldown 5 \
  --confirm-ready \
  --allow-dirty
git diff --check
```

Results:

- Metal and CPU builds passed; the CPU object retained only the existing
  unused-code warnings.
- Focused model-free tests passed.
- The five-case runtime matrix passed reasoning, Italian, medium context,
  rolling window, and resumed chat byte-for-byte.
- Retained logs recorded the one-command-batch proposal path on every
  specialized round and no specialized fallback.
- Stats-only raw artifact:
  `speed-bench/local-runs/20260716-125544`.
- Exact schedules matched:
  - `14` proposal rounds;
  - `13` target evaluations;
  - `64` target positions and emitted tokens;
  - `14/14` specialized successes and zero fallback.

Per-emitted-token attribution:

| mode | bridge | proposal core | chain | sidecar |
|:---|---:|---:|---:|---:|
| default exact | 0.422 ms | 3.033 ms | 1.046 ms | 4.501 ms |
| consolidated Metal drafter | 0.615 ms | 2.203 ms | 1.126 ms | 3.943 ms |

Interpretation:

- Consolidation saves `0.830 ms/emitted` in the proposal core.
- Total sidecar savings are `0.558 ms/emitted`, a `0.8760x` sidecar ratio.
- The bridge still costs `0.192 ms/emitted` more because it refreshes the
  persistent rolling KV cache.
- Compared with the pre-consolidation attribution, candidate sidecar cost fell
  from `4.212` to `3.943 ms/emitted`; candidate proposal core fell from
  `2.523` to `2.203 ms/emitted`.
- This is large enough to justify a fresh user-run paired throughput ablation,
  but not yet a broad HumanEval/generalization rerun.
- Codex ran no timed throughput benchmark in this phase.

User-run throughput gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --metal-drafter-ablation \
  --confirm-idle
```

This phase is committed before the user-run timing gate, keeping the branch at
a compact, recoverable checkpoint. Record the throughput result in a follow-up
commit before beginning the planned community-progress survey.

Phase 1.09 user-run throughput result:

- Raw artifact:
  `speed-bench/local-runs/20260716-125850`.
- Default exact median: `23.86 t/s`.
- Consolidated persistent-KV Metal drafter median: `24.09 t/s`.
- Median paired ratio: `1.0100x`, or `+1.0%`.
- All three pairs favored the consolidated route:
  `1.0100x`, `1.0038x`, and `1.0122x`.
- Every output hash matched. Diagnostics and runtime stats were disabled.

Conclusion:

- The persistent-KV Metal drafter remains a valid but modest optimization.
- Consolidating four command submissions into one improved synchronized
  sidecar attribution, but did not improve end-to-end throughput beyond the
  pre-consolidation `+1.1%` result.
- The difference between the two small ablations is within the interference
  expected on this machine. Do not claim that consolidation regressed or
  improved throughput.
- Keep `DS4_DSPARK_METAL_DRAFTER` default-off while surveying current
  community work. The next useful optimization should target a larger measured
  cost or import a demonstrated upstream win rather than further tuning this
  command boundary.

## Phase 1.10: Community Progress Survey

Date: 2026-07-16.

Surveyed:

- `antirez/ds4` issue 468:
  `https://github.com/antirez/ds4/issues/468`
- open integration PR 502 at commit `7ad4ea2`:
  `https://github.com/antirez/ds4/pull/502`
- `lobanov/ds4` branch `dspark-research` through commit `d0aae3d`.
- current `origin/main` at `80ebbc3`.

Current community state:

- No DSpark implementation has merged into `origin/main`.
- PR 502 remains open and under requested changes. It reconciles the earlier
  Metal runtime, B2 rejection sampling, adaptive block sizing, and exact
  prefix-checkpoint work.
- PR 502's hardware report is strongly workload-dependent: repetitive JSON
  reached a large gain, tables a modest gain, code was near neutral, and prose
  remained substantially slower than ordinary decoding.
- The staged adaptive policy did not solve the prose regression. Its author
  concluded that the drafter's acceptance on prose remains below break-even
  even at small block sizes.
- A ROCm port demonstrated correct noncausal attention but remained slower than
  ordinary decoding on bandwidth-bound Strix Halo hardware. This is useful
  confirmation of the verifier-economics limit, but is outside this branch's
  Metal-only scope.

Lobanov milestone-3 result:

- The reported full stack is:
  committing batched verify, anchor reuse, exact prefix checkpoints, Metal
  drafter, and STS confidence scheduling.
- On the corrected 176-entry corpus it measured `40.04` versus `38.16 t/s`,
  or `1.049x`.
- Its final 92-question score comparison was `61/92` versus plain `60/92`.
- This is not a strict-output result. The branch reports about `40%` token
  divergence on its exactness corpus and `56.6%` on the engaged benchmark when
  using the committing batched verifier. The result is framed as
  score-neutral target-argmax correctness, not byte-identical target replay.
- Draft depth six was measured negative and reverted.
- The Metal-specific STS threshold changed from `0.08` to `0.15`, but those are
  calibrated STS values and are not directly comparable with this branch's raw
  sigmoid threshold `0.455`.

Overlap with this branch:

- Confidence scheduling is already implemented, cross-domain validated, and
  promoted at raw-confidence threshold `0.455`.
- Exact prefix checkpoints are already implemented and promoted.
- Persistent rolling DSpark KV and noncausal Metal attention are implemented
  in the default-off Metal drafter.
- The exact multi-commit path already performs the useful form of anchor
  reuse: it validates `first_token` against the existing target logits and
  includes draft row zero in the same exact target batch. It does not perform a
  standalone target decode before a successful batch commit.
- Runtime stats confirm this structure: the canonical 64-token run emits 64
  tokens using 13 target evaluations over 64 target positions.

Port decisions:

- Do not port PR 502 or Lobanov's full stack wholesale. Their main throughput
  win depends on the numerically different committing batched verifier, which
  violates this branch's byte-identity contract and reproduces the class of
  output mismatch already observed in our earlier fast-verifier experiment.
- Do not port staged adaptive depth or draft depth six; their own measurements
  rejected both as general solutions.
- Do not transplant the `0.15` STS threshold into the raw-confidence scheduler.
- Do not spend time on CUDA or ROCm parity in this Metal-focused effort.

Conclusion:

- The community result is real under a looser correctness contract, but it does
  not close the strict exact-runtime gap.
- This branch's broad exact HumanEval result remains `0.8081x` median paired
  ratio after scheduling and prefix checkpoints. The remaining cost is target
  verification, not the drafter.
- The next strict-compatible engineering phase should return to the exact
  verifier. The most promising bounded experiment is command-batch
  consolidation inside each serial exact-attention tail: preserve row order and
  arithmetic, but remove repeated Metal begin/end synchronizations across KV
  update, compressor/indexer, attention, inverse RoPE, and output projections.
- Keep the existing fast verifier as an observer only unless the user
  explicitly chooses a separate non-byte-identical performance mode.

## Phase 1.11: Exact Attention-Tail Command-Boundary Audit

Date: 2026-07-16.

Goal:

- Validate the Phase 1.10 proposal before adding a default-off implementation
  or asking the user to run another timed benchmark.
- Determine whether successful exact verification really submits separate
  Metal command buffers or compute encoders for KV update,
  compressor/indexer, attention, inverse RoPE, and output projections.

Findings:

- The proposed optimization already exists in the production exact verifier.
- `metal_graph_verify_decode_exact()` begins one Metal command batch before
  the 43-layer loop. On the successful promoted path it keeps that batch open
  while encoding attention preparation, every serial attention-tail row,
  prefix-state captures, and exact FFN batches.
- `ds4_gpu_command_buffer()` returns the active `g_batch_cb` to every helper
  with `owned=0`.
- `ds4_gpu_compute_encoder()` also reuses one persistent `g_batch_enc` for
  that command buffer. Helper calls to `ds4_gpu_end_compute_encoder()` leave
  the active encoder open, and `ds4_gpu_finish_command_buffer()` is a no-op
  for borrowed command buffers.
- Compressor and indexer tensor helpers preserve an already active batch.
  Their local begin/end logic runs only when called without an outer batch.
- The apparent per-component submissions in the tail, compressor, and
  attention-mode profiles are deliberate synchronized diagnostic boundaries.
  They are not present in uninstrumented production execution.
- Remaining begin/end boundaries in the exact verifier are limited to
  profiler/observer capture, candidate failure recovery, and unrelated
  selected-expert readback or streaming paths. They do not provide a
  command-consolidation opportunity inside the successful serial
  exact-attention tail.

Decision:

- Do not implement a tail command-batch flag or prepare a throughput ablation.
  It would duplicate the current command-buffer and encoder ownership model and
  should be a no-op.
- No runtime code changed in this phase. The source audit is the validation;
  no timed benchmark is needed.

Next measured target:

- The serial exact-attention path still performs substantial host-side object
  churn. `metal_graph_encode_exact_attention_prepared()` constructs and frees
  row views for the same batch tensors on every proposal row of every target
  layer.
- Each `ds4_gpu_tensor_view()` allocates a retained `DS4MetalTensor` object and
  updates the global tensor-view tracker under `g_tensor_mu`.
- A bounded strict-compatible experiment can prebuild the fixed row views once
  per exact verifier call and reuse them across all layers. It must preserve
  tensor offsets, command order, cache mutation, prefix captures, arithmetic,
  and the legacy fallback.
- Gate that experiment first with model-free lifetime tests and the five-case
  byte-identity matrix. Prepare a user-run paired throughput benchmark only if
  stats or host timing show that view reuse removes material production work.

## Phase 1.12: Exact Attention Row-View Cache

Date: 2026-07-16.

Goal:

- Remove repeated host-side tensor-view allocation from the successful serial
  exact-attention tail without changing target arithmetic, cache mutation,
  command order, or synchronization.
- Keep the candidate default-off until a user-run paired throughput gate.

Implementation:

- Added `DS4_DSPARK_EXACT_ATTN_ROW_VIEWS=1`.
- For each multi-row exact verifier call, the candidate creates fixed row views
  over:
  - batched attention norm;
  - Q LoRA norm;
  - Q and KV;
  - attention heads;
  - HC split, pre, post, and combine.
- The views are retained through all target layers and freed through one
  cleanup path after the verifier finishes.
- `metal_graph_encode_exact_attention_prepared()` and the existing
  attention-output fallback borrow cached views when available. The legacy
  per-layer allocation path remains unchanged when the candidate is disabled
  or cache construction fails.
- Cache construction verifies that Q LoRA rank is uniform across target
  layers before reusing a fixed row stride.
- Diagnostics report cached view count and completed layer-row uses. A normal
  successful V4 Flash verifier call must record exactly
  `43 * proposal_width` uses and `result=pass`.

Work removed:

- The promoted exact tail normally creates eight temporary view objects per
  proposal row per target layer.
- The candidate creates nine retained views per proposal row once per verifier
  call. The additional heads view keeps the cache compatible with the existing
  default-off attention-suffix path.
- At proposal width five this changes `8 * 5 * 43 = 1720` temporary view
  allocations into `9 * 5 = 45`, removing 1675 Objective-C allocations,
  retain/releases, and tensor-tracker mutex updates per target evaluation.

Correctness and validation:

```sh
make -j4
make ds4_cpu.o
make ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
python3 -m unittest discover -s tests -p 'test_dspark*.py'
python3 -m py_compile \
  speed-bench/run_dspark_comparison.py \
  tests/test_dspark_exact_attention_row_views.py
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_ATTN_ROW_VIEWS=1 \
DS4_TEST_KEEP_LOGS=1 \
  tests/dspark_gpu_candidates_correctness.sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-row-views-ablation \
  --dry-run \
  --allow-dirty
git diff --check
```

Results:

- Metal and CPU builds passed. The CPU object retained only the existing
  unused-code warnings.
- All 44 DSpark model-free tests passed.
- DSpark validation and shape binding passed.
- The runtime correctness matrix passed reasoning, Italian, medium context,
  rolling window, and resumed two-turn chat byte-for-byte.
- The same five-case matrix also passed with the cache disabled, confirming
  that the legacy exact-runtime control remains unchanged and emits no cache
  diagnostics.
- Retained candidate diagnostics included:
  - width 2: `18` views and `86/86` uses;
  - width 3: `27` views and `129/129` uses;
  - width 4: `36` views and `172/172` uses.
- Every cache record reported `result=pass`; no cache fallback occurred.
- The paired runner dry-run emits ordinary exact DSpark as the reference and
  only adds `DS4_DSPARK_EXACT_ATTN_ROW_VIEWS=1` to the candidate. Runtime
  diagnostics and stats are disabled in both measured modes.
- Codex ran no timed throughput benchmark.

User-run throughput gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-row-views-ablation \
  --confirm-idle
```

Use the paired ratio as the promotion decision. This is a host-overhead
optimization, so a small or neutral result is plausible even though it removes
real production allocation work. Do not broaden to HumanEval unless the local
three-pair gate is consistently positive.

Phase 1.12 user-run throughput result:

- Raw artifact:
  `speed-bench/local-runs/20260716-135143`.
- Default exact median: `23.79 t/s`.
- Cached attention row-view median: `23.71 t/s`.
- Ratio of medians: `0.9966x`.
- Median paired ratio: `0.9992x`, or effectively neutral.
- Individual paired ratios were `0.9206x`, `1.0008x`, and `0.9992x`.
- Every measured output hash matched.
- The process snapshot showed substantial unrelated activity, including
  `duetexpertd` near one full CPU core and Logitech services using additional
  CPU. The first pair is therefore treated as an interference outlier rather
  than evidence of an 8% regression.

Conclusion:

- The two undisturbed pairs are within roughly one tenth of one percent of
  parity. Removing tensor-view allocation and tracker locking does not produce
  a measurable end-to-end gain on this workload.
- Reject the row-view cache for promotion and do not run HumanEval or
  generalization throughput for it.
- Keep the candidate default-off only as a reproducible bounded experiment.
  Ordinary exact runtime remains unchanged.
- This result rules out host-side tensor-view churn as a meaningful remaining
  bottleneck. The next optimization should target measured GPU verifier work,
  not another allocation or command-boundary micro-optimization.

## Phase 1.13: Exact Attention-Output NR4 Candidate

Date: 2026-07-16.

Goal:

- Reduce measured GPU work in the retained serial exact-attention output tail.
- Preserve byte-exact target verification by changing activation reuse only,
  without changing any per-output Q8 arithmetic.

Source audit:

- DeepSeek V4 Flash uses:
  - embedding width `4096`;
  - eight attention output groups;
  - output LoRA rank `1024`;
  - projection A group width `4096`;
  - projection B input width `8192`.
- The serial tail performs:
  - eight independent `4096 x 1024` Q8 projection-A matvecs;
  - one `8192 x 4096` Q8 projection-B matvec fused with four-stream HC
    expansion.
- Both existing Metal kernels use `NR0=2`: one threadgroup computes two
  adjacent output rows and reuses each activation load across those rows.
- The synchronized tail profile attributed roughly `0.35-0.36 ms/row` to each
  projection under diagnostic fences, so this is measured GPU work rather
  than inferred host overhead.

Implementation:

- Added default-off `DS4_DSPARK_EXACT_ATTN_OUT_NR4=1`.
- Added NR4 variants for:
  - `kernel_dsv4_attn_out_low_q8_0_f32`;
  - `kernel_dsv4_q8_hc_expand4_q8_0`.
- Each NR4 kernel computes four adjacent output rows per threadgroup.
- The candidate preserves, independently for every output row:
  - Q8 block and lane assignment;
  - block traversal order;
  - scalar accumulation order;
  - SIMD and threadgroup reduction order;
  - output and HC expansion arithmetic.
- Projection dimensions must be divisible by four. Unsupported shapes keep
  the ordinary NR2 path.
- `DS4_DSPARK_EXACT_ATTN_OUT_NR4_TRACE=1` is correctness-only and reports once
  when projection A and projection B plus HC actually select NR4. The paired
  timing harness never enables this trace.

Validation:

```sh
make -j4
python3 -m py_compile \
  speed-bench/run_dspark_comparison.py \
  tests/test_dspark_exact_attention_output_nr4.py
python3 -m unittest tests.test_dspark_exact_attention_output_nr4
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR4=1 \
  tests/dspark_gpu_candidates_correctness.sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-output-nr4-ablation \
  --dry-run \
  --allow-dirty
git diff --check
```

Results so far:

- CPU and Metal builds passed. The CPU object retained only the existing
  unused-code warnings.
- All 49 DSpark model-free tests passed.
- DSpark validation and shape binding passed.
- The user-run harness dry-run emits ordinary exact DSpark as reference and
  adds only `DS4_DSPARK_EXACT_ATTN_OUT_NR4=1` to the candidate.
- Runtime stats, diagnostics, and the NR4 trace are disabled in both timed
  modes.
- The five-case runtime correctness matrix passed reasoning, Italian, medium
  context, rolling window, and resumed two-turn chat byte-for-byte.
- Correctness traces confirmed that both NR4 projections executed.
- The same five-case matrix passed with NR4 disabled, confirming that ordinary
  exact runtime remains unchanged.
- Codex ran no timed throughput benchmark.

User-run throughput gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-output-nr4-ablation \
  --confirm-idle
```

Use the paired ratio as the first decision. If the combined candidate is
clearly positive, confirm it before promotion. If it is neutral or negative,
split projection A and projection B plus HC into separate ablations before
rejecting the underlying NR4 specialization.

First user-run throughput result:

- Raw artifact:
  `speed-bench/local-runs/20260716-140827`.
- Default exact median: `23.69 t/s`.
- NR4 attention-output median: `23.96 t/s`.
- Ratio of medians: `1.0114x`.
- Median paired ratio: `1.0118x`, or `+1.1%`.
- Individual paired ratios were:
  - `1.0118x`;
  - `1.0122x`;
  - `1.0114x`.
- All output hashes matched.
- The spread between the three paired ratios was under `0.1` percentage point.
  This unusually close agreement survived alternating order even though the
  process snapshot contained substantial Logitech and system activity.

Promotion decision:

- The combined result is positive and highly reproducible within the local
  gate. Do not split projection A from projection B plus HC before promotion.
- Promote NR4 as the exact attention-output default on Metal.
- Retain `DS4_DSPARK_EXACT_ATTN_OUT_NR4=0` as a legacy NR2 opt-out.
- Convert the paired harness into a promotion confirmation:
  - reference: explicit legacy NR2;
  - candidate: ordinary exact DSpark with promoted NR4;
  - no runtime stats, trace, or diagnostics in either mode.
- Require both explicit NR4 and explicit legacy NR2 correctness matrices before
  the confirmation benchmark.

Promotion implementation and validation:

- Absence of `DS4_DSPARK_EXACT_ATTN_OUT_NR4` now selects NR4.
- `DS4_DSPARK_EXACT_ATTN_OUT_NR4=0` selects legacy NR2.
- `DS4_DSPARK_EXACT_ATTN_OUT_NR4=1` remains an explicit NR4 control.
- The paired harness now emits:
  - reference: `DS4_DSPARK_EXACT_ATTN_OUT_NR4=0`;
  - candidate: no NR override, using the promoted default.
- The harness labels the result as an NR4 promotion confirmation and keeps
  diagnostics, trace, and runtime stats disabled in both modes.
- CPU and Metal builds passed.
- All 49 DSpark model-free tests passed.
- DSpark validation and shape binding passed.
- Three five-case runtime correctness matrices passed byte-for-byte:
  - explicit NR4, with traces confirming both projections;
  - explicit legacy NR2;
  - ordinary exact runtime with no NR override.
- The promotion-confirmation dry-run printed the intended Metal-only,
  uninstrumented command pair.
- Codex ran no timed confirmation benchmark.

User-run promotion confirmation:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-output-nr4-ablation \
  --confirm-idle
```

If the paired direction remains positive and output hashes match, keep NR4
promoted. A result near parity is acceptable only if the individual pairs do
not show a consistent regression; a clear negative result should restore NR2
and trigger separate projection-A and projection-B-plus-HC experiments.

Promotion confirmation result:

- Raw artifact:
  `speed-bench/local-runs/20260716-142043`.
- Legacy NR2 median: `23.76 t/s`.
- Promoted NR4 median: `23.94 t/s`.
- Ratio of medians: `1.0076x`.
- Median paired ratio: `1.0076x`, or `+0.8%`.
- Individual paired ratios were:
  - `1.0097x`;
  - `1.0072x`;
  - `1.0076x`.
- Every output hash matched.
- The process snapshot was noisy:
  - `duetexpertd` used about one CPU core;
  - Logitech updater and agent processes used substantial additional CPU.
- Despite that interference, all three alternating pairs remained positive
  with a spread of about `0.25` percentage point.

Final decision:

- Keep exact attention-output NR4 promoted on Metal.
- Retain `DS4_DSPARK_EXACT_ATTN_OUT_NR4=0` as the legacy rollback.
- Do not spend a HumanEval or cross-domain throughput run on this isolated
  sub-1% kernel improvement. The local gate and promotion confirmation are
  directionally unanimous, byte-exact, and sufficient for the bounded change.
- Do not split projection A from projection B plus HC unless a future workload
  shows a regression.
- This closes Phase 1.13 at a clean checkpoint. The next phase should pause
  local micro-optimization and review current DS4 issue 468/community progress
  for independently discovered wins that can be ported or compared.

## Phase 1.14: HumanEval break-even oracle

Motivation:

- The frozen scheduled HumanEval study reached a median paired ratio of
  `0.8081x`; all 32 tasks remained slower than baseline, with the best task at
  `0.9250x`.
- HumanEval acceptance is already strong (`0.700` verify rate), so the central
  question is no longer whether the drafter works. It is whether any current
  proposal rounds are locally profitable and how much target-verifier cost
  must fall before broad DSpark can reach parity.
- Aggregate width histograms and the confidence trace could not answer this:
  they did not join acceptance, committed progress, sidecar cost, and exact
  target cost for the same proposal round.

Runtime trace:

- Added a default-off `DS4_DSPARK_ORACLE_TRACE=1` diagnostic.
- It requires `DS4_DSPARK_GPU_RUNTIME_STATS=1`; ordinary and timed runtime paths
  remain unchanged.
- One record is emitted for every completed multi-commit round:
  - full proposed width;
  - confidence-scheduler selected width;
  - actual verified width after capacity/EOS limits;
  - accepted draft prefix;
  - committed output tokens;
  - sidecar milliseconds;
  - target milliseconds, eval count, and evaluated positions;
  - raw confidence-head values.
- Per-round target counters accumulate through the existing target timing
  hook, including replay/fallback target calls when present.
- The record is emitted immediately before multi-commit cleanup, so all exact
  and serial completion paths use the same trace contract.

Generation-cost accounting:

- The first traced sidecar block is prepared during prefill. It must not be
  charged against the frozen baseline's generation t/s.
- The final unused sidecar block is not attached to a scheduler round but is a
  generation cost. The analyzer derives it as:
  `generation_sidecar_ms - traced_sidecar_ms_after_round_1`.
- This identity was confirmed exactly on the smoke task:
  - first/prefetched sidecar: `28.815 ms`;
  - terminal sidecar: `20.781 ms`;
  - traced scheduled sidecar: `691.350 ms`;
  - runtime scheduled histogram: `691.350 ms`.

Offline oracle:

- Added `speed-bench/run_dspark_humaneval_oracle_audit.py`.
- It reuses the frozen 32-task scheduled HumanEval throughput artifact for:
  - each task's uninstrumented baseline generation t/s;
  - prompt identity;
  - byte-exact expected output.
- It runs only one stats-enabled exact DSpark process per task. There is no
  fresh baseline process and no timed throughput comparison.
- It reports:
  - accounted current DSpark/baseline ratio;
  - a future-knowing per-round router ceiling;
  - the same router with free sidecar;
  - the same router with free target verification;
  - current profitable-round and profitable-token shares;
  - the target-time scale required for all-DSpark parity and `1.10x`.
- The router is an optimistic local counterfactual. Routing a round to baseline
  would change later proposal boundaries, and measured target plus sidecar time
  does not include every host/runtime overhead.

Validation:

- Metal build passed.
- CPU build passed with only the project's existing unused-code warnings.
- All 55 DSpark model-free tests passed.
- The 32-task user command dry-run validated the frozen reference, prompt
  selection, environment, and stats-only command surface.
- One full `humaneval_000` traced runtime matched the frozen output
  byte-for-byte.
- Its trace reconciled exactly:
  - `32` traced rounds;
  - `98` traced/stat emitted tokens;
  - `32` traced/stat target evaluations;
  - `129` traced/stat target positions;
  - `5587.478 ms` traced/stat target time;
  - `691.350 ms` traced/stat scheduled sidecar time.
- The one-task result is only a sanity check, not the study:
  - accounted ratio `0.6774x`;
  - current-cost route oracle `1.0000x` because no round was locally
    profitable;
  - free-sidecar route oracle `1.0100x`;
  - target time would need to fall to about `0.638x` of current cost for
    all-DSpark parity on this task.

User-run audit:

```sh
python3 speed-bench/run_dspark_humaneval_oracle_audit.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json \
  --confirm-ready
```

This is a stats-only diagnostic, not a tok/s benchmark. Machine idleness is not
required, although large concurrent GPU workloads should still be avoided
because they can distort component timing.

Completed 32-task oracle audit:

- Raw artifact:
  `speed-bench/local-runs/humaneval-oracle-audit-20260716-144959`.
- Every traced runtime output matched its frozen uninstrumented HumanEval
  artifact byte-for-byte.
- Accounted current DSpark ratio: `0.8076x`.
- Frozen measured scheduled HumanEval ratio: `0.8081x`.
- The `0.0005x` absolute difference validates the aggregate cost model closely
  enough for architectural decisions.

Aggregate ceilings:

- Perfect future-knowing round router at current cost: `1.0058x`.
- Perfect router with free sidecar: `1.0517x`.
- Perfect router with free target verification: `7.4594x`.
- Profitable current-cost rounds: `185/1084`, or `17.1%`.
- Those rounds emitted `899/3532`, or `25.5%`, of output tokens.
- All-DSpark parity under the same schedule requires target time at `0.784x`
  current cost, a `21.6%` reduction.
- An accounted `1.10x` result requires target time at `0.702x`, a `29.8%`
  reduction.

Interpretation:

- The router ceiling is technically above parity but only by `0.58%`. Any
  classification error, routing overhead, or changed proposal sequence would
  consume it. Do not pursue a pre-sidecar selective router as the primary
  speedup strategy.
- Confidence cannot realize even that ceiling because confidence is available
  only after sidecar computation. A future-knowing router that pays every
  sidecar before choosing target verification reaches only `0.9217x`.
- Removing all sidecar while continuing to run DSpark on every round reaches
  only about `0.9056x`; sidecar optimization alone cannot produce broad parity.
- Target verification remains the decisive cost center.

Round-shape attribution:

- Profitable rounds averaged:
  - `4.86` committed tokens;
  - `4.85` accepted drafts;
  - `4.88` verified positions;
  - `189.68 ms` target time;
  - `17.23 ms` sidecar time;
  - `211.68 ms` equivalent baseline time.
- Unprofitable rounds averaged:
  - `2.93` committed tokens;
  - `2.70` accepted drafts;
  - `3.57` target positions;
  - `149.14 ms` target time;
  - `19.30 ms` sidecar time;
  - `126.86 ms` equivalent baseline time.
- `171/185` profitable rounds were full five-token accepts.
- Full K=5 accepts as a class are already close to parity at `0.9765x`.
- Partial K=5 rounds are the main failure mode at `0.5401x`.
- Rounds verifying fewer than five positions aggregate to `0.7020x`.

Rejected staged-verifier direction:

- Aggregate measured target cost by exact verifier width:
  - K=1: `45.84 ms/eval`;
  - K=2: `89.34 ms/eval`;
  - K=3: `125.85 ms/eval`;
  - K=4: `174.26 ms/eval`;
  - K=5: `202.51 ms/eval`.
- A local staged-verification model using those measured costs was negative:
  - serial `1+1+1+1+1`: target time `1.024x` current;
  - `2+3`: target time `1.052x` current;
  - `3+2`: target time `1.069x` current.
- The extra target invocation cost outweighs rejected-suffix savings because
  full acceptance is common and width cost is already close to linear.
- Do not implement staged exact verification without a materially cheaper
  continuation mechanism than an ordinary second target eval.

Remaining scheduler hypothesis:

- A rough local cost sweep on the traced rounds suggests that thresholds around
  `0.75-0.85` may outperform the current `0.455` cost policy.
- This model is not an exact replay: higher thresholds alter later proposal
  boundaries, and its absolute current-policy estimate did not reproduce the
  measured ratio.
- Treat the sweep only as justification for a bounded runtime ablation, not as
  a speed prediction.
- Even the optimistic local curve remains below parity. A scheduler win would
  reduce the remaining verifier gap, not eliminate the need for a cheaper exact
  target path.

Decision boundary:

- The project still has credible speedup potential, but not through additional
  isolated 1% micro-optimizations or confidence routing.
- First run one bounded cost-aware scheduler gate comparing the current
  threshold with substantially more aggressive fixed thresholds.
- If that gate does not approach parity, stop scheduler tuning and return to an
  architectural Metal exact-verifier optimization with a target of at least
  `22%` lower target time under the scheduled HumanEval workload.

## Phase 1.15: Aggressive scheduler throughput gate

Protocol:

- Added
  `speed-bench/run_dspark_humaneval_aggressive_scheduler_gate.py`.
- The gate compares four frozen modes:
  - ordinary target baseline;
  - current confidence threshold `0.455`;
  - aggressive threshold `0.75`;
  - aggressive threshold `0.85`.
- The eight tasks were fixed before throughput:
  - `humaneval_152` and `humaneval_032`: lowest-acceptance controls;
  - `humaneval_000` and `humaneval_121`: middle-acceptance controls;
  - `humaneval_131` and `humaneval_137`: high profitable-token-share cases;
  - `humaneval_011`: high-throughput/high-acceptance case;
  - `humaneval_079`: high-acceptance adversarial case with no locally
    profitable oracle rounds.
- Each mode runs once per task. Four cyclic orders are repeated twice, so every
  mode occupies every order position exactly two times.
- Four global warmups, one per mode, are excluded.
- Total process count is `36`: four warmups and `32` measured runs.
- Every output must match the frozen scheduled HumanEval output byte-for-byte.
- The runner enables no runtime stats, acceptance audit, trace, diagnostics,
  profiler, or fast verifier.

Predeclared promotion gate:

- Candidate geometric mean versus current threshold `0.455` must be at least
  `1.03x`.
- Candidate must win at least `6/8` tasks versus current.
- No task may fall below `0.90x` versus current.
- If both candidates pass, choose threshold `0.85` only if its geometric mean
  is at least `1.01x` the threshold-`0.75` result. Otherwise prefer `0.75`.
- Fresh ordinary baseline is included so the result also shows whether either
  threshold approaches or crosses end-to-end parity.

Correctness-harness repair:

- The existing runtime correctness harness inferred that every scheduler
  selection of K>=2 must execute an exact batch verifier.
- That is false when output capacity subsequently reduces the usable width
  below two. Aggressive thresholds exposed this in resumed chat: K=2 was
  selected with one output slot remaining, and runtime correctly used the
  one-token capacity fallback.
- The harness now requires detailed FFN/attention/Q8 verifier records only when
  an exact or fast batch verifier call is actually observed in that log.
- Other prompts in the same matrix still exercise and validate the exact batch
  component records.

Validation:

- All 62 DSpark model-free tests passed.
- Python compilation, shell syntax, `git diff --check`, and the 36-process dry
  run passed.
- The dry run validated the frozen 32-task reference, selected task identities,
  command environments, and exact four-mode order balance.
- Threshold `0.75` passed reasoning, Italian, medium context, rolling window,
  and resumed two-turn chat byte-for-byte.
- Threshold `0.85` passed the same five-case matrix byte-for-byte.
- Codex ran no timed throughput benchmark.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_aggressive_scheduler_gate.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json \
  --confirm-idle
```

Use the printed promotion gate as the primary decision. Even if a candidate
passes versus `0.455`, do not promote it globally unless its fresh
candidate/baseline ratios show a meaningful path toward parity.

Completed aggressive scheduler gate:

- Raw artifact:
  `speed-bench/local-runs/humaneval-aggressive-scheduler-20260716-151432`.
- The run was collected at clean commit `4dfc29b`.
- All `32` measured outputs matched byte-for-byte within each task.
- The artifact contains exactly eight rows per mode and every mode occupied
  every order position exactly twice.
- No thermal or performance warning was recorded. The process snapshot still
  showed substantial Logitech, Stats, WindowServer, and system activity, so
  aggregate direction is more trustworthy than small single-task differences.

Current threshold `0.455` on the selected subset:

- Median ratio versus fresh baseline: `0.7775x`.
- Geometric mean versus fresh baseline: `0.7591x`.
- No task was faster than baseline.

Threshold `0.75`:

- Median ratio versus fresh baseline: `0.8529x`.
- Geometric mean versus fresh baseline: `0.8552x`.
- Median ratio versus current threshold: `1.1290x`.
- Geometric mean versus current threshold: `1.1266x`.
- Won all `8/8` tasks versus current.
- Worst task ratio versus current: `1.0123x`.
- Passed every predeclared promotion rule.
- Relative to threshold `0.455`, it closed about `39.9%` of the geometric
  deficit to baseline and `33.9%` of the median deficit.

Threshold `0.85`:

- Median ratio versus fresh baseline: `0.8616x`.
- Geometric mean versus fresh baseline: `0.8296x`.
- Median ratio versus current threshold: `1.0436x`.
- Geometric mean versus current threshold: `1.0929x`.
- Won `6/8` tasks versus current.
- Worst task ratio versus current: `0.9114x`.
- Passed the minimum gate but lost the predeclared selection rule to `0.75`.
- Its larger low-acceptance gains were offset by unstable middle/high
  acceptance behavior, including `0.9114x` on `humaneval_137`.

Decision:

- Select threshold `0.75` for the next experiment.
- Reject threshold `0.85` as the general candidate. It is more aggressive but
  less consistent and has a lower geometric ratio versus baseline.
- Do not promote `0.75` as the runtime default yet:
  - all eight candidate tasks remain slower than ordinary baseline;
  - one run per task establishes broad direction, not a final workload result;
  - the selected subset intentionally overrepresents difficult and
    decision-sensitive tasks.
- The scheduler result is large enough that returning immediately to verifier
  architecture would be premature. Run one frozen 32-task baseline-versus-0.75
  confirmation first.
- The full confirmation should:
  - use the same 32 HumanEval tasks and frozen exact outputs;
  - pair ordinary baseline with threshold `0.75`;
  - alternate baseline-first and runtime-first order;
  - use one measured pair per task and excluded global warmups;
  - enable no instrumentation;
  - report within-run candidate/baseline ratios as authoritative;
  - compare descriptively with the completed threshold-`0.455` study.
- If the full study remains clearly below parity, freeze scheduler tuning at
  `0.75` and return to the Metal exact-verifier architecture. If it approaches
  or crosses parity on a meaningful fraction of tasks, audit threshold-`0.75`
  acceptance and component costs before deciding whether to promote it.

## Phase 1.16: Full threshold-0.75 HumanEval confirmation

Implementation:

- Added
  `speed-bench/run_dspark_humaneval_threshold075_throughput.py`.
- The runner binds to:
  - the frozen 32-task threshold-`0.455` throughput artifact;
  - the completed aggressive scheduler gate that selected threshold `0.75`.
- It rejects drift in:
  - model and sidecar paths;
  - context, output length, seed, non-thinking mode, and instrumentation state;
  - frozen HumanEval selection and prompts;
  - aggressive-gate tasks, thresholds, selected candidate, and source
    throughput reference.
- Every baseline and threshold-`0.75` output must match the frozen exact
  artifact byte-for-byte.

Execution protocol:

- `32` frozen HumanEval tasks.
- One measured baseline/runtime pair per task.
- Baseline-first and runtime-first order alternate by task.
- Two excluded global warmup pairs use opposite orders.
- Total process count: `68`:
  - four excluded warmup processes;
  - `64` measured processes.
- Runtime mode enables only:
  - `DS4_DSPARK_GPU_RUNTIME=1`;
  - `DS4_DSPARK_MULTI_COMMIT=1`;
  - `DS4_DSPARK_CONFIDENCE_THRESHOLD=0.75`.
- No runtime stats, acceptance audit, trace, diagnostics, profiler, or fast
  verifier is enabled.

Predeclared decision gates:

- Within-run threshold-`0.75`/baseline ratios are authoritative.
- Threshold-`0.455` values from the frozen prior study are descriptive
  cross-run context only.
- Broad scheduler confirmation requires:
  - geometric task-level movement versus `0.455` at least `1.05x`;
  - at least `24/32` improved tasks;
  - no task below `0.80x` movement.
- The result is treated as near parity if either:
  - fresh threshold-`0.75`/baseline geometric mean is at least `0.95x`; or
  - at least eight tasks are faster than baseline.
- If near parity passes, the next phase is a threshold-`0.75` acceptance and
  component-cost audit.
- If near parity fails, freeze scheduler tuning at `0.75` and return to the
  exact Metal verifier architecture.

Validation:

- The dedicated tests cover:
  - frozen policy and decision constants;
  - alternating measured order;
  - uninstrumented threshold-`0.75` environment;
  - scheduler-confirmation pass/fail accounting;
  - near-parity next-path selection;
  - authoritative versus historical report language.
- The dry run validated both source artifacts, all 32 frozen tasks, command
  environments, alternating order, and the intended 68-process schedule.
- Threshold `0.75` already passed the full five-case runtime correctness matrix
  in Phase 1.15.
- Codex will not run the timed confirmation.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_threshold075_throughput.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json \
  --gate-reference \
  speed-bench/local-runs/humaneval-aggressive-scheduler-20260716-151432/summary.json \
  --confirm-idle
```

Completed full threshold-0.75 confirmation:

- Raw artifact:
  `speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112`.
- The run was collected at clean commit `7b954be`.
- The artifact contains exactly:
  - `32` baseline rows;
  - `32` threshold-`0.75` rows;
  - `16` baseline-first pairs;
  - `16` runtime-first pairs.
- Every task has one output hash across both modes; all `32` threshold-`0.75`
  outputs matched ordinary baseline byte-for-byte.
- No thermal or performance warning was recorded. The process snapshot still
  included Logitech, WindowServer, Ghostty, Stats, and system activity, but the
  alternating pair design and broad task direction are decisive.

Authoritative threshold-0.75 result:

- Baseline median: `22.95 t/s`.
- Threshold-`0.75` median: `19.80 t/s`.
- Ratio of medians: `0.8632x`.
- Median paired ratio: `0.8627x`.
- Geometric mean paired ratio: `0.8634x`.
- Interquartile range: `0.8332x-0.8859x`.
- Full range: `0.7970x-0.9702x`.
- Faster/equal/slower tasks: `0/0/32`.
- Six tasks reached at least `0.90x`; two reached at least `0.95x`.
- The best task was `humaneval_053` at `0.9702x`.
- The worst task was `humaneval_000` at `0.7970x`.

Confirmed movement from threshold `0.455`:

- Prior median paired ratio: `0.8081x`.
- Median task-level movement: `1.0712x`.
- Geometric task-level movement: `1.0915x`.
- Improved/equal/regressed tasks: `28/0/4`.
- Worst task movement: `0.9524x`.
- The broad scheduler confirmation gate passed.
- Relative to the prior `0.8081x` median, threshold `0.75` closes roughly
  `28.5%` of the remaining median gap to baseline.

Final scheduler decision:

- Freeze confidence-scheduler tuning at threshold `0.75`.
- Threshold `0.75` is the selected research policy for subsequent DSpark
  optimization and diagnostics.
- Do not promote it as the user-facing runtime default yet:
  - no task is faster than baseline;
  - the workload geometric mean remains `0.8634x`;
  - the near-parity gate failed.
- Do not test more fixed thresholds or per-task policies. The full study
  confirms diminishing scheduler returns and removes scheduler selection as
  the primary uncertainty.

Remaining performance target:

- At the end-to-end workload level, parity requires about `1.158x` the current
  threshold-`0.75` throughput, equivalent to removing `13.7%` of current total
  generation time.
- Because sidecar and host costs remain, the exact target-verifier reduction
  required will be larger than `13.7%`.
- The next verifier phase should begin with a threshold-`0.75` stats-only cost
  audit to measure:
  - target milliseconds per emitted token;
  - sidecar milliseconds per emitted token;
  - verifier widths and positions;
  - partial/full batch outcomes;
  - the exact target-time scale required for parity under the frozen `0.75`
    schedule.
- After that audit, optimize the exact Metal verifier architecture against the
  measured target rather than continuing scheduler or isolated micro-kernel
  tuning.

### Phase 1.17: Threshold-0.75 exact-verifier cost audit

Prepared a dedicated stats-only audit:

- Runner:
  `speed-bench/run_dspark_humaneval_threshold075_cost_audit.py`.
- Model-free tests:
  `tests/test_dspark_threshold075_cost_audit.py`.
- Frozen policy:
  - confidence threshold `0.75`;
  - exact target verification;
  - no fast verifier;
  - no fresh baseline or throughput pair;
  - no acceptance audit, oracle trace, or layer profiler.
- The completed threshold-`0.75` confirmation is the sole throughput
  reference. The loader requires:
  - the expected experiment kind and frozen 32-task selection;
  - threshold `0.75`;
  - a passed scheduler-confirmation gate;
  - the verifier-optimization next-path decision;
  - exact model paths and generation configuration;
  - one baseline and one runtime row per task;
  - matching baseline/runtime output hashes;
  - prompt and output bytes that still match the artifact.
- Each fresh stats runtime must reproduce the frozen output byte-for-byte.
- The report will aggregate:
  - frozen baseline and threshold-`0.75` generation-time budgets;
  - target, generation-sidecar, and residual milliseconds per emitted token;
  - target evals per emitted token and target positions per eval;
  - full, partial, and fallback verifier outcomes;
  - verifier-width target economics;
  - scheduler-width sidecar economics;
  - the target-time scale required for parity.
- Two parity scales are deliberately separated:
  - end-to-end calibrated: assigns the frozen measured deficit to target
    verification while holding sidecar and residual costs fixed;
  - component accounted: uses only fresh target and sidecar timings.
- Residual is a cross-run quantity, not a direct host timer. It includes host
  work plus mismatch between frozen uninstrumented generation time and fresh
  instrumented component timing.
- This is a diagnostic run and does not require an idle machine.

Harness validation:

- `python3 -m unittest discover -s tests -p 'test_dspark_*.py'`
  passes all `74` DSpark model-free tests.
- The real-artifact dry run validated:
  - all `32` frozen tasks and prompt bytes;
  - one baseline and one runtime reference row per task;
  - exact output hashes;
  - threshold `0.75`;
  - the expected model paths and generation configuration;
  - stats enabled without oracle trace or fast verification;
  - the intended `32`-process diagnostic schedule.

Stats-only run command:

```sh
python3 speed-bench/run_dspark_humaneval_threshold075_cost_audit.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-ready
```

Completed threshold-0.75 exact-verifier cost audit:

- Raw artifact:
  `speed-bench/local-runs/humaneval-threshold075-cost-20260716-161553`.
- The run was collected from clean commit `a4b50d8`.
- All `32` instrumented runtime outputs matched the frozen uninstrumented
  threshold-`0.75` artifact byte-for-byte.
- The artifact contains one stats row for every frozen task.
- Stats reconciliation passed:
  - `1360` total target evals equal the verifier-width histogram;
  - `760` width-2-through-width-5 evals equal the batch-attempt count;
  - `684` batches completed fully;
  - `76` batches completed partially;
  - no batch fallback occurred;
  - no fast-verifier call occurred;
  - verifier-width target-time shares sum to exactly one.

Aggregate frozen budget and fresh component timing:

- Frozen ordinary baseline: `43.543 ms/emitted`.
- Frozen threshold-`0.75` runtime: `50.432 ms/emitted`.
- End-to-end deficit: `6.889 ms/emitted`.
- Fresh exact target verification: `46.450 ms/emitted`.
- Fresh generation sidecar: `8.314 ms/emitted`.
- Cross-run residual: `-4.332 ms/emitted`.
- Frozen paired geometric mean: `0.8634x`.
- Pooled frozen generation-time ratio: `0.8634x`.
- Target evals per emitted token: `0.3854`.
- Target positions per eval: `2.706`.

Parity target:

- Assigning only the frozen end-to-end deficit to the verifier requires target
  time scale `0.8517x`, or a `14.8%` reduction.
- This lowers exact target cost from `46.450` to about
  `39.56 ms/emitted`.
- Fresh target plus sidecar component accounting requires target scale
  `0.7584x`, or a `24.2%` reduction.
- This conservative target is about `35.23 ms/emitted`.
- The negative residual proves that fresh synchronized component totals are
  not additive with frozen uninstrumented wall time. Treat `14.8%` as the
  optimistic minimum and `24.2%` as the conservative engineering target, not
  as two competing exact predictions.

Verifier-width economics:

- Width 1:
  - `600` evals;
  - `47.904 ms/eval`;
  - `17.5%` of target time.
- Width 2:
  - `134` evals;
  - `48.398 ms/position`;
  - `7.9%` of target time.
- Width 3:
  - `113` evals;
  - `44.294 ms/position`;
  - `9.2%` of target time.
- Width 4:
  - `92` evals;
  - `45.629 ms/position`;
  - `10.2%` of target time.
- Width 5:
  - `421` evals;
  - `42.947 ms/position`;
  - `55.1%` of target time.
- Widths 2 through 5 account for `82.5%` of target time.
- Width 5 lowers cost per target position by only about `10.3%` relative to
  width 1. The exact multi-row verifier therefore scales weakly with proposal
  width even though it avoids target-eval calls.

Decision:

- The remaining gap is not caused by fallback handling:
  every multi-position verifier attempt completed through the exact batch
  path, and no fast or serial recovery path was used.
- Scheduler control flow is no longer the primary target.
- Width-5 exact verification is the largest single optimization surface, but
  widths 2 through 4 are collectively material. Optimize shared multi-row
  layer execution rather than a width-5-only special case.
- Do not return to host allocation, command-boundary, isolated sidecar, or
  confidence-threshold micro-tuning.

Next bounded phase:

- Add a width-stratified exact-layer diagnostic under threshold `0.75`.
- Use frozen task `humaneval_079`, which produced:
  - `20` width-5 verifier evals;
  - at least one eval at every width from 2 through 4;
  - `128` emitted tokens.
- Profile representative target layers `0`, `21`, and `42`.
- Group the existing exact-layer records by their `tokens` field so attention
  preparation, serial attention tail, and exact FFN cost can be compared at
  widths 2 through 5.
- Require the profiled output to match the frozen threshold-`0.75` artifact.
- This diagnostic should identify which promoted layer component fails to
  amortize with width before another runtime candidate is implemented.

### Phase 1.18: Width-stratified exact-layer profile

Prepared a dedicated synchronized diagnostic:

- Runner:
  `speed-bench/run_dspark_threshold075_width_layer_profile.py`.
- Model-free tests:
  `tests/test_dspark_threshold075_width_layer_profile.py`.
- Frozen task: `humaneval_079`.
- Frozen threshold: `0.75`.
- Profiled layers: `0`, `21`, and `42`.
- Reported verifier widths: `2`, `3`, `4`, and `5`.
- The task was selected from the completed cost audit because it has:
  - one width-2 target eval;
  - one width-3 target eval;
  - four width-4 target evals;
  - twenty width-5 target evals.
- Each layer process enables:
  - exact DSpark runtime;
  - multi-commit;
  - runtime stats;
  - threshold `0.75`;
  - exact-layer synchronized profiling for one layer.
- The runner rejects:
  - any output that differs from the frozen threshold-`0.75` artifact;
  - any change in emitted tokens, target evals, target positions, batch
    outcomes, scheduler-width counts, or verifier-width counts;
  - incomplete attention-pre, serial-tail, or exact-FFN stage schedules.
- No runtime implementation candidate is enabled.

Harness validation:

- All `79` DSpark model-free tests pass.
- The real-reference dry run validated:
  - the threshold-`0.75` throughput artifact;
  - the completed cost-audit artifact;
  - task `humaneval_079`;
  - the required nonzero width-2-through-width-5 counts;
  - the three Metal profile commands and synchronized stage contract.

Completed width-stratified exact-layer profile:

- Raw artifact:
  `speed-bench/local-runs/threshold075-width-layer-20260716-193143`.
- The run was collected from clean commit `cce5f5c`.
- All three profiled outputs matched the frozen threshold-`0.75` artifact
  byte-for-byte.
- Every layer process reproduced:
  - `128` emitted tokens;
  - `34` target evals;
  - width histogram `8/1/1/4/20` for widths `1/2/3/4/5`;
  - zero verifier fallback;
  - zero fast-verifier call.

Sampled layer cost by verifier width:

- Width 2: `4.669 ms/row` across layers `0`, `21`, and `42`.
- Width 3: `3.578 ms/row`, or `0.766x` width 2.
- Width 4: `3.295 ms/row`, or `0.706x` width 2.
- Width 5: `3.098 ms/row`, or `0.663x` width 2.
- Multi-row exact execution therefore amortizes real work, but the reduction
  from width 4 to width 5 is only about `6.0%` per row.

Width-5 sampled component totals:

- Attention preparation: `0.584 ms/row`, `18.8%`.
- Serial attention tail: `1.341 ms/row`, `43.3%`.
- Exact FFN: `1.174 ms/row`, `37.9%`.

Width-2-to-width-5 per-row scaling:

- Attention preparation: `0.560x`.
- Serial attention tail: `0.700x`.
- Exact FFN: `0.685x`.
- The serial attention tail is both:
  - the largest width-5 sampled component;
  - the weakest-amortizing promoted component.

Layer detail at width 5:

- Layer 0:
  - total `0.893 ms/row`;
  - tail `0.319 ms/row`, or `35.7%`.
- Layer 21:
  - total `1.064 ms/row`;
  - tail `0.473 ms/row`, or `44.4%`.
- Layer 42:
  - total `1.141 ms/row`;
  - tail `0.549 ms/row`, or `48.1%`.
- Tail cost grows with layer depth while attention preparation and FFN remain
  comparatively stable.

Interpretation:

- Widths 2 and 3 have only one observation each and are directional.
- Width 5 has twenty observations and is the stable optimization target.
- Attention preparation has already captured most of the available
  cross-row amortization and should not be reopened.
- FFN remains material but is not the weakest-scaling component.
- The next exact-verifier candidate should be chosen from the late-layer
  serial attention tail.

Next bounded phase:

- Add a threshold-`0.75`, width-stratified serial-tail component profile for
  `humaneval_079`, starting with layer `42`.
- Reuse the existing tail boundaries:
  - KV/cache update;
  - compressor/indexer;
  - attention;
  - inverse RoPE;
  - projection A;
  - projection B plus HC.
- Map each one-row tail record back to its enclosing proposal batch, then
  report component medians separately for widths `2`, `3`, `4`, and `5`.
- Require the output and all proposal-width counts to match the completed
  threshold-`0.75` artifacts.
- Use the width-5 distribution and scaling result to choose one architectural
  runtime candidate. Do not retry the previously rejected whole-suffix batch
  implementation without a materially different operation schedule.

### Phase 1.19: Width-stratified layer-42 serial-tail profile

Prepared the dedicated tail diagnostic:

- Runner:
  `speed-bench/run_dspark_threshold075_width_tail_profile.py`.
- Model-free tests:
  `tests/test_dspark_threshold075_width_tail_profile.py`.
- Frozen task: `humaneval_079`.
- Frozen layer: `42`.
- Frozen threshold: `0.75`.
- Required references:
  - full threshold-`0.75` throughput confirmation;
  - threshold-`0.75` cost audit;
  - completed width-stratified exact-layer profile.
- One synchronized profile process enables the existing exact-tail component
  boundaries plus runtime stats.
- Batch mapping uses event sequence:
  - attention-pre and FFN control records define ordered `(start, width)`
    batches;
  - each tail component's one-row events are consumed in that same order;
  - every consumed row must match `start..start+width-1`.
- Sequence mapping is required because consecutive target evaluations may
  overlap positions after partial acceptance, making a position-only lookup
  ambiguous.
- The report will separate:
  - KV/cache update;
  - compressor/indexer;
  - attention;
  - inverse RoPE;
  - projection A;
  - projection B plus HC.
- Any output drift, width-count drift, batch-outcome drift, or incomplete tail
  stage fails the diagnostic.

Harness validation:

- `python3 -m unittest discover -s tests -p 'test_dspark_*.py'`:
  `85` tests passed.
- `python3 -m py_compile` passed for the runner and its model-free test.
- `git diff --check` passed.
- A real-reference `--dry-run --allow-dirty` completed successfully against:
  - throughput:
    `humaneval-threshold075-throughput-32-20260716-155112`;
  - cost:
    `humaneval-threshold075-cost-20260716-161553`;
  - width-layer:
    `threshold075-width-layer-20260716-193143`.

Diagnostic result:

- Artifact:
  `speed-bench/local-runs/threshold075-width-tail-20260716-200622`.
- Clean source commit:
  `f4f42c96f2379a8161debb67010f92ddd170c03b`.
- The profiled output matched the frozen threshold-`0.75` HumanEval output
  byte-for-byte.
- The exact verifier schedule matched the frozen cost audit:
  - width 2: `1` evaluation;
  - width 3: `1`;
  - width 4: `4`;
  - width 5: `20`.
- Stable width-5 component medians at layer 42:
  - KV/cache update: `0.277 ms/row`, `13.2%`;
  - compressor/indexer: `0.388 ms/row`, `18.5%`;
  - attention: `0.455 ms/row`, `21.7%`;
  - inverse RoPE: `0.271 ms/row`, `12.9%`;
  - projection A: `0.351 ms/row`, `16.7%`;
  - projection B plus HC: `0.359 ms/row`, `17.1%`.
- Projection A and projection B plus HC together account for `33.8%` of the
  width-5 synchronized tail, larger than any individual component.
- Width-5 per-row cost is effectively unchanged from the single width-2
  observation for KV/cache, compressor/indexer, attention, and inverse RoPE.
  Projection A and projection B plus HC are each about `1.10x` the width-2
  observation. The retained tail therefore has no proposal-width amortization:
  it still executes one complete serial row at a time.
- The sole width-3 batch was a broad timing outlier (`4.415 ms/row`) and is not
  used for an optimization decision. Widths 2 and 3 remain directional only.
- The machine snapshot contained substantial unrelated OBS, camera, Logitech,
  and system activity. This is acceptable for a synchronized attribution
  diagnostic but reinforces that absolute component milliseconds are not
  throughput measurements.

Interpretation:

- Do not revive the retired whole-suffix batch schedule. Earlier direct-write
  attribution showed that separating all attention cores from all output
  projections made the candidate core alone slower than the complete retained
  serial tail.
- Keep the interleaved per-row schedule and optimize inside its existing Metal
  kernels.
- Attention is the largest individual component, but the two exact output
  projections are the largest combined tractable block and already have a
  byte-exact output-row reuse specialization.

Next bounded phase:

- Add a default-off Metal-only NR8 candidate for both exact attention-output
  kernels:
  - projection A;
  - projection B plus HC.
- Compare it against the promoted NR4 path. Preserve every output row's Q8
  block traversal, accumulation, reduction, and HC arithmetic; only reuse each
  activation load across eight adjacent output rows instead of four.
- Require explicit NR8 and NR4 correctness matrices, trace confirmation that
  both kernels selected NR8, normal Metal and CPU builds, DSpark
  validation/shape binding, and a user-run paired ablation.
- Treat this as a bounded kernel experiment, not a parity strategy by itself:
  the cost audit still requires a much larger verifier reduction than any
  single projection micro-optimization is likely to provide.

## Phase 1.20: Exact Attention-Output NR8 Candidate

Date: 2026-07-16.

Goal:

- Test whether the retained serial exact-attention output tail benefits from
  reusing each activation load across eight adjacent output rows instead of
  the promoted four-row schedule.
- Keep the interleaved per-proposal-row verifier schedule unchanged.

Implementation:

- Added default-off `DS4_DSPARK_EXACT_ATTN_OUT_NR8=1`.
- Added NR8 Metal entry points for:
  - `kernel_dsv4_attn_out_low_q8_0_f32`;
  - `kernel_dsv4_q8_hc_expand4_q8_0`.
- The existing templated Q8 implementations now instantiate `NR0=8`.
- When NR8 is requested and the output dimension is divisible by eight, it
  takes precedence over promoted NR4. Otherwise the existing NR4/NR2 selection
  remains unchanged.
- Projection A uses `32 * 8 * sizeof(float)` threadgroup scratch and dispatches
  one threadgroup per eight output rows.
- Projection B plus HC uses the same eight-row Q8 reduction shape and retains
  its existing per-output HC arithmetic.
- For every output row, NR8 preserves:
  - Q8 block and lane assignment;
  - block traversal order;
  - scalar accumulation order;
  - SIMD reduction order;
  - threadgroup reduction order;
  - attention-output and HC expansion arithmetic.
- Added correctness-only
  `DS4_DSPARK_EXACT_ATTN_OUT_NR8_TRACE=1`. It reports separately when
  projection A and projection B plus HC select NR8.
- Promoted NR4 remains the default. NR8 is not enabled unless explicitly
  requested.

Correctness and benchmark plumbing:

- Added `DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR8=1` to the runtime correctness
  matrix. It:
  - requires exact attention-pre runtime;
  - rejects simultaneous explicit NR4 control;
  - enables NR8 and its trace;
  - requires both projection traces before passing.
- Added
  `tests/test_dspark_exact_attention_output_nr8.py`.
- Added
  `--exact-attention-output-nr8-ablation` to the paired benchmark harness:
  - reference: ordinary exact DSpark with promoted NR4;
  - candidate: only `DS4_DSPARK_EXACT_ATTN_OUT_NR8=1`;
  - Metal is explicit;
  - runtime stats, traces, and diagnostics are disabled;
  - outputs must match byte-for-byte;
  - Codex does not run the timed benchmark.

Validation:

```sh
make -j4
python3 -m py_compile \
  speed-bench/run_dspark_comparison.py \
  tests/test_dspark_exact_attention_output_nr8.py
python3 -m unittest discover -s tests -p 'test_dspark_*.py'
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR8=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR4=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
make ds4_cpu.o ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-output-nr8-ablation \
  --dry-run \
  --allow-dirty
git diff --check
```

Validation result:

- Metal build passed and both NR8 shader entry points compiled during the
  runtime correctness matrix.
- All `90` DSpark model-free tests passed.
- DSpark validation and shape binding passed.
- Explicit NR8 passed reasoning, Italian, medium-context, rolling-window, and
  resumed two-turn chat byte-for-byte.
- NR8 traces confirmed that both projection A and projection B plus HC used
  their eight-row kernels.
- Explicit promoted NR4 passed the same five-case matrix byte-for-byte and both
  NR4 traces remained present.
- The benchmark dry run emitted the intended uninstrumented pair:
  - promoted NR4 reference with no NR override;
  - NR8 candidate with only
    `DS4_DSPARK_EXACT_ATTN_OUT_NR8=1`.
- Codex ran no timed throughput benchmark.

User-run gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-output-nr8-ablation \
  --confirm-idle
```

Decision rule:

- Require byte-identical outputs in every pair.
- Promote NR8 only if the paired direction is consistently positive and large
  enough to justify doubling per-threadgroup output-row state.
- A neutral or negative result retires NR8 while leaving promoted NR4
  unchanged.

User-run throughput result:

- Artifact:
  `speed-bench/local-runs/20260716-204606`.
- Clean source commit:
  `cd3ece3cde944674779df8964b5f4fe31e1e7e52`.
- Promoted NR4 median: `23.55 t/s`.
- Opt-in NR8 median: `22.82 t/s`.
- Ratio of medians: `0.9690x`.
- Median paired ratio: `0.9667x`, or `-3.1%`.
- Individual paired ratios:
  - `0.9633x`;
  - `0.9667x`;
  - `0.9724x`.
- Every warmup and measured output had the same SHA-256:
  `86f4851c044b82fffe568644343670a83ea6815b70c160b5a28a0fb357c52998`.
- Prefill was also directionally lower for NR8, but this gate is decided by
  generation throughput.
- The machine snapshot contained substantial Logitech, OBS, camera, and
  WindowServer activity. Alternating order still produced three negative
  pairs with a narrow enough spread that the result is not plausibly an order
  artifact.

Decision:

- Reject NR8 for promotion.
- Keep promoted NR4 as the exact attention-output default.
- Retain NR8 only as default-off research code and a reproducible upper-row
  reuse experiment.
- Do not test NR8 on HumanEval or cross-domain workloads. A local `-3.1%`
  result is sufficiently negative to close the candidate.
- The likely cause is occupancy pressure from doubling per-threadgroup output
  accumulators and scratch from four to eight rows. The extra activation reuse
  does not compensate on M1 Ultra.
- Treat NR4 as the practical output-row reuse optimum for the current kernels.
  Do not spend another phase on NR6/NR8-style projection-row tuning.

Next bounded phase:

- Move away from attention-output projection tuning.
- Audit the retained Metal dense-mixed attention path, which is the dominant
  layer-42 attention mode on the existing exact-attention transition profile:
  `201` dense-mixed rows versus `37` sparse-indexed rows.
- Compare its shader and dispatch contract with the already promoted
  sparse-indexed RB16-direct path and the raw decode path.
- Select one in-place, per-row dense-mixed attention candidate that preserves
  cache mutation order and exact verifier scheduling.
- Start with source/dispatch analysis and a correctness observer or
  synchronized component diagnostic if ownership is ambiguous. Do not create
  a timed benchmark until the candidate has a measured operation-level reason
  to help.

## Phase 1.22: dense-mixed FlashAttention attribution prepared

Source audit:

- The exact verifier already batches attention preparation and FFN work.
- The retained stateful attention tail remains serial because each proposal
  row mutates and observes the target KV/compressor state in order.
- At layer 42 on the established 8K transition fixture, dense-mixed attention
  is the dominant mode: `201` rows versus `37` sparse-indexed rows.
- Dense-mixed rows currently use the generic one-query gathered
  FlashAttention route. Each call may:
  - linearize a wrapped raw ring;
  - copy the raw F32 cache to contiguous F16 scratch;
  - copy compressed cache rows into the same scratch;
  - build and optionally copy a mask;
  - pad the key range;
  - run split-K FlashAttention;
  - run a separate reduction.
- The promoted sparse-indexed route instead reads raw and compressed cache
  state directly in a specialized one-token kernel. A direct dense-mixed
  kernel is therefore plausible, but its expected ownership must be measured
  before implementation.

Prepared diagnostic:

- Added default-off
  `DS4_METAL_FLASH_ATTN_GATHERED_PROFILE`.
- Added synchronized boundaries around:
  - `linearize_raw`;
  - `copy_raw`;
  - `copy_comp`;
  - `mask_fill`;
  - `mask_comp_copy`;
  - `pad`;
  - `attention_vec`;
  - `attention_reduce`.
- The profiler updates the active batched command-buffer pointer after every
  boundary so instrumentation does not leave the caller using a completed
  buffer.
- Added
  `speed-bench/run_dspark_dense_mixed_flash_profile.py`.
- The runner uses one uninstrumented exact reference and one synchronized
  layer-42 profile, requires byte-identical output, checks that gathered-call
  count equals dense-mixed row count, and emits per-stage ownership.
- Added model-free parser and report tests in
  `tests/test_dspark_dense_mixed_flash_profile.py`.
- No timed throughput candidate exists in this phase.

Validation:

```sh
make -j4
python3 -m py_compile \
  speed-bench/run_dspark_dense_mixed_flash_profile.py \
  tests/test_dspark_dense_mixed_flash_profile.py
python3 -m unittest discover -s tests -p 'test_dspark_*.py'
python3 speed-bench/run_dspark_dense_mixed_flash_profile.py \
  --dry-run \
  --allow-dirty
git diff --check
```

Validation result:

- Metal build passed.
- All `95` DSpark model-free tests passed.
- The dry run emitted an uninstrumented exact reference and a profile process
  with only the intended gathered FlashAttention diagnostic added.
- The runner records path, size, and modification time for the 81 GB target
  and 11 GB sidecar rather than hashing either model before execution.

Next command after committing the diagnostic:

```sh
python3 speed-bench/run_dspark_dense_mixed_flash_profile.py \
  --confirm-ready
```

Decision rule:

- If cache staging, masks, padding, and reduction own a material share, build
  one default-off direct dense-mixed one-token Metal kernel that reads target
  cache state in place.
- If the attention core itself overwhelmingly owns the synchronized call,
  avoid a large custom-kernel phase and return to a narrower verifier
  scheduling or target-layer candidate.

Diagnostic correction:

- The first synchronized attempt at commit `95bd902` preserved exact output
  but the report rejected the artifact:
  - gathered FlashAttention calls: `5970`;
  - layer-42 dense-mixed rows: `115`.
- Cause: the new Metal boundary was enabled for every target layer, while the
  exact-attention mode observer was correctly filtered to layer 42.
- This was an observer-scoping error, not a runtime mismatch.
- Added an explicit `profile_gathered` argument to the one-row decode-attention
  API:
  - ordinary target and sidecar callers pass `0`;
  - the exact verifier route passes `1` only when
    `DS4_METAL_DECODE_STAGE_PROFILE_LAYER` matches the current layer;
  - the Metal boundary additionally requires this per-call flag.
- This avoids synchronizing unrelated layers and makes the gathered-call count
  directly comparable with the layer-42 dense-mixed mode count.
- The rejected artifact is
  `speed-bench/local-runs/dense-mixed-flash-20260716-205854`.
- A second attempt correctly scoped the observer to layer 42 but still found
  `130` gathered calls versus `115` dense verifier rows.
- The remaining `15` calls were ordinary one-token target decode/fallback work
  at layer 42, which shares the same attention route but has no exact-attention
  mode record.
- The route now also receives an explicit exact-verifier call-site gate.
  Gathered boundaries require both:
  - the exact attention-tail observer to be active for the call;
  - the existing layer filter to match.
- The second rejected artifact is
  `speed-bench/local-runs/dense-mixed-flash-20260716-210300`.

Successful attribution:

- Artifact:
  `speed-bench/local-runs/dense-mixed-flash-20260716-210501`.
- Exact output matched the uninstrumented reference byte-for-byte.
- Dense-mixed exact rows: `115`.
- Gathered FlashAttention calls: `115`.
- Key range: `1127-1152`.
- Compressed-row range: `999-1024`.
- Mean synchronized call total: `2.938 ms`.

Mean synchronized ownership per call:

| stage | contribution | share |
|---|---:|---:|
| raw-ring linearization | 0.431 ms | 14.7% |
| raw F32 to F16 copy | 0.465 ms | 15.8% |
| compressed-cache copy | 0.355 ms | 12.1% |
| mask fill | 0.341 ms | 11.6% |
| padding | 0.392 ms | 13.4% |
| attention vector kernel | 0.511 ms | 17.4% |
| separate reduction | 0.443 ms | 15.1% |

Interpretation:

- Only `17.4%` of the synchronized gathered call belongs to the attention
  vector kernel itself.
- Cache staging, mask/pad work, and separate reduction own `82.6%`.
- Absolute synchronized times are not throughput measurements, but ownership
  is decisive enough to justify one direct dense-mixed candidate.

## Phase 1.23: direct dense-mixed Metal candidate prepared

Implementation:

- Added default-off `DS4_METAL_DENSE_MIXED_DIRECT=1`.
- Added
  `kernel_dsv4_dense_mixed_attention_heads8_rb16_direct`.
- The candidate is deliberately narrow:
  - one token;
  - no dense compressed mask;
  - no fused inverse-RoPE;
  - `head_dim=512`;
  - direct raw-ring reads;
  - direct compressed-cache reads;
  - sixteen-row threadgroup staging shared across eight heads;
  - one kernel with no gathered-cache copy, mask fill, padding dispatch, or
    separate reduction dispatch.
- Raw rows preserve logical ring order with
  `(raw_start + row) % raw_cap`.
- Compressed rows are consumed in ascending cache order.
- The kernel uses the same half Q/K/V conversion, online-softmax update, and
  sink handling as the validated indexed RB16-direct route.
- Unsupported or masked shapes retain gathered FlashAttention.
- Added default-off route trace
  `DS4_METAL_DENSE_MIXED_DIRECT_TRACE=1`.

Correctness and harness:

- Extended `tests/dspark_gpu_candidates_correctness.sh` with
  `DS4_TEST_DSPARK_DENSE_MIXED_DIRECT=1`.
- The gate requires the direct-route trace to appear.
- Added model-free source and benchmark-contract tests in
  `tests/test_dspark_dense_mixed_direct.py`.
- Added `--dense-mixed-direct-ablation` to the paired benchmark harness:
  - reference: ordinary exact DSpark with gathered FlashAttention;
  - candidate: only `DS4_METAL_DENSE_MIXED_DIRECT=1`;
  - Metal is explicit;
  - runtime stats, traces, and diagnostics are disabled;
  - outputs must match byte-for-byte;
  - Codex does not run the timed benchmark.

Validation:

```sh
make -j4
python3 -m py_compile \
  speed-bench/run_dspark_comparison.py \
  tests/test_dspark_dense_mixed_direct.py
python3 -m unittest discover -s tests -p 'test_dspark_*.py'
bash -n tests/dspark_gpu_candidates_correctness.sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_DENSE_MIXED_DIRECT=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
python3 speed-bench/run_dspark_comparison.py \
  --dense-mixed-direct-ablation \
  --dry-run \
  --allow-dirty
git diff --check
```

Validation result:

- Metal host build passed.
- The new Metal shader compiled at runtime.
- All `100` DSpark model-free tests passed.
- Reasoning, Italian, medium-context, rolling-window, and resumed two-turn
  chat outputs matched their ordinary baselines byte-for-byte.
- The correctness matrix confirmed the direct dense-mixed route engaged.
- A combined check with opt-in inverse-RoPE fusion preserved final output but
  exceeded that experiment's internal dense-attention observer bound.
  The candidate is therefore explicitly disabled when fused inverse-RoPE is
  active; that combination retains gathered FlashAttention.
- Re-running the combined matrix after this restriction passed all five prompt
  cases plus the raw, dense-mixed, and indexed inverse-RoPE observers.
- The benchmark dry run emitted the intended uninstrumented pair.
- Codex ran no timed throughput benchmark.

User-run gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --dense-mixed-direct-ablation \
  --confirm-idle
```

Decision rule:

- Require byte-identical output in every warmup and measured pair.
- Promote only on a clear positive paired direction. This candidate replaces
  several dispatches and large cache copies, so a merely neutral result is not
  enough to justify the extra specialized kernel.
- If positive locally, confirm on the 32-sample HumanEval scheduled workload
  before promotion because dense/sparse mode mix varies by prompt.

User-run throughput result:

- Artifact:
  `speed-bench/local-runs/20260716-211825`.
- Clean source commit:
  `552d66ee35c5a926891f69b9241effd503d6174f`.
- Gathered FlashAttention median: `23.93 t/s`.
- Direct dense-mixed median: `25.15 t/s`.
- Ratio of medians: `1.0510x`.
- Median paired ratio: `1.0523x`, or `+5.1%`.
- Individual paired ratios:
  - `1.0535x`;
  - `1.0523x`;
  - `1.0501x`.
- Every measured output had the same SHA-256:
  `86f4851c044b82fffe568644343670a83ea6815b70c160b5a28a0fb357c52998`.
- Runtime stats and all diagnostic instrumentation were disabled.
- No thermal warning was recorded before or after the gate.
- The machine snapshot still contained substantial Logitech and other desktop
  activity, but alternating order produced three positive pairs with only
  `0.34` percentage points between the minimum and maximum gain.

Decision:

- The candidate passes the local throughput gate.
- Do not promote from the short local fixture alone.
- Next run a paired confirmation on the frozen 32-sample HumanEval scheduled
  workload, comparing:
  - current scheduled exact DSpark with gathered dense-mixed attention;
  - the same scheduler and workload with only
    `DS4_METAL_DENSE_MIXED_DIRECT=1`.
- Require byte-identical output for every task.
- Promotion should require a positive aggregate direction and no meaningful
  low-acceptance regression. Dense-mixed opportunity depends on each task's
  dense/sparse transition mix, so task distribution matters more here than on
  the short fixture.

## Phase 1.24: HumanEval direct dense-mixed confirmation prepared

Purpose:

- Confirm that the short-fixture `+5.1%` direct dense-mixed result generalizes
  across the frozen 32-task HumanEval workload.
- Keep the already promoted threshold-`0.75` confidence scheduler fixed.
- Isolate only the dense-mixed attention implementation:
  - reference: gathered one-query FlashAttention;
  - candidate: `DS4_METAL_DENSE_MIXED_DIRECT=1`.
- This is the final distribution gate before deciding whether the direct
  dense-mixed route should become the default.

Harness:

- Added `speed-bench/run_dspark_humaneval_dense_mixed_direct.py`.
- Added model-free contract tests in
  `tests/test_dspark_humaneval_dense_mixed_direct.py`.
- Documented the user-run command in `speed-bench/README.md`.
- The runner requires the frozen threshold-`0.75` throughput artifact:
  `speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json`.
- The artifact supplies:
  - the exact 32-task deterministic selection;
  - each task's frozen byte-exact output;
  - the prior acceptance verify rate used for the low-acceptance subgroup.
- Every gathered and direct output must match that frozen output
  byte-for-byte.

Protocol:

- Metal backend only.
- Non-thinking, greedy generation, seed `1`.
- Context `16384`; output limit `128`.
- Exact target verification.
- Threshold `0.75` in both modes.
- Two excluded warmup pairs.
- One measured pair for each of 32 tasks.
- Measured order alternates gathered/direct and direct/gathered.
- Total: `68` uninstrumented model processes.
- Runtime stats, acceptance tracing, diagnostics, profilers, route traces, and
  the fast verifier are disabled.
- Codex does not run the timed benchmark.

Frozen promotion gate:

- Paired-ratio geometric mean at least `1.02x`.
- Direct faster on at least `24/32` tasks.
- No individual task below `0.95x`.
- For tasks with prior verify rate at most `0.65`, subgroup geometric mean at
  least `1.00x`.

Validation:

```sh
python3 -m py_compile \
  speed-bench/run_dspark_humaneval_dense_mixed_direct.py \
  tests/test_dspark_humaneval_dense_mixed_direct.py
python3 -m unittest discover -s tests -p 'test_dspark_*.py'
python3 speed-bench/run_dspark_humaneval_dense_mixed_direct.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --dry-run \
  --allow-dirty
git diff --check
```

Validation result:

- All `106` DSpark model-free tests passed.
- The frozen reference metadata, task selection, CSV, output files, and output
  hashes were accepted.
- The dry run emitted exactly `68` uninstrumented processes:
  - `4` excluded warmup processes;
  - `64` measured processes.
- The dry run showed threshold `0.75` in both modes and
  `DS4_METAL_DENSE_MIXED_DIRECT=1` only in the candidate.
- No prompts were materialized and no model process was executed.
- `git diff --check` passed.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_dense_mixed_direct.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-idle
```

## Phase 1.25: Dense-mixed correctness failure and exact fused-gather repair

HumanEval gate failure:

- The first 32-task confirmation attempt stopped during the first warmup.
- Artifact:
  `speed-bench/local-runs/humaneval-dense-mixed-direct-32-20260716-212748`.
- The gathered `humaneval_000` output exactly matched the frozen threshold
  `0.75` artifact:
  `cd533e5ddc1e8d344efc797dee4f6011109a4e7e15e610512624abeb103b9064`.
- The direct candidate produced a different output:
  `956ead742c855b93d35f336414e9c565cbb8431bf9492ac4cc85a580a895a1d3`.
- The divergence was semantic at the generated-token level, not whitespace:
  the candidate changed the implementation of `has_close_elements`.

Root cause:

- The original direct kernel computed equivalent attention with a sequential
  online-softmax scan.
- Gathered FlashAttention partitions keys across 32 workgroups, processes
  32-row chunks inside SIMD groups, emits partial `(output, sum, max)` state,
  and performs a separate reduction.
- The different floating-point reduction order was sufficient to cross a
  target argmax boundary on HumanEval even though the shorter correctness
  fixtures happened to retain the same generated tokens.
- Therefore the prior short-fixture `+5.1%` result belongs to a
  correctness-rejected implementation and cannot be used as evidence for the
  replacement.

Repair exploration:

- A first repair loaded caches directly while reproducing the split-K
  FlashAttention partial layout and reusing the exact reduction kernel.
- It restored the frozen `humaneval_000` output byte-for-byte.
- Its one-off correctness run was directionally much slower than gathered
  attention, so it was removed before asking for another timed benchmark.

Exact fused-gather replacement:

- Added `kernel_dsv4_dense_mixed_prepare_f16`.
- One preparation dispatch now:
  - reads the raw ring in logical order;
  - converts raw F32 rows to F16;
  - copies or converts compressed rows to F16;
  - fills the zero attention mask;
  - writes zero K/V and `-MAXHALF` mask values for the padded tail.
- The candidate then calls the unchanged
  `kernel_flash_attn_ext_vec_f16_dk512_dv512`.
- It also calls the unchanged `kernel_flash_attn_ext_vec_reduce`.
- Attention chunking, score arithmetic, sink placement, partial reduction,
  and final output arithmetic are therefore the same as the gathered
  reference.
- The compatibility switch remains
  `DS4_METAL_DENSE_MIXED_DIRECT=1`.
- The preferred benchmark spelling is now
  `--dense-mixed-fused-gather-ablation`; the old direct spelling remains an
  alias.

Correctness validation:

```sh
make -j4
python3 -m unittest discover -s tests -p 'test_dspark_*.py'
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_DENSE_MIXED_DIRECT=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
```

- Build passed.
- All `106` DSpark model-free tests passed.
- Reasoning, Italian, medium-context, rolling-window, and resumed two-turn
  chat outputs matched their ordinary baselines byte-for-byte.
- The correctness matrix confirmed
  `Metal dense mixed attention route=fused_gather`.
- A fresh `humaneval_000` candidate replay matched the frozen output exactly:
  `cd533e5ddc1e8d344efc797dee4f6011109a4e7e15e610512624abeb103b9064`.
- Codex ran only correctness reproductions and made no throughput claim.

Next gate:

- Do not resume the 68-process HumanEval confirmation yet.
- First rerun the short three-pair ablation because the implementation changed
  completely:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --dense-mixed-fused-gather-ablation \
  --confirm-idle
```

- If the fused-gather result is clearly positive and every output remains
  byte-identical, commit the result and then return to the frozen 32-task
  HumanEval confirmation.

User-run fused-gather throughput result:

- Artifact:
  `speed-bench/local-runs/20260716-215613`.
- Clean source commit:
  `660447f4dd197d69ea5b09ef8dbd6cd063e98b6b`.
- Gathered FlashAttention median: `24.01 t/s`.
- Fused-gather median: `25.04 t/s`.
- Ratio of medians: `1.0429x`.
- Median paired ratio: `1.0429x`, or `+4.3%`.
- Individual paired ratios:
  - `1.0472x`;
  - `1.0377x`;
  - `1.0429x`.
- Every warmup and measured output had the same SHA-256:
  `86f4851c044b82fffe568644343670a83ea6815b70c160b5a28a0fb357c52998`.
- Runtime stats and diagnostic instrumentation were disabled.
- No thermal or performance warning was recorded before or after the gate.
- The machine still had ordinary desktop interference, but alternating order
  produced three positive pairs with less than one percentage point between
  the minimum and maximum gain.

Decision:

- The exact fused-gather candidate passes the short throughput gate.
- The replacement retains roughly four-fifths of the rejected direct
  candidate's local gain while restoring byte-exact HumanEval behavior.
- Resume the frozen 32-task HumanEval confirmation:

```sh
python3 speed-bench/run_dspark_humaneval_dense_mixed_direct.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-idle
```

- The broad gate compares gathered versus fused-gather preparation with the
  threshold-`0.75` scheduler fixed in both modes.
- It must still satisfy the frozen distribution gate:
  - geometric mean at least `1.02x`;
  - wins on at least `24/32` tasks;
  - no task below `0.95x`;
  - low-acceptance subgroup geometric mean at least `1.00x`.

## Phase 1.26: Fused-gather HumanEval gate and outlier adjudication

Broad gate result:

- Artifact:
  `speed-bench/local-runs/humaneval-dense-mixed-direct-32-20260717-080040`.
- Clean source commit: `952b3f6dfbf70783bf6e0e604caaa57afbd02319`.
- Gathered median: `19.72 t/s`; fused-gather median: `21.67 t/s`.
- Ratio of medians: `1.0991x`; median paired ratio: `1.1012x`;
  geometric mean: `1.0949x`.
- Fused gather won `31/32` tasks.
- Low-acceptance subgroup geometric mean: `1.0794x` across ten tasks.
- Every output matched the frozen threshold-`0.75` exact artifact
  byte-for-byte.
- The broad promotion gate is formally **FAIL**, solely because
  `humaneval_121` measured `0.9064x`, below the frozen `0.95x` per-task floor.
- The other three criteria passed by wide margins. The initial machine
  snapshot recorded substantial ordinary desktop activity, and no thermal or
  performance warning was recorded.

Adjudication protocol:

- Do not alter or relabel the original broad result.
- Freeze `humaneval_121` as the sole failed task before collecting more data.
- Added `speed-bench/run_dspark_humaneval_dense_mixed_outlier.py`.
- The harness refuses a confirmation artifact unless:
  - it is the failed 32-task fused-gather gate;
  - every aggregate and subgroup criterion passed;
  - `humaneval_121` is the only task below `0.95x`;
  - the gathered and fused outputs have the same hash.
- Run two excluded warmup pairs and six measured pairs with exactly balanced
  gathered-first/fused-first order.
- Every output must still match the original frozen exact HumanEval artifact.
- Predeclared adjudication gate:
  - median paired ratio at least `1.02x`;
  - geometric paired ratio at least `1.02x`;
  - fused gather wins at least `4/6` pairs;
  - at least `5/6` pairs are at or above the original `0.95x` floor.
- Codex does not run the timed gate.

Harness validation:

- Python compilation passed.
- All `112` DSpark model-free tests passed.
- The frozen failed confirmation artifact, source throughput artifact, task
  selection, output hashes, and exact output files were accepted.
- The dry run selected only `humaneval_121` and emitted exactly `16` planned
  uninstrumented processes:
  - four excluded warmup processes;
  - twelve measured processes;
  - three measured pairs in each order.
- No prompt was materialized and no model process was executed.
- `git diff --check` passed.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_dense_mixed_outlier.py \
  --confirmation-reference \
  speed-bench/local-runs/humaneval-dense-mixed-direct-32-20260717-080040/summary.json \
  --confirm-idle
```

Adjudication result:

- Artifact:
  `speed-bench/local-runs/humaneval-dense-mixed-outlier-20260717-084845`.
- Clean source commit: `13fa5534d9e02b00efea3bc4a95fd446a8bf204d`.
- Gathered median: `19.38 t/s`; fused-gather median: `21.26 t/s`.
- Ratio of medians: `1.0967x`; median paired ratio: `1.0959x`;
  geometric mean: `1.1078x`.
- Fused gather won `6/6`; all six pairs exceeded the original `0.95x` floor.
- Five pairs formed a tight `1.0931x`-`1.0981x` cluster. The first pair's
  gathered arm was slower and raised that pair to `1.1729x`; it does not drive
  the median decision.
- Every measured output had SHA-256
  `40f76b822d120cad7af7616373522a79864ac07ee99af9cc7320d27ed6c0e0ba`.
- The artifact was produced from a clean tree with balanced order, no runtime
  instrumentation, and no thermal or performance warning.
- The predeclared adjudication gate passed every criterion.

Promotion:

- The original broad gate remains recorded as a formal failure, followed by
  this successful predeclared adjudication of its sole failed task.
- Fused gather is promoted to the normal Metal dense-mixed path.
- `DS4_METAL_DENSE_MIXED_GATHERED_LEGACY=1` restores the former gathered path.
- Historical `DS4_METAL_DENSE_MIXED_DIRECT=1` commands remain compatible and
  force fused gather if both controls are present.
- Dense-mixed benchmark/reference paths now set the legacy switch explicitly;
  their default/candidate arms do not silently collapse after promotion.
- The gathered-stage profiler explicitly selects the legacy route.

Promotion validation:

- Normal build passed.
- All `112` DSpark model-free tests passed.
- The comparison dry run uses legacy gathered as the reference and promoted
  default exact DSpark as the candidate, with diagnostics disabled.
- The default fused-gather correctness matrix passed reasoning, Italian,
  medium-context, rolling-window, and resumed two-turn chat outputs
  byte-for-byte; route tracing confirmed fused gather engaged.
- The explicit legacy-gathered rollback passed the same five-scenario matrix.
- No timed benchmark was run by Codex.

## Phase 1.27: Cumulative end-to-end throughput reassessment

Purpose:

- Measure ordinary target baseline versus the accumulated exact DSpark runtime
  after promoting fused-gather dense-mixed preparation.
- Reuse the exact 32-task threshold-`0.75` HumanEval workload so the result has
  a direct historical anchor.
- Keep fresh paired DSpark/baseline ratios separate from descriptive cross-run
  movement.
- Determine whether the accumulated runtime is below near parity, near parity,
  or at/above end-to-end parity before selecting another optimization.

Frozen historical reference:

- Artifact:
  `speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112`.
- Clean source commit: `7b954be938db1cd7daf7a37237bc13eb553d27d6`.
- Historical geometric paired ratio: `0.8634x`.
- Historical median paired ratio: `0.8627x`.
- The new runner validates:
  - experiment kind and clean source commit;
  - all 32 deterministic task selections;
  - model, sidecar, and binary paths;
  - context, token limit, non-thinking mode, seed, temperature, and threshold;
  - 64 measured rows and complete baseline/runtime pairs;
  - byte-identical pair hashes and every frozen output file;
  - each raw CSV ratio against its summary value.

New harness:

- Added `speed-bench/run_dspark_humaneval_cumulative_throughput.py`.
- Added model-free contract tests in
  `tests/test_dspark_humaneval_cumulative_throughput.py`.
- Protocol:
  - Metal backend only;
  - context `16384`;
  - output limit `128`;
  - non-thinking greedy generation, temperature `0`, seed `1`;
  - exact target verification;
  - explicit confidence threshold `0.75`;
  - current promoted runtime defaults;
  - no `DS4_METAL_DENSE_MIXED_DIRECT` force switch;
  - no `DS4_METAL_DENSE_MIXED_GATHERED_LEGACY` rollback switch;
  - two excluded warmup pairs;
  - one alternating measured pair per task;
  - `68` total uninstrumented model processes.
- Every current baseline and runtime output must match the frozen historical
  output byte-for-byte.
- Runtime stats, acceptance audit, traces, diagnostics, profilers, and the
  fast verifier are disabled.

Predeclared interpretation:

- Accumulated movement gate:
  - geometric current/historical task movement at least `1.05x`;
  - at least `24/32` tasks improve;
  - no task below `0.90x` movement.
- End-to-end outcome uses only fresh paired ratios:
  - below near parity: geometric mean below `0.95x`;
  - near parity: geometric mean at least `0.95x` but below `1.00x`;
  - parity or speedup: geometric mean at least `1.00x`.
- A historical movement pass is not an end-to-end speedup claim.

Validation before commit:

- Python compilation passed.
- Six new synthetic contract tests passed.
- All `118` DSpark model-free tests passed.
- The real historical artifact passed every provenance, protocol, hash, and
  ratio check.
- The dry run emitted exactly `68` uninstrumented processes:
  - four excluded warmup processes;
  - 64 measured processes;
  - 16 baseline-first and 16 runtime-first measured pairs.
- The dry-run runtime environment contains only the required DSpark runtime,
  multi-commit, and explicit threshold controls.
- No prompt was materialized and no model process was executed.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_cumulative_throughput.py \
  --historical-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-idle
```

Result:

- Artifact:
  `speed-bench/local-runs/humaneval-cumulative-throughput-32-20260717-092241`.
- Clean source commit: `f0edb16884aafd7e8ce95054da4d9a07117f5719`.
- Fresh baseline median: `24.65 t/s`; current exact DSpark median:
  `21.50 t/s`.
- Ratio of medians: `0.8722x`; median paired ratio: `0.8758x`;
  geometric paired ratio: `0.8826x`.
- Paired-ratio interquartile range: `0.8570x`-`0.9141x`; full range:
  `0.8015x`-`0.9756x`.
- Current DSpark was faster/equal/slower than ordinary baseline on `0/0/32`
  tasks. The geometric break-even gap is `11.7%`, so the result remains below
  the predeclared `0.95x` near-parity boundary.
- Historical-to-current task movement was `1.0073x` by median and `1.0222x`
  geometrically. `30/32` tasks improved and the worst task movement was
  `0.9916x`.
- The accumulated movement gate failed only its `1.05x` geometric threshold;
  the task-count and minimum-task criteria passed.
- Across the two benchmark sessions, absolute baseline throughput increased
  by about `7.4%` while DSpark throughput increased by about `8.6%`. The fresh
  paired ratio, rather than multiplying earlier isolated ablation ratios, is
  therefore authoritative. Relative to the old controlled gap, the run closes
  about `14%` of the distance to parity.
- The artifact contains 64 measured rows with balanced order, no output-hash
  mismatches, no enabled instrumentation markers, and no thermal or performance
  warning.

Decision:

- Fused gather remains promoted: the isolated optimization and its broad
  confirmation are valid, but machine/session movement reduced its observed
  contribution to the fresh end-to-end ratio.
- Do not claim near parity or speedup. The current best controlled statement is
  that exact DSpark improved from `0.8634x` to `0.8826x` geometrically on the
  frozen threshold-`0.75` HumanEval workload while preserving byte-exact output.
- Before another implementation experiment, refresh the stats-only cost audit
  on this exact post-promotion artifact. The older audit identified target
  verification as dominant, but its component timings predate fused gather and
  no longer quantify the remaining `11.7%` gap precisely.

## Phase 1.28: Post-promotion exact-verifier cost audit prepared

Purpose:

- Recalibrate the remaining `11.7%` end-to-end gap against the exact clean
  Phase 1.27 cumulative artifact.
- Preserve the frozen uninstrumented baseline and runtime generation times as
  the end-to-end budget while collecting fresh component timings from the
  current promoted runtime.
- Decide whether target verification remains the dominant optimization path
  after fused-gather promotion, and quantify the target-time reduction required
  for parity before changing verifier code again.

New harness:

- Added `speed-bench/run_dspark_humaneval_cumulative_cost_audit.py`.
- Added model-free contract tests in
  `tests/test_dspark_humaneval_cumulative_cost_audit.py`.
- The reference loader pins:
  - experiment `dspark_humaneval_cumulative_throughput`;
  - clean source commit `f0edb16884aafd7e8ce95054da4d9a07117f5719`;
  - the deterministic 32-task selection and threshold `0.75`;
  - promoted runtime defaults and the full uninstrumented protocol;
  - binary, target model, and DSpark sidecar paths;
  - 64 complete measured rows with alternating order;
  - prompt bytes, both output files, paired hashes, per-task t/s and ratios,
    and the aggregate geometric ratio.
- The audit executes exactly 32 stats-only exact-runtime processes. It runs no
  baseline, warmup, paired throughput, acceptance audit, oracle trace, layer
  profiler, fast verifier, or dense-mixed route experiment.
- Each instrumented output must match its frozen cumulative output byte-for-byte.
- The report reuses the established Phase 1.17 accounting for:
  - pooled frozen baseline/runtime time and deficit per emitted token;
  - fresh target, sidecar, and cross-run residual cost;
  - end-to-end-calibrated and component-accounted target scales for parity;
  - target evaluations and positions;
  - exact verifier-width timing and target-time share;
  - confidence-scheduler width progress and sidecar cost.

Validation:

- Python compilation passed.
- Nine targeted current and legacy cost-audit tests passed.
- All `122` DSpark model-free tests passed.
- The real Phase 1.27 artifact passed every provenance, protocol, prompt,
  output, row, per-task ratio, and aggregate-ratio check.
- The dry run emitted exactly 32 commands containing only the normal GPU
  runtime, multi-commit, stats, and threshold-`0.75` controls.
- No prompt was materialized and no model process was executed by Codex.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_cumulative_cost_audit.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260717-092241/summary.json \
  --confirm-ready
```

Result:

- Artifact:
  `speed-bench/local-runs/humaneval-cumulative-cost-20260717-094819`.
- Clean source commit: `5396ce554296cb8821c218a2694ea3bdf036d55a`.
- Every instrumented output matched the frozen cumulative artifact
  byte-for-byte. The run reported no thermal or performance warning.
- The frozen pooled end-to-end budget is:
  - ordinary baseline: `40.625 ms/emitted`;
  - current exact DSpark: `46.051 ms/emitted`;
  - deficit: `5.426 ms/emitted`;
  - pooled ratio: `0.8822x`, consistent with the `0.8826x` task geometric mean.
- Fresh component accounting is:
  - target verification: `39.010 ms/emitted`, `84.7%` of runtime;
  - generation sidecar: `6.831 ms/emitted`, `14.8%` of runtime;
  - residual: `0.210 ms/emitted`, about `0.5%` of runtime.
- End-to-end-calibrated parity requires target scale `0.8609x`, or a `13.9%`
  reduction. Component-only accounting independently gives `0.8663x`, or a
  `13.4%` reduction. Their close agreement makes `13%`-`14%` the current
  engineering target rather than the older audit's broad `14.8%`-`24.2%`
  interval.
- Eliminating all measured sidecar generation while leaving target and residual
  unchanged would produce only an optimistic `1.0358x` pooled ceiling. Sidecar
  remains material, but target verification offers the larger tractable budget.

Workload and verifier economics:

- The current and pre-promotion audits have identical behavior:
  - `3529` emitted tokens;
  - `1360` target evaluations;
  - `3680` target positions;
  - `760` multi-position attempts;
  - `684` full, `76` partial, and zero fallback outcomes.
- Target evaluations per emitted token remain `0.3854`; positions per target
  evaluation remain `2.706`. Fused gather did not change scheduler or verifier
  semantics.
- Width 1: `600` evals, `40.880 ms/position`, `17.8%` of target time.
- Width 2: `134` evals, `39.398 ms/position`, `7.7%` of target time.
- Width 3: `113` evals, `37.545 ms/position`, `9.2%` of target time.
- Width 4: `92` evals, `36.672 ms/position`, `9.8%` of target time.
- Width 5: `421` evals, `36.274 ms/position`, `55.5%` of target time.
- Widths 2 through 5 therefore consume `82.2%` of target time. Width 5 is only
  `11.3%` cheaper per position than width 1, so exact multi-row execution still
  amortizes weakly despite avoiding target-evaluation calls.

Cross-audit interpretation:

- Absolute fresh target timing fell from `46.450` to `39.010 ms/emitted`, and
  sidecar timing fell from `8.314` to `6.831 ms/emitted`. These are instrumented
  cross-session comparisons and include machine/session movement; they must not
  be attributed wholly to fused gather.
- The workload identity and near-zero new residual make the new audit the best
  current cost model. The old negative `-4.332 ms/emitted` residual is retired
  for forward planning.
- Scheduler policy, fallback recovery, and output correctness are not the
  remaining problem. The exact target verifier is.

Decision:

- Keep the threshold `0.75` schedule and all promoted defaults frozen.
- Do not repeat a full throughput or acceptance study yet.
- Refresh the Phase 1.18 width-stratified layer profile on the current runtime,
  using the cumulative throughput and cost artifacts as references. Profile
  layers `0`, `21`, and `42` on `humaneval_079`, whose verifier schedule is
  unchanged and contains widths `2` through `5`.
- Use the fresh attention-preparation, serial-tail, and exact-FFN split to choose
  the next shared multi-row verifier optimization. The prior split predates
  fused gather and should not decide the next implementation by itself.

## Phase 1.29: Post-promotion width-layer profile prepared

Purpose:

- Refresh the Phase 1.18 exact-layer attribution after fused-gather promotion.
- Keep the threshold-`0.75` schedule, selected task, sampled layers, stage
  boundaries, and width grouping unchanged so the structural comparison remains
  meaningful.
- Determine which shared exact-layer component now dominates width-5 target
  verification before implementing another verifier candidate.

Harness update:

- Extended `speed-bench/run_dspark_threshold075_width_layer_profile.py` rather
  than cloning its established parser and summarizer.
- The profiler now recognizes two explicit reference contracts:
  - the legacy Phase 1.17 threshold-`0.75` cost audit;
  - the Phase 1.28 post-promotion cumulative cost audit.
- The cumulative path requires:
  - cost experiment and analysis
    `dspark_humaneval_cumulative_exact_verifier_cost`;
  - clean cost-audit source commit
    `5396ce554296cb8821c218a2694ea3bdf036d55a`;
  - promoted defaults and the exact Phase 1.27 throughput reference;
  - the current 32-task selection and frozen output bytes.
- The cumulative throughput loader independently pins the clean Phase 1.27
  source commit, protocol, prompt bytes, output hashes, per-task ratios, and
  aggregate ratio.
- The selected cost row for `humaneval_079` still contains:
  - `128` emitted tokens;
  - `34` target evaluations and `129` target positions;
  - `26` multi-position attempts, `25` full, `1` partial, zero fallbacks;
  - width histogram `8/1/1/4/20` for widths `1/2/3/4/5`.
- Each layer process enables only current exact DSpark, runtime stats,
  threshold `0.75`, and the synchronized exact-layer profile for its selected
  layer. No fast verifier, trace, dense-mixed experiment switch, or runtime
  implementation candidate is enabled.
- The output report is labeled as post-promotion and records that it matched the
  frozen cumulative artifact byte-for-byte.
- Legacy throughput and cost artifacts remain accepted and retain the original
  report identity, preserving historical reproducibility.

Validation:

- Python compilation passed.
- Ten targeted width-layer and cumulative-cost tests passed.
- All `123` DSpark model-free tests passed.
- The current cumulative throughput and cost artifacts passed their complete
  reference checks.
- The post-promotion dry run emitted exactly three profile commands for layers
  `0`, `21`, and `42`, with no prompt materialization or inference run; only
  the existing model-metadata inspection was executed.
- A second dry run against the original Phase 1.17 references also passed,
  confirming backward compatibility.

User-run command:

```sh
python3 speed-bench/run_dspark_threshold075_width_layer_profile.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260717-092241/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260717-094819/summary.json \
  --confirm-ready
```

Result:

- Artifact:
  `speed-bench/local-runs/post-promotion-width-layer-20260717-100542`.
- Clean source commit: `fd506ee20c2b1db56830122492db3e56237512a4`.
- All three profiled outputs matched the frozen cumulative artifact
  byte-for-byte. The run reported no thermal or performance warning.
- Every layer reproduced the frozen `humaneval_079` verifier schedule:
  - width 2: `1` evaluation;
  - width 3: `1`;
  - width 4: `4`;
  - width 5: `20`.

Sampled exact-layer scaling:

- Width 2: `4.329 ms/row` across layers `0`, `21`, and `42`.
- Width 3: `3.302 ms/row`, or `0.763x` width 2.
- Width 4: `3.021 ms/row`, or `0.698x` width 2.
- Width 5: `2.853 ms/row`, or `0.659x` width 2.
- Width 5 remains the stable observation set; widths 2 and 3 each contain only
  one evaluation and are directional anchors.

Width-5 component split:

- Attention preparation: `0.569 ms/row`, `19.9%` of the sampled total.
- Serial attention tail: `1.120 ms/row`, `39.2%`.
- Exact FFN: `1.165 ms/row`, `40.8%`.
- Exact FFN is now the largest sampled stage, narrowly ahead of the serial
  attention tail.
- Width-2-to-width-5 per-row scaling is:
  - attention preparation: `0.563x`;
  - serial attention tail: `0.670x`;
  - exact FFN: `0.707x`.
- Exact FFN is therefore also the weakest-amortizing current stage.

Layer detail at width 5:

- Layer 0: total `0.893 ms/row`; preparation `0.192`, tail `0.317`, FFN
  `0.384`.
- Layer 21: total `0.932 ms/row`; preparation `0.185`, tail `0.356`, FFN
  `0.391`.
- Layer 42: total `1.028 ms/row`; preparation `0.192`, tail `0.446`, FFN
  `0.390`.
- FFN cost is notably stable across depth, while tail cost still grows in later
  layers. A shared exact-FFN improvement should therefore generalize across the
  target stack rather than optimize one late layer only.

Comparison with Phase 1.18:

- Sampled width-5 total: `3.098` to `2.853 ms/row`, `0.921x`.
- Attention preparation: `0.584` to `0.569`, `0.975x`.
- Serial attention tail: `1.341` to `1.120`, `0.835x`.
- Exact FFN: `1.174` to `1.165`, `0.992x`.
- These are synchronized diagnostics from different sessions, not isolated
  throughput deltas. Do not attribute the complete tail movement to fused
  gather or use the ratios as speedup claims.
- The structural conclusion is nonetheless strong: tail work became cheaper
  relative to FFN, while exact FFN is essentially unchanged and is now the
  largest weakly amortized component.

Decision:

- Keep fused gather, threshold `0.75`, and the exact verifier schedule frozen.
- Do not reopen attention preparation or repeat the broad throughput gate yet.
- Add a width-stratified exact-FFN sub-stage profile on `humaneval_079`, using
  the existing synchronized boundaries inside
  `metal_graph_encode_layer_ffn_batch`:
  - `hc_pre`;
  - `norm`;
  - `router`;
  - `shared_gate_up`;
  - `shared_down`;
  - `routed_moe`;
  - `hc_post`.
- Profile representative layers `0`, `21`, and `42`, map records to exact
  proposal batches, and report width-5 share plus width scaling. This should
  identify whether the next general candidate belongs in HC handling, routing,
  shared expert, or routed MoE execution.
- The sampled FFN share is about `40.8%`; closing the entire `13.9%` target-time
  gap through FFN alone would directionally require roughly one third of FFN
  cost. Expect a broader verifier program rather than assuming one FFN
  micro-optimization will reach parity by itself.

## Phase 1.30: Width-stratified exact-FFN profile prepared

Purpose:

- Split the new dominant `ffn_batch` stage into its existing synchronized
  implementation components.
- Preserve the exact Phase 1.29 task, layer sample, width distribution, and
  proposal schedule.
- Identify a shared Metal FFN component with enough width-5 cost or weak enough
  width scaling to justify the next runtime candidate.

New harness:

- Added `speed-bench/run_dspark_post_promotion_width_ffn_profile.py`.
- Added model-free tests in
  `tests/test_dspark_post_promotion_width_ffn_profile.py`.
- The harness requires and cross-validates:
  - Phase 1.27 cumulative throughput;
  - Phase 1.28 cumulative cost accounting;
  - Phase 1.29 post-promotion width-layer attribution.
- The width-layer reference must be experiment and analysis
  `dspark_post_promotion_width_stratified_exact_layer`, come from clean commit
  `fd506ee20c2b1db56830122492db3e56237512a4`, reference the exact supplied
  throughput and cost artifacts, and reproduce the frozen task, layers, widths,
  and width counts.
- The profile environment enables:
  - current exact DSpark with multi-commit;
  - runtime stats and confidence threshold `0.75`;
  - the outer exact-layer profiler for proposal-batch control records;
  - the existing Metal layer-stage profiler for FFN sub-stages.
- No new runtime instrumentation or target implementation code was added. The
  existing `metal_graph_encode_layer_ffn_batch` boundaries are:
  - `hc_pre`;
  - `norm`;
  - `router`;
  - `shared_gate_up`;
  - `shared_down`;
  - `routed_moe`;
  - `hc_post`.
- Batch mapping walks the profile event sequence. For every outer exact
  `ffn_batch` control at widths `2` through `5`, it requires exactly one of each
  internal stage with the same layer, position, and width before accepting the
  batch. Missing or duplicate stages fail the diagnostic.
- The report provides:
  - summed sub-stage and outer FFN medians by width;
  - a reconciliation ratio between the separately synchronized sub-stage sum
    and outer FFN control;
  - width-5 component shares and per-layer values;
  - width-2-to-width-5 amortization by component;
  - largest width-5 and weakest-amortizing sub-stages.
- Each profiled output and all runtime counters must match the frozen cumulative
  references exactly.

Validation:

- Python compilation passed.
- Eleven targeted FFN and width-layer tests passed.
- Synthetic tests cover exact-batch assignment, missing-stage rejection,
  component aggregation, control reconciliation, and report identity.
- All `128` DSpark model-free tests passed.
- The real cumulative throughput, cumulative cost, and post-promotion
  width-layer artifacts passed their full reference checks together.
- The dry run emitted exactly three synchronized commands for layers `0`, `21`,
  and `42`, with the two required profile controls and no runtime candidate.
- No prompt was materialized and no inference was run; only model metadata was
  inspected.

First-run mapper correction:

- The first user run stopped after collecting layer `0` with
  `layer 0 batch 2 duplicates hc_pre`.
- Artifact:
  `speed-bench/local-runs/post-promotion-width-ffn-20260717-103810`.
- The raw event stream revealed two same-position, same-width FFN stage sets:
  - the normal serial FFN work emitted before the exact attention batch;
  - the exact batched FFN work emitted after `attention_tail_serial` and before
    the outer exact `ffn_batch` control.
- Position, width, and the previous selected FFN control were therefore
  insufficient to identify the exact internal stages.
- The mapper now requires one matching exact `attention_tail_serial` control
  and accepts internal FFN stages only from the event interval between that
  control and the enclosing exact `ffn_batch` control.
- A regression test injects a complete same-key FFN stage set before the exact
  tail and verifies that it is ignored.
- The corrected mapper was applied offline to the failed run's real layer-0
  stderr and recovered exactly `26` batches:
  - width 2: `1`;
  - width 3: `1`;
  - width 4: `4`;
  - width 5: `20`.
- All `129` DSpark model-free tests pass after the correction.

User-run command:

```sh
python3 speed-bench/run_dspark_post_promotion_width_ffn_profile.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260717-092241/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260717-094819/summary.json \
  --width-layer-reference \
  speed-bench/local-runs/post-promotion-width-layer-20260717-100542/summary.json \
  --confirm-ready
```

Result:

- Artifact:
  `speed-bench/local-runs/post-promotion-width-ffn-20260717-104106`.
- Clean source commit: `862da2c3195df81db0a359a41718e039d1799700`.
- Every profiled output matched the frozen cumulative HumanEval artifact
  byte-for-byte. The FFN sub-stage sum reconciled to `0.984x..0.988x` of the
  enclosing exact FFN control.
- The exact width distribution was reproduced at all three sampled layers:
  width 2: `1` evaluation; width 3: `1`; width 4: `4`; width 5: `20`.

Sampled exact-FFN scaling:

- Width 2: `4.303 ms/row` across layers `0`, `21`, and `42`.
- Width 3: `2.933 ms/row`, or `0.682x` width 2.
- Width 4: `2.439 ms/row`, or `0.567x` width 2.
- Width 5: `2.114 ms/row`, or `0.491x` width 2.
- Widths 2 and 3 each contain one evaluation and remain directional anchors;
  the width-5 medians are the stable optimization guide.

Width-5 component split:

- `routed_moe`: `0.865 ms/row`, `40.9%`.
- `shared_gate_up`: `0.274 ms/row`, `13.0%`.
- `shared_down`: `0.245 ms/row`, `11.6%`.
- `router`: `0.221 ms/row`, `10.5%`.
- `hc_pre`: `0.194 ms/row`, `9.2%`.
- `hc_post`: `0.165 ms/row`, `7.8%`.
- `norm`: `0.150 ms/row`, `7.1%`.
- Routed MoE cost is stable across depth: `0.289`, `0.289`, and
  `0.287 ms/row` at layers `0`, `21`, and `42` respectively.

Width-2-to-width-5 scaling:

- `routed_moe`: `0.747x`.
- `shared_gate_up`: `0.431x`.
- `shared_down`: `0.422x`.
- `router`: `0.410x`.
- `hc_pre`: `0.396x`.
- `hc_post`: `0.375x`.
- `norm`: `0.325x`.
- Routed MoE is both the largest width-5 FFN component and the only component
  that fails to approach inverse-width amortization.

Implementation interpretation:

- The exact FFN path enters `metal_graph_routed_moe_decode_rows`, creates row
  views, and calls `ds4_gpu_routed_moe_one_tensor` once per proposal row.
- This composition is intentional. The generic batched routed-MoE path changed
  width-5 down/sum arithmetic and failed exactness; the one-token path preserves
  fused pair-SwiGLU and direct six-expert down/sum arithmetic.
- The row calls are made while the outer Metal command batch is open.
  `ds4_gpu_routed_moe_one_tensor` acquires that existing command buffer with
  `owned=0`, and its finish helper therefore does not submit one command buffer
  per row. A command-buffer sharing wrapper alone would not remove the measured
  bottleneck.
- The one-token path already exposes synchronized `gate_up`,
  `activation_weight`, `down`, and `sum` stage records through
  `DS4_METAL_MOE_ONE_STAGE_PROFILE`; selected model paths may also expose a
  gather stage.

Decision:

- Keep the exact row-wise arithmetic, threshold `0.75`, fused gather, prefix
  checkpoints, and the frozen HumanEval schedule unchanged.
- Do not switch the exact verifier back to the generic routed-MoE batch path.
- Next build a diagnostic-only mapper for the existing one-token routed-MoE
  stage records. As with Phase 1.30, accept only records inside the exact
  `attention_tail_serial` to `ffn_batch` interval so normal serial decode rows
  cannot contaminate the attribution.
- Report exact routed-MoE stage share for width-5 batches at layers `0`, `21`,
  and `42`. This should distinguish a gate/up kernel candidate from a
  down/sum candidate and avoid another arithmetic-risking broad batch rewrite.

## Phase 1.31: Exact routed-MoE stage profile prepared

Purpose:

- Attribute the dominant exact `routed_moe` FFN component without changing its
  arithmetic or enabling a runtime candidate.
- Separate one-token gate/up, activation-weight, down, and sum costs inside the
  same frozen post-promotion verifier schedule.
- Choose the next Metal candidate from measured inner structure rather than
  reopening the known-inexact generic width-5 routed-MoE batch path.

New harness:

- Added
  `speed-bench/run_dspark_post_promotion_width_routed_moe_profile.py`.
- Added model-free tests in
  `tests/test_dspark_post_promotion_width_routed_moe_profile.py`.
- The harness requires and cross-validates the Phase 1.27 cumulative
  throughput, Phase 1.28 cumulative cost, Phase 1.29 width-layer, and Phase
  1.30 width-FFN artifacts.
- The width-FFN reference must be the clean
  `862da2c3195df81db0a359a41718e039d1799700` artifact and must identify
  `routed_moe` as both its largest width-5 stage and weakest-amortizing stage.
- Each layer process enables the current exact runtime, stats, exact-layer and
  FFN stage controls, plus the existing
  `DS4_METAL_MOE_ONE_STAGE_PROFILE` boundary for the selected layer.
- Mapping first identifies the enclosing exact `attention_tail_serial` to
  `ffn_batch` interval, then accepts one-row MoE records only between its
  matching `router` and `routed_moe` controls. This excludes ordinary
  serial decode work with the same position and width.
- Every exact batch must contain exactly `width` copies of this ordered stage
  sequence:
  - `gate_up`;
  - `activation_weight`;
  - `down`;
  - `sum`.
- Every row must retain the same Metal path and gate/down types and report six
  selected expert pairs. Missing, duplicated, reordered, or unexpected stages
  fail the diagnostic.
- The report gives width-grouped inner and outer routed-MoE costs, width-5
  component shares, and layer detail for layers `0`, `21`, and `42`.
- Inner boundaries synchronize after each component and alter scheduling.
  Their values are attribution only, never throughput measurements.

Validation:

- Python compilation passed.
- Seven targeted routed-MoE tests cover environment isolation, raw record
  parsing, exact-row assignment, serial-record exclusion, missing-stage
  rejection, aggregation, and report identity.
- All `136` DSpark model-free tests passed.
- A dry run validated the complete real artifact chain and emitted exactly
  three commands for layers `0`, `21`, and `42`.
- The dry run materialized no prompt and ran no inference; only existing model
  metadata inspection was performed.
- `git diff --check` passed.

First-run mapper correction:

- The first user run stopped after collecting layer `0` with
  `layer 0 batch 1 routed-MoE stage sequence mismatch: []`.
- Artifact:
  `speed-bench/local-runs/post-promotion-width-moe-20260717-105340`.
- The records were present and complete. The exact FFN event order is
  `router`, routed-MoE inner rows, `routed_moe`, then the shared-expert stages.
  The initial mapper had incorrectly looked between `shared_down` and
  `routed_moe`, an empty reversed interval.
- The mapper now accepts records between the matching exact `router` and
  `routed_moe` controls, still bounded by `attention_tail_serial` and
  `ffn_batch`.
- Applying the corrected mapper offline to the real failed layer-0 stderr
  recovers all `26` exact batches and `510` output rows:
  - width 2: `1` control and `8` inner stage records;
  - width 3: `1` control and `12` inner stage records;
  - width 4: `4` controls and `64` inner stage records;
  - width 5: `20` controls and `400` inner stage records.
- Every recovered row uses the same `pair_swiglu` Metal path. The synthetic
  mapper fixture now reproduces the real routed-before-shared stage order.

User-run command:

```sh
python3 speed-bench/run_dspark_post_promotion_width_routed_moe_profile.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260717-092241/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260717-094819/summary.json \
  --width-layer-reference \
  speed-bench/local-runs/post-promotion-width-layer-20260717-100542/summary.json \
  --width-ffn-reference \
  speed-bench/local-runs/post-promotion-width-ffn-20260717-104106/summary.json \
  --confirm-ready
```

Result:

- Artifact:
  `speed-bench/local-runs/post-promotion-width-moe-20260717-110416`.
- Clean source commit: `b3288a6e31f683d2d426289ef9f0e9e57cbd00c9`.
- Every profiled output matched the frozen cumulative HumanEval artifact
  byte-for-byte and reproduced the expected width counts `1/1/4/20`.
- The inner stage sum reconciled to `0.937x..0.956x` of the enclosing
  routed-MoE control across widths 2 through 5.

Width-5 stage split across layers `0`, `21`, and `42`:

- Gate/up: `1.261 ms/row`, `49.5%`.
- Down: `1.113 ms/row`, `43.7%`.
- Activation-weight: `0.086 ms/row`, `3.4%`.
- Sum boundary: `0.085 ms/row`, `3.4%`.
- Gate/up is stable at `0.419`, `0.424`, and `0.419 ms/row` by sampled layer.
- Down is stable at `0.370`, `0.372`, and `0.372 ms/row`.

Width behavior:

- Width 2: `2.586 ms/row` inner, `2.760 ms/row` outer.
- Width 3: `2.499 ms/row` inner, `2.659 ms/row` outer.
- Width 4: `2.552 ms/row` inner, `2.668 ms/row` outer.
- Width 5: `2.546 ms/row` inner, `2.676 ms/row` outer.
- Each inner record is still one proposal row, so the flat cost confirms the
  current exact path does not amortize its matrix work as verifier width grows.

Decision:

- Do not optimize activation-weight or the nominal sum boundary; together they
  account for only `6.8%` of this synchronized routed-MoE attribution.
- Reuse the earlier Phase 0.58 proof: generic batched routed weighted-mid was
  bit-exact for widths 2 through 5, while only width-5 down/sum diverged.
- Build a default-off hybrid correctness candidate inside the existing Metal
  batch MoE path:
  - use current batched gate/up and weighted activation to produce exact F32
    routed mid rows;
  - encode the existing direct six-expert down/sum primitive once per output
    row using those batch mids;
  - keep ordinary prefill, sidecar execution, and non-Metal backends unchanged.
- Prove internal and generated-output exactness before any timed ablation. This
  candidate attacks the measured `49.5%` gate/up share while retaining the
  arithmetic boundary already known to be necessary at width 5.

## Phase 1.32: Exact routed-MoE hybrid ablation prepared

Implementation:

- Extended the internal Metal `ds4_gpu_routed_moe_batch_tensor` API with a
  default-false `exact_down_rows` request.
- Added the exact-runtime-only environment gate
  `DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID`.
- When the gate is absent, exact DSpark retains the established fully row-wise
  routed-MoE path. Ordinary prefill, layer-major prefill, DSpark sidecar FFN,
  and every non-exact caller pass `false` and are unchanged.
- When enabled for exact DSpark verification, the hybrid:
  - enters the existing batch routed-MoE encoder;
  - uses the current tiny-pair gate/up path for widths 2 through 4 and current
    batch MV gate/up path for width 5;
  - forces the weighted SwiGLU mid to remain F32;
  - creates one-token down arguments and encodes the existing
    `ds4_gpu_encode_mul_mv_id_sum6` Q2_K direct-down primitive once per proposal
    row using row-specific mid, selected-ID, and output offsets;
  - skips the generic separate expert-output sum after those direct writes.
- Unsupported table, streaming-address, MM, expert-count, or down-kernel
  combinations fail explicitly instead of silently dropping to a different
  arithmetic path.
- No Metal shader was added or changed. The candidate composes existing
  kernels and remains Metal-only.

Correctness:

- `make -j4 ds4 ds4_test` passed with no new warnings.
- The runtime GPU candidate correctness matrix passed with the hybrid enabled:
  reasoning, Italian, medium context, rolling window, and resumed two-turn chat
  all matched ordinary baseline byte-for-byte.
- The same matrix passed with the selected-layer exact FFN observer at layers
  `0`, `30`, and `42`. This covers the known width-5 boundary and requires every
  observed exact FFN batch to remain exact through routed mid/output, FFN
  combine, and HC post.
- All `141` DSpark model-free tests passed, including five new tests for mode
  isolation, command construction, paired summary, and direct-down structure.
- Normal builds passed for `ds4`, `ds4_test`, `ds4-server`, `ds4-eval`,
  `ds4-agent`, `ds4_cpu.o`, and `ds4-warm-prefill-bench`.
- `./ds4_test --dspark-validation --dspark-shape-binding` passed.
- The unrelated approximate fast-verifier soak passed long generation, rolling
  window, code completion, and Italian, then produced a Spanish output mismatch
  with `DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID` unset. Logs were retained at
  `/var/folders/vq/rkb95_8n02d4c91kltpxr1r80000gp/T/ds4-dspark-fast-soak.xlTLnH`.
  The candidate does not run in that fast-verifier path and its exact-runtime
  matrices passed; do not present the broader soak as fully green.
- `git diff --check` passed.

Ablation harness:

- Added `--exact-routed-moe-hybrid-ablation` to
  `speed-bench/run_dspark_comparison.py`.
- The reference arm is current default exact DSpark; the candidate arm adds
  only `DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID=1`.
- Both arms force Metal, threshold/runtime defaults remain identical, and all
  runtime stats, traces, diagnostics, and profilers are disabled.
- Warmups and measured pairs retain alternating order. Every candidate output
  must match the first frozen exact output byte-for-byte.
- The dry run passed and executed no model process.

User-run command:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-routed-moe-hybrid-ablation \
  --confirm-idle
```

Result:

- Artifact: `speed-bench/local-runs/20260719-162309`.
- Clean source commit: `7491c0761bbad8b13f1913ff59badf7b55a83445`.
- Default exact median: `24.74 t/s`.
- Routed-MoE hybrid median: `25.13 t/s`.
- Ratio of medians: `1.0158x`; median paired ratio: `1.0149x`, or
  `+1.6%`.
- Every measured output used the same SHA-256. No runtime stats, diagnostics,
  trace, or profiler was enabled.
- All three paired ratios favored the hybrid:
  - pair 1: `1.0113x`;
  - pair 2: `1.0227x`;
  - pair 3: `1.0149x`.
- The final thermal snapshot reported no thermal or performance warning.

Decision:

- Retain the hybrid as default-off. The focused result is consistent and worth
  carrying forward, but a `1.6%` gain is too small for promotion from one
  prompt and six measured processes.
- Confirm on the frozen 32-task threshold-`0.75` HumanEval workload, pairing
  current default exact against the hybrid within every task and validating
  both against the frozen exact output.
- Predeclare a modest but nonzero promotion gate appropriate to the focused
  effect size:
  - geometric hybrid/default ratio at least `1.005x`;
  - at least `20/32` tasks faster;
  - no task below `0.95x`;
  - low-acceptance subgroup geometric mean at least `1.00x`.
- If the gate passes, promote the hybrid and then rerun the cumulative
  baseline-versus-DSpark HumanEval assessment. If it fails, keep the code
  available for attribution but do not stack its microbenchmark result into
  the current `0.8826x` end-to-end figure.

## Phase 1.33: HumanEval routed-MoE hybrid gate prepared

New harness:

- Added `speed-bench/run_dspark_humaneval_routed_moe_hybrid.py`.
- Added model-free tests in
  `tests/test_dspark_humaneval_routed_moe_hybrid.py`.
- The harness reuses the frozen deterministic 32-task selection, prompts,
  acceptance labels, and exact outputs from the threshold-`0.75` HumanEval
  study.
- Each task receives one uninstrumented current-default/hybrid pair. Odd tasks
  run default first; even tasks run hybrid first.
- Two global warmup pairs use opposite mode orders and are excluded from every
  reported statistic. The complete invocation count is `68`: four warmup and
  `64` measured processes.
- Both arms use Metal, exact DSpark, multi-commit, confidence threshold `0.75`,
  non-thinking mode, greedy decoding, context `16384`, and `128` output tokens.
- The hybrid arm adds only
  `DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID=1`.
- Every default and hybrid output must match the frozen exact task output
  byte-for-byte. Stats, acceptance audit, traces, diagnostics, profilers, and
  fast verification are forbidden.

Predeclared promotion gate:

- Geometric hybrid/default paired ratio at least `1.005x`.
- At least `20/32` tasks faster with the hybrid.
- No task paired ratio below `0.95x`.
- For tasks whose prior verify rate is at most `0.65`, geometric
  hybrid/default ratio at least `1.00x`.
- These thresholds were frozen after the focused `+1.6%` result but before any
  HumanEval hybrid process was executed.

Validation:

- Python compilation passed.
- Six targeted tests cover constants, balanced order, environment isolation,
  Metal command construction, passing aggregation, and subthreshold failure.
- All `147` DSpark model-free tests passed.
- The real dry run accepted the frozen reference and corpus, printed exactly
  `32` measured task schedules, and reported `68` total processes.
- Inspection of the dry-run commands confirms the candidate switch appears
  only in the hybrid arm and no instrumentation is enabled.
- No prompt was materialized and no model process was run.
- `git diff --check` passed.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_routed_moe_hybrid.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-idle
```

Result:

- Artifact:
  `speed-bench/local-runs/humaneval-routed-moe-hybrid-32-20260719-163559`.
- Clean source commit: `929593e4e49b37bb38c4daa657b84e3025e84942`.
- Every default and hybrid output matched the frozen exact artifact
  byte-for-byte.
- Default median: `20.60 t/s`; hybrid median: `20.96 t/s`; ratio of medians
  `1.0172x`.
- Median paired ratio: `1.0158x`; geometric paired ratio: `1.0164x`.
- The hybrid won all `32/32` tasks. The interquartile range was
  `1.0132x..1.0200x`; the complete range was `1.0073x..1.0305x`.
- The ten low-acceptance tasks had a `1.0119x` geometric hybrid/default ratio.
- Acceptance and movement had a descriptive Pearson correlation of `0.679`:
  high-acceptance tasks benefit somewhat more, but every measured acceptance
  regime improved.
- The final machine snapshot reported no thermal or performance warning,
  though WindowServer and Stats were active. Paired within-task direction is
  therefore more authoritative than cross-session absolute medians.

Promotion decision:

- PASS: geometric ratio `1.0164x >= 1.005x`.
- PASS: `32/32 >= 20/32` faster tasks.
- PASS: minimum task ratio `1.0073x >= 0.95x`.
- PASS: low-acceptance geometric ratio `1.0119x >= 1.00x`.
- Promote the hybrid for exact Metal verification. Preserve the old fully
  row-wise routed-MoE path behind an explicit `=0` environment opt-out for
  regression and attribution.
- Do not multiply `1.0164x` into the prior cross-session `0.8826x` cumulative
  result. Measure a fresh ordinary-baseline/current-DSpark paired workload
  after promotion.

## Phase 1.34: exact routed-MoE hybrid promoted

Implementation:

- Exact Metal verification now selects the gate/up batched plus exact one-row
  down/sum hybrid when `DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID` is absent.
- Explicit `DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID=0` or `off` preserves the
  previous fully row-wise routed-MoE implementation. Other nonempty values
  continue to select the hybrid.
- The focused comparison and frozen HumanEval confirmation harnesses now
  compare `legacy_routed_moe_rows` (`=0`) against promoted `default_exact`
  (no candidate flag). This prevents future reruns from silently comparing
  two identical promoted-default arms.
- The cumulative HumanEval runtime arm uses only promoted defaults and pins
  threshold `0.75`; it does not set the routed-MoE environment variable.

Validation:

- `make -j4 ds4 ds4_test` passed without a new compiler warning.
- The default runtime correctness matrix passed reasoning, Italian,
  medium-context, rolling-window, and resumed two-turn chat byte-for-byte.
- The same five-case matrix passed with the explicit legacy `=0` opt-out.
- `./ds4_test --dspark-validation --dspark-shape-binding` passed.
- All `148` DSpark model-free tests and Python compilation passed.
- The focused promotion-confirmation dry run printed a legacy `=0` arm and a
  flag-free default arm. The HumanEval promotion-confirmation and cumulative
  harnesses each accepted their frozen references in dry-run mode, printed
  `32` measured schedules and `68` total uninstrumented processes, and
  materialized no prompts.
- Dry-run command inspection confirms that only the legacy comparison arm
  contains `DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID=0`; the promoted and cumulative
  runtime arms contain no routed-MoE override.
- `git diff --check` passed. No timed throughput benchmark was run by Codex.

Next user-run command:

```sh
python3 speed-bench/run_dspark_humaneval_cumulative_throughput.py \
  --historical-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-idle
```

Interpret the fresh paired DSpark/baseline geometric mean directly. Compare
task-level movement with the prior cumulative artifact, but do not multiply
the `1.0164x` routed-MoE confirmation into the old `0.8826x` result.

Result:

- Artifact:
  `speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-171040`.
- Clean source commit: `bc6557eb66e283bbd3be702910e24f509e1a40b9`.
- Every current DSpark output matched ordinary baseline and the frozen
  historical output byte-for-byte.
- Baseline median: `24.88 t/s`; current DSpark median: `21.75 t/s`.
- Ratio of medians: `0.8744x`; median paired ratio: `0.8748x`; geometric
  paired ratio: `0.8782x`.
- The paired-ratio interquartile range was `0.8433x..0.9178x`; the full range
  was `0.7810x..0.9902x`. No task reached baseline throughput (`0/32`).
- The geometric gap to parity is `12.2%`, so the end-to-end outcome remains
  below the predeclared `0.95x` near-parity threshold.
- Against the older threshold-`0.75` historical artifact (`0.8634x`), task
  movement was `1.0171x` geometrically with `23/32` improvements, which failed
  the intentionally demanding accumulated-movement gate.

Immediate-prior comparison:

- The directly preceding cumulative artifact is
  `speed-bench/local-runs/humaneval-cumulative-throughput-32-20260717-092241`,
  with `0.8826x` geometric paired throughput.
- Current/prior task movement is `0.9950x` geometrically and `1.0033x` by
  median, with `22/32` tasks improving.
- Two tasks dominate the geometric regression: `humaneval_000` moved
  `0.8989x` and `humaneval_110` moved `0.8982x`. Excluding only those two gives
  `1.0018x` geometric movement across the other 30 tasks. This is descriptive
  sensitivity analysis, not a revised benchmark result.
- The final snapshot had no thermal or performance warning, but WindowServer
  was using `44.5%` CPU versus `3.1%` in the prior cumulative artifact. The
  user cannot guarantee an idle machine, so paired within-task values remain
  authoritative while cross-session movement must be treated cautiously.

Decision:

- Keep the routed-MoE hybrid promoted. Its frozen same-session A/B gate won
  `32/32` tasks with `1.0164x` geometric movement and is the correct evidence
  for that local default decision; this cumulative run has no legacy arm and
  cannot isolate the promotion.
- Do not claim that the promotion improved cumulative end-to-end throughput.
  The fresh measured DSpark/baseline state is `0.8782x`, still `12.2%` from
  parity.
- This is a safe development checkpoint: implementation and fallback are
  committed, correctness is green, the end-to-end state is freshly measured,
  and no candidate is left half-implemented. Review new community work before
  selecting the next exact-verifier optimization. If no transferable win has
  appeared, return to the measured target-verifier bottleneck rather than
  further tuning the already-small sidecar or routed-MoE surface.

## Phase 1.35: upstream baseline gate and external implementation survey

Safe-state context:

- The current branch is `codex/dspark-observability-0` at source commit
  `82f224b` before this phase. It descends directly from the pinned pre-DSpark
  `origin/main` commit
  `80ebbc396aee40eedc1d829222f3362d10fa4c6c` and is 103 commits ahead.
- Fetched `origin/main` and created a clean detached sibling worktree at
  `/Users/deathcodevision/dev/ds4-master-baseline`, pinned to that exact SHA.
- Built `/Users/deathcodevision/dev/ds4-master-baseline/ds4` with
  `make -j4 ds4`. No benchmark process was run by Codex.

Ordinary-decode comparison prepared:

- Added `speed-bench/run_ds4_master_baseline_comparison.py` and model-free
  tests in `tests/test_ds4_master_baseline_comparison.py`.
- The runner compares the pinned upstream binary with the current branch on
  the deterministic 32-task HumanEval workload from
  `humaneval-cumulative-throughput-32-20260719-171040`.
- Both arms are ordinary Metal decode with no `--dspark`; every `DS4_*`
  variable is cleared. Both outputs must match the frozen clean artifact
  byte-for-byte.
- Measured order alternates upstream/current by task. Two balanced warmup
  pairs are excluded, for 68 total processes and 64 measured processes.
- The runner records both source SHAs and binary hashes, rejects dirty tracked
  trees, and refuses an upstream worktree not at the pinned SHA.
- The predeclared meaningful-progress gate is geometric current/upstream at
  least `1.01x`, at least `24/32` current wins, and no task below `0.95x`.
- A real dry run accepted the frozen reference, printed 32 schedules and 68
  processes, and confirmed the two commands use distinct binaries with no
  DSpark flag. It materialized no prompt and ran no model.
- Python compilation and the seven targeted harness tests passed. All 148
  existing DSpark model-free tests also passed, for 155 model-free tests across
  the two suites.
- `make -j4 ds4` confirmed the current binary is up to date, and
  `git diff --check` passed.

First-run correction:

- Artifact `ds4-master-baseline-32-20260719-183948` stopped during the first
  upstream warmup, before any measured row. The upstream process exited with
  signal 6 while creating
  `kernel_dsv4_indexed_mixed_attention_heads8`.
- This was a harness isolation bug, not an upstream model failure. ds4 loads
  and compiles relative `metal/*.metal` files at runtime. The harness launched
  both binaries with the current branch as their working directory, combining
  the upstream host binary with newer current-branch Metal source. The newer
  kernel requires function-constant specialization that the old host did not
  request.
- Each mode now runs from its own source worktree: upstream from
  `/Users/deathcodevision/dev/ds4-master-baseline`, current from
  `/Users/deathcodevision/dev/ds4`. Commands and metadata record the working
  directory, and a model-free regression test freezes this requirement.
- The failed artifact contains no usable timing result and must not be
  summarized or compared.

User-run command:

```sh
python3 speed-bench/run_ds4_master_baseline_comparison.py \
  --output-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-171040/summary.json \
  --confirm-idle
```

Result:

- Artifact:
  `speed-bench/local-runs/ds4-master-baseline-32-20260719-190740`.
- Clean current source commit:
  `8eb7d3aec57183afe6bb4d2cd61c8c1271c676cd`; pinned upstream source commit:
  `80ebbc396aee40eedc1d829222f3362d10fa4c6c`.
- Both ordinary, non-DSpark binaries reproduced all 32 frozen outputs
  byte-for-byte.
- Upstream median: `22.80 t/s`; current median: `24.68 t/s`; ratio of
  medians: `1.0822x`.
- Median paired current/upstream ratio: `1.0848x`; geometric paired ratio:
  `1.0822x`, or `+8.2%`.
- The current branch won all `32/32` tasks. The interquartile range was
  `1.0806x..1.0881x`; the complete range was `1.0282x..1.0979x`.
- Order did not explain the gain. The 16 upstream-first tasks had a `1.0818x`
  geometric ratio; the 16 current-first tasks had a `1.0827x` geometric
  ratio.
- The initial and final machine snapshots reported no thermal or performance
  warning, but WindowServer, Stats, and a browser renderer were active. The
  tight paired distribution and balanced order make the relative result
  persuasive; absolute medians remain specific to this run.

Interpretation:

- PASS: the branch contains a real, broad ordinary Metal decode improvement,
  not only DSpark-specific work. The predeclared `1.01x`, `24/32`, and `0.95x`
  gates all passed comfortably.
- Keep two performance questions separate. Current DSpark is still `0.8782x`
  versus the current branch's own faster baseline on the frozen cumulative
  study; that is the correct incremental cost of enabling DSpark. The complete
  current branch is `1.0822x` faster than pinned upstream without DSpark; that
  is shared engine progress and must not be called a DSpark speedup.
- Multiplying the two independently paired ratios gives a descriptive
  cross-session estimate of `0.9504x` for current DSpark versus old upstream
  baseline. This suggests the complete branch may be roughly 5% below the
  pinned upstream product on this workload, rather than 12.2%, but it is not a
  direct measurement and must not be reported as one.
- A direct upstream-baseline/current-DSpark pair would answer the product-level
  comparison. It is not required before optimizing the algorithm, because
  parity with the same current baseline remains the stricter and more honest
  DSpark target.

External survey snapshot, 2026-07-19:

- MTPLX was inspected at `54a1d9a`. Its most relevant work is
  verify-shape quantized matmul for narrow row counts: one kernel streams a
  quantized weight tile while accumulating all verify rows, with split-K and
  multi-simdgroup geometries chosen by shape. Its repository reports material
  verify gains on Apple Silicon. Its Q4 arithmetic can differ from stock at
  tail ULPs, so morphology is portable but its numerical implementation
  cannot be copied into ds4's byte-exact path without a correctness gate.
- oMLX was inspected at `ed8337e` (`v0.5.2.dev1`). It now explicitly ports
  MTPLX's verify-QMM kernels and arms them only around MTP target verification.
  It also uses a minimum output-width gate because dispatching hundreds of
  small custom projections costs more on the host than it saves on the GPU.
  This independently validates both shape specialization and selective
  routing as design requirements.
- mlx-dspark was inspected at `9e39ea2` (`v0.5.0`). Its published diagnosis is
  the same: narrow multi-row quantized target verification is the limiting
  Apple-Silicon cost. It also offers lossless prompt-copy drafting, which can
  be useful for editing workloads but is a separate drafter/router feature,
  not a general improvement to the current DSpark verifier.
- MLX PR 3120 added split-K for small-M quantized matmul. The published M3 Max
  examples improve M=12 and M=16 by roughly 25-30%. Our dominant widths are
  2-5, so the occupancy principle is relevant but its exact geometry is not a
  drop-in answer.
- llama.cpp PR 25173 is an open DSpark implementation layered on DFlash. Its
  reported wins are CUDA/Qwen results and its principal changes concern model
  support, Markov bias, and graph reuse. It does not presently offer a Metal
  verifier kernel to port. Its confidence data also says pruning helps mainly
  at serving concurrency, consistent with keeping our single-session
  threshold policy evidence separate.
- vLLM's current MTP work is useful for scheduler and serving architecture,
  but its main optimization mechanisms are CUDA graphs, batched serving, and
  GPU-specific kernels. They are not direct single-session Metal ports.

Ranked portable candidates:

1. Add a width-specialized exact Q2_K routed-down kernel for verifier widths
   2-5. The promoted hybrid still loops proposal rows and invokes
   `ds4_gpu_encode_mul_mv_id_sum6` once per row. A kernel that reuses each
   selected Q2_K weight stream across proposal rows attacks the current target
   verifier directly while preserving the existing gate/up hybrid.
2. Revisit attention batching only as a genuinely fused KV-streaming kernel.
   MTPLX's packed verify attention reuses KV across queries. Our earlier suffix
   batching regressed because capture, gather, and projection overhead exceeded
   the gain; the promoted fused-gather path already removed much of that
   overhead. A new candidate must stream the cache once and preserve exact
   operation order, not merely wrap the old serial calls in a batch API.
3. Consider prompt-copy/n-gram drafting after target parity. It can produce
   large editing/code gains, but it is workload-dependent and does not reduce
   the cost of a DSpark round.

Deferred as low transfer value:

- Python graph compilation/dispatch caching: ds4 already owns a direct C/Metal
  command path, so MTPLX/oMLX host-level graph wins do not map cleanly.
- CUDA graphs and continuous batching: valuable for server aggregate
  throughput, outside the current one-session Metal objective.
- NAX-only kernels: hardware-specific and not a general Apple-Silicon default.
- Further sidecar tuning: post-promotion target verification remains the
  dominant gap, while the sidecar is already comparatively small.

Decision boundary:

- Run the pinned upstream/current ordinary-decode comparison first. It tells
  us whether baseline movement is real and prevents attributing shared Metal
  improvements solely to DSpark.
- After that safe checkpoint, the next bounded implementation should be an
  opt-in, exact Q2_K verifier-width kernel with byte-exact correctness and
  focused attribution before any timed promotion gate.

## Phase 1.36: exact Q2_K routed-down single dispatch

Mapping correction:

- The external-survey candidate was initially described as a new kernel that
  would reuse one selected Q2_K expert-weight stream across proposal rows.
  That does not match routed-MoE semantics: different proposal rows can select
  different experts, so MTPLX's dense-weight reuse is not directly available.
- The existing Metal shader `kernel_mul_mv_id_q2_K_sum6_f32` already has a
  token dimension. Its host encoder dispatches `args->nei1` token rows in one
  grid, preserving each row's selected expert ids and sum-six arithmetic.
- The promoted exact routed-MoE hybrid nevertheless called that encoder once
  per proposal row. The bounded candidate therefore removes repeated host
  encoding/dispatch work by passing the existing multi-token `down_args` once;
  it does not change shader arithmetic or claim cross-row weight reuse.

Implementation:

- Added the default-off `DS4_DSPARK_EXACT_Q2_DOWN_BATCH=1` route in
  `ds4_metal.m`. It applies only to the promoted exact-down path, Q2_K weights,
  six selected experts, and verifier widths 2-5. Width 1 and every unsupported
  case retain the existing one-row loop unchanged.
- Added `DS4_DSPARK_EXACT_Q2_DOWN_BATCH_TRACE=1` for correctness diagnostics.
  The trace records layer, actual verifier width, and encode result; it is not
  enabled in throughput runs.
- Extended `tests/dspark_gpu_candidates_correctness.sh` with an exact-runtime
  candidate gate that requires successful route records and rejects accidental
  use in control modes.
- Added `--exact-q2-down-batch-ablation` to
  `speed-bench/run_dspark_comparison.py`. It pairs the current exact default
  against the opt-in candidate on Metal, clears inherited experiment state,
  and disables runtime stats and diagnostics in both timed modes.
- Added model-free benchmark/source-contract coverage in
  `tests/test_dspark_exact_q2_down_batch.py`.

Validation before timing:

- `make` rebuilt all binaries successfully.
- All 154 `test_dspark_*.py` model-free tests passed; Python compilation,
  shell syntax, and `git diff --check` also passed.
- Three runtime correctness matrices passed every reasoning, Italian,
  medium-context, rolling-window, and resumed-chat case byte-for-byte against
  ordinary baseline output.
- A forced `0` confidence threshold exercised width 5. The promoted `0.75`
  threshold exercised widths 3 and 4. A `0.85` threshold exercised width 2.
  The route therefore has observed byte-exact coverage for every supported
  width 2-5, with no candidate failure or verifier fallback.
- A real benchmark dry run printed only the expected environment difference:
  `DS4_DSPARK_EXACT_Q2_DOWN_BATCH=1` in the candidate. Both timed arms omit
  DSpark stats and diagnostics.

User-run timing command:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-q2-down-batch-ablation \
  --confirm-idle
```

Interpretation boundary:

- This is dispatch consolidation, not the originally hypothesized
  weight-reuse kernel. A small gain is plausible because it removes up to four
  encoder/dispatch calls per layer and verifier evaluation, but it does not
  reduce Q2_K weight traffic.
- Do not promote it from one short run. A positive focused ablation should be
  followed by the frozen 32-task HumanEval threshold-0.75 confirmation before
  changing the default.

Focused result:

- User-run artifact:
  `speed-bench/local-runs/20260719-194951` at clean commit `26e9130`.
- Default exact median: `24.90 t/s`; single-dispatch candidate median:
  `24.81 t/s`; ratio of medians: `0.9964x`, or `-0.4%`.
- Paired ratios were `1.0020x`, `0.9815x`, and `1.0309x`; their median was
  `1.0020x`. All outputs had the same SHA-256.
- The initial process snapshot was busy, including an Arc renderer at 96.5%
  CPU, `duetexpertd` at 81.5%, and WindowServer at 47.8%. There was no thermal
  or performance warning. The interference explains the wide pair spread but
  does not turn the near-zero central result into evidence of a useful gain.

Decision:

- Do not promote and do not spend a 32-task HumanEval confirmation on this
  candidate. The ratio-of-medians and paired median straddle parity, and the
  best central estimate is far below a meaningful optimization threshold.
- Keep the route default-off as a documented exact experiment. Its result is
  useful: consolidating the existing Q2_K down dispatch removes host encoding
  calls but does not reduce expert-weight traffic, and host dispatch was not a
  material part of the remaining verifier gap.
- The next verifier candidate must reduce GPU work or weight traffic rather
  than only encoder calls. Shared-weight narrow-row projections are a better
  match for MTPLX/oMLX-style verify QMM than routed expert rows, whose selected
  matrices can differ by proposal position.

## Phase 1.37: exact shared-expert Q8 proposal rows

Target mapping:

- The post-promotion exact FFN still executed all three shared-expert Q8_0
  projections one proposal row at a time: fused gate/up/SwiGLU and shared
  down. Unlike routed experts, every proposal row uses the same shared gate,
  up, and down matrices, so this is a direct match for MTPLX/oMLX-style
  narrow-row weight reuse.
- The prior width-5 FFN profile attributed `0.274 ms/row` to shared gate/up
  and `0.245 ms/row` to shared down across sampled layers. Together they were
  about one quarter of the sampled exact FFN sub-stage time, making this a
  bounded but material surface.
- ds4's promoted exact Q8 proposal-row kernel already proved that one weight
  traversal can preserve decode-identical block traversal, accumulation, and
  SIMD/threadgroup reduction for widths 2-5. The new route extends that proven
  arithmetic to the shared expert instead of changing quantization math.

Implementation:

- Added width-specialized Metal entry points
  `kernel_dsv4_shared_gate_up_swiglu_q8_0_exact_rows_{2,3,4,5}`. Each loads a
  gate and up Q8 block once, applies it to every proposal activation, preserves
  the ordinary one-row reduction order independently for each row, and applies
  the same clamp/SwiGLU operations as the existing fused one-row kernel.
- Shared down reuses the already-promoted
  `ds4_gpu_matmul_q8_0_exact_rows_tensor`; no second shader was needed.
- Added the default-off `DS4_DSPARK_EXACT_SHARED_Q8_ROWS=1` route. It is
  limited to exact FFN verification, Q8_0 shared weights, and widths 2-5. The
  established one-row fused gate/up and one-row down loops remain unchanged as
  the default and unsupported-shape fallback.
- Added `DS4_DSPARK_EXACT_SHARED_Q8_ROWS_TRACE=1`. Correctness records identify
  layer, actual width, `gate_up` or `down`, and pass/fail; timed modes do not
  enable it.
- Extended `tests/dspark_gpu_candidates_correctness.sh` to require matching,
  successful gate/up and down records and reject accidental control-mode use.
- Added `--exact-shared-q8-rows-ablation` to the paired uninstrumented Metal
  runner and model-free coverage in
  `tests/test_dspark_exact_shared_q8_rows.py`.

Validation before timing:

- `make` completed successfully. The runtime Metal compiler accepted every new
  width specialization.
- All 161 `test_dspark_*.py` model-free tests passed; Python compilation, shell
  syntax, and `git diff --check` passed.
- A forced-full-block correctness matrix and a threshold-0.75 matrix each
  passed reasoning, Italian, medium-context, rolling-window, and resumed-chat
  cases byte-for-byte against ordinary baseline output.
- Trace coverage observed widths 2, 3, and 5 in the forced-block matrix and
  widths 3 and 4 under threshold 0.75. Both shared stages passed at every
  observed width, giving complete width-2-through-5 coverage with no verifier
  fallback.
- A real benchmark dry run showed only
  `DS4_DSPARK_EXACT_SHARED_Q8_ROWS=1` in the candidate. Runtime stats,
  diagnostics, trace, and fast verification are absent from both timed arms.

User-run focused command:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-shared-q8-rows-ablation \
  --confirm-idle
```

Decision boundary:

- This candidate genuinely reduces shared Q8 weight traffic and dispatches;
  it is not another host-only consolidation. Even so, keep it default-off
  until measured.
- Require a clearly positive focused paired result before spending time on the
  frozen 32-task HumanEval confirmation. Promotion still requires that broader
  threshold-0.75 gate and byte-exact output on every task.

Focused result:

- User-run artifact: `speed-bench/local-runs/20260719-201407` at clean commit
  `63b5834`.
- Default exact median: `24.82 t/s`; exact shared-Q8 median: `25.37 t/s`;
  ratio of medians: `1.0222x`, or `+2.2%`.
- All three paired ratios were wins: `1.0193x`, `1.0226x`, and `1.0234x`;
  paired median `1.0226x`. Every output SHA-256 matched.
- The initial process snapshot was busy, including an Arc renderer at 95.9%
  CPU and WindowServer at 45.6%, but there was no thermal/performance warning.
  The unusually tight all-positive pair range makes the direction persuasive
  despite desktop interference.

Broad confirmation prepared:

- Added `speed-bench/run_dspark_humaneval_shared_q8_rows.py` and model-free
  tests in `tests/test_dspark_humaneval_shared_q8_rows.py`.
- It compares default exact against the opt-in shared-Q8 route on the frozen
  32-task HumanEval selection, pins confidence threshold `0.75`, alternates
  order exactly, and excludes two balanced global warmup pairs.
- Every measured output must match the frozen exact artifact from
  `humaneval-threshold075-throughput-32-20260716-155112` byte-for-byte.
- Both arms are uninstrumented exact Metal verification. The only candidate
  difference is `DS4_DSPARK_EXACT_SHARED_Q8_ROWS=1`; stats, trace,
  diagnostics, profiler, and fast verification are forbidden.
- The predeclared promotion gate requires geometric candidate/default at least
  `1.005x`, at least `20/32` task wins, no task below `0.95x`, and geometric
  ratio at least `1.00x` across tasks with acceptance at or below `0.65`.
- Model-free harness tests passed. A real dry run accepted the frozen
  provenance and printed 68 processes: four excluded warmups and 64 measured.

User-run broad command:

```sh
python3 speed-bench/run_dspark_humaneval_shared_q8_rows.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-idle
```

Broad result and promotion:

- User-run artifact:
  `speed-bench/local-runs/humaneval-shared-q8-rows-32-20260719-202319`.
- Default median: `21.54 t/s`; candidate median: `21.89 t/s`; ratio of
  medians: `1.0165x`.
- Median paired ratio: `1.0177x`; geometric paired ratio: `1.0196x`, or about
  `+2.0%` across tasks.
- The candidate won all `32/32` tasks. The interquartile range was
  `1.0134x..1.0218x`; the complete range was `1.0043x..1.0503x`. Even the
  worst task remained positive.
- The ten low-acceptance tasks had a `1.0161x` geometric ratio. The gain is
  therefore broad rather than dependent on unusually long accepted blocks.
- Every default and candidate output matched the frozen exact artifact
  byte-for-byte. All predeclared gates passed comfortably.

Promotion change:

- Exact shared-expert Q8 rows are now enabled by default in the exact verifier.
  `DS4_DSPARK_EXACT_SHARED_Q8_ROWS=0` or `off` selects the retained legacy
  row-wise path. Width 1, non-Q8 weights, and unsupported shapes continue to
  use the existing fallback automatically.
- The focused and HumanEval runners now compare explicit legacy opt-out against
  the promoted unset-variable default. Their labels, reports, metadata, and
  model-free contracts were updated so a future run cannot compare the
  promoted path with itself.
- The correctness harness now distinguishes `default`, `0`, and `1` controls.
  After promotion, both the ordinary default matrix and an explicit legacy
  opt-out matrix passed reasoning, Italian, medium-context, rolling-window,
  and resumed-chat cases byte-for-byte against baseline.
- `make`, benchmark dry runs, Python/shell syntax checks, and the focused
  model-free tests passed after the default flip.

Decision:

- PROMOTE. This is a measured shared engine improvement within exact DSpark,
  with unanimous per-task direction and no acceptance-dependent regression.
- The branch is again at a safe checkpoint. Future optimization work should
  use the promoted default and retain `=0` only for attribution or regression
  diagnosis.

## Phase 1.41: refreshed width-stratified serial tail

User-run artifact:

- `speed-bench/local-runs/post-promotion-width-tail-20260719-212250`
- Every profiled output matched the frozen cumulative HumanEval artifact
  byte-for-byte.
- At width 5, the median serial tail was `2.434 ms/row` across twenty verifier
  evaluations.
- Component medians were compressor/indexer `0.403 ms/row` (`20.1%`),
  attention `0.362` (`18.1%`), projection B plus HC `0.343` (`17.1%`),
  projection A `0.331` (`16.5%`), KV/cache update `0.295` (`14.7%`), and
  inverse RoPE `0.270` (`13.5%`).
- Widths 2 and 3 each have only one observation. Use width 5 as the stable
  optimization guide and do not infer useful width scaling from those sparse
  anchors.

Interpretation:

- No single tail component dominates. A broad suffix rewrite is not justified
  by this profile and previously regressed because deferred scheduling made the
  attention core more expensive.
- Attention already benefits from promoted dense-mixed fused gather.
  Projection A/B plus HC already use promoted NR4; NR8 was measured at `-3.1%`
  and remains rejected.
- Compressor/indexer is the largest component whose current execution shape
  had not been refreshed, making a narrowly scoped projection candidate the
  next useful experiment.

## Phase 1.42: exact compressor projection prebatch

Implementation:

- `DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH=1` enables the default-off candidate.
- The paired F16 Metal projection primitive now supports exact verifier widths
  1 through 5 with independent activation and output rows in the dispatch Y
  dimension.
- Exact attention preparation computes paired main compressor KV/score
  projections across all proposal rows before any row mutates compressor state.
- Ratio-4 layers also precompute their paired indexer-compressor projections
  into dedicated transient batch buffers.
- Serial attention consumes one non-owning row view at a time. Compressor state
  updates, cache writes, sparse-index construction, and attention calls retain
  their original row order.
- Unsupported shapes or routes fall back through the existing exact verifier.
  The ordinary default remains unchanged when the environment variable is
  absent.

Why this differs from rejected compressor-pair NR4:

- `DS4_METAL_COMPRESSOR_PAIR_NR4` changed output-row packing inside each
  one-row projection and measured `-0.4%`.
- This candidate removes repeated one-row projection dispatch and weight work
  across the verifier block while preserving each recurrent operation. It is a
  scheduling experiment, not another NR tuning experiment.

Validation before timing:

- `make` completed successfully after the runtime changes.
- All 175 `test_dspark_*.py` model-free tests passed; Python compilation, shell
  syntax, and `git diff --check` passed.
- The exact runtime correctness matrix passed reasoning, Italian,
  medium-context, rolling-window, and resumed-chat cases byte-for-byte.
- Trace coverage observed ratio-128 and ratio-4 layers and verifier widths 2,
  3, 4, and 5, all with `result=ok` and no fallback.
- The focused runner dry run shows no runtime stats, trace, diagnostics, or fast
  verifier in either timed arm. The only candidate difference is
  `DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH=1`.

User-run focused command:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-compressor-pre-batch-ablation \
  --confirm-idle
```

Decision boundary:

- Require a clearly positive focused paired result before building a 32-task
  HumanEval confirmation runner. A flat or negative result retires the
  candidate without changing the promoted default runtime.
- If focused timing is positive, broad confirmation must remain
  uninstrumented, threshold `0.75`, byte-exact, and acceptance-stratified.

Focused result:

- User-run artifact: `speed-bench/local-runs/20260719-214131` at clean commit
  `be7f101`.
- Default exact median: `25.56 t/s`; compressor projection prebatch median:
  `26.10 t/s`; ratio of medians `1.0211x` (`+2.1%`).
- Paired ratios were `1.0309x`, `1.0120x`, and `1.0212x`; all three pairs
  improved and the median paired ratio was `1.0212x`.
- Every measured arm produced the same stdout SHA-256. Runtime stats and
  instrumentation columns were empty as required.
- PROCEED to broad confirmation. Do not promote from this three-pair focused
  result alone.

## Phase 1.43: compressor-prebatch HumanEval gate prepared

Harness:

- Added `speed-bench/run_dspark_humaneval_compressor_pre_batch.py` and
  model-free tests in
  `tests/test_dspark_humaneval_compressor_pre_batch.py`.
- It compares the current promoted exact default against
  `DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH=1` on the frozen 32-task HumanEval
  selection at confidence threshold `0.75`.
- Measured order alternates exactly across tasks. Two balanced global warmup
  pairs are excluded, for 68 total model processes and 64 measured processes.
- Each output must match the byte-exact frozen threshold-0.75 reference from
  `humaneval-threshold075-throughput-32-20260716-155112`.
- Both arms forbid runtime stats, acceptance tracing, candidate tracing,
  profilers, diagnostics, and fast verification.

Predeclared promotion gate:

- Geometric candidate/default ratio at least `1.005x`.
- Candidate faster on at least `20/32` tasks.
- No task below `0.95x` candidate/default.
- Geometric candidate/default ratio at least `1.00x` across tasks whose
  acceptance verify rate is at most `0.65`.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_compressor_pre_batch.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-idle
```

Decision:

- Promote compressor projection prebatching only if this broad gate passes and
  all outputs remain byte-exact. Otherwise retain the current default and
  retire or revise the candidate.

Broad result:

- User-run artifact:
  `speed-bench/local-runs/humaneval-compressor-prebatch-32-20260719-215706`
  at clean commit `ce2c1e4`.
- Default median: `22.18 t/s`; prebatch median: `22.39 t/s`; ratio of medians
  `1.0095x`.
- Median paired ratio: `1.0127x`; geometric paired ratio: `1.0096x`.
- The candidate won `31/32` tasks. The interquartile range was
  `1.0093x..1.0166x` and the full range was `0.9138x..1.0222x`.
- The ten low-acceptance tasks gained `1.0094x` geometrically, and acceptance
  had little descriptive correlation with gain (`0.171`).
- Every output matched the frozen exact artifact byte-for-byte. No stats,
  traces, diagnostics, profilers, or fast verification were active.
- The formal gate failed only because `humaneval_095` measured `0.9138x`,
  below the `0.95x` per-task floor. All remaining tasks were positive.

Decision revision:

- Do not promote from the failed gate. Because the single failure is an
  isolated `-8.6%` observation against 31 consistent wins, adjudicate that
  task with repeated balanced pairs before deciding whether it is noise.

## Phase 1.44: compressor-prebatch outlier adjudication prepared

Harness:

- Added `speed-bench/run_dspark_humaneval_compressor_pre_batch_outlier.py` and
  model-free coverage in
  `tests/test_dspark_humaneval_compressor_pre_batch_outlier.py`.
- The loader accepts only the clean failed Phase 1.43 artifact, verifies that
  every other broad-gate condition passed, reconstructs every task ratio from
  the raw CSV, and requires `humaneval_095` to be the sole sub-floor task.
- It runs six measured pairs with exactly balanced order plus two excluded
  balanced warmup pairs: 16 uninstrumented model processes total.
- Both modes must match the same frozen threshold-0.75 exact output. The only
  candidate difference is `DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH=1`.

Predeclared adjudication gate:

- Median paired candidate/default ratio at least `1.005x`.
- Geometric paired ratio at least `1.005x`.
- Candidate wins at least `4/6` pairs.
- At least `5/6` pairs are at or above the original `0.95x` task floor.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_compressor_pre_batch_outlier.py \
  --confirmation-reference \
  speed-bench/local-runs/humaneval-compressor-prebatch-32-20260719-215706/summary.json \
  --confirm-idle
```

Decision:

- PASS rescues the otherwise-passing broad result and permits promotion.
- FAIL confirms unresolved task-level risk and retires the candidate.

Adjudication result:

- User-run artifact:
  `speed-bench/local-runs/humaneval-compressor-prebatch-outlier-20260719-222352`
  at clean commit `f5ca0ce`.
- Default median: `22.60 t/s`; prebatch median: `22.95 t/s`; ratio of medians
  `1.0157x`.
- Median paired ratio: `1.0170x`; geometric paired ratio: `1.0322x`.
- The candidate won all `6/6` pairs. Every pair cleared the original `0.95x`
  floor, and the complete range was `1.0066x..1.1238x`.
- Pair 3 contained a slow default observation and inflated the geometric mean,
  but the remaining five pairs were independently positive at
  `1.0066x..1.0199x`; the median result does not depend on that outlier.
- Every output remained byte-exact and the predeclared adjudication gate passed.

Promotion:

- Exact compressor projection prebatching is enabled by default when
  `DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH` is unset or empty.
- `DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH=0` or `off` selects the retained
  legacy serial projection route. Explicit `=1` remains accepted for targeted
  trace validation.
- The focused and 32-task HumanEval runners now compare explicit legacy opt-out
  against the promoted unset-variable default. Labels, summaries, metadata,
  and tests were changed together so future runs cannot compare the promoted
  path with itself.
- The adjudication loader retains the pre-promotion historical mode names when
  validating the already-recorded Phase 1.43 artifact, while future executions
  use explicit legacy versus promoted modes.
- After the default flip, both the ordinary exact-runtime correctness matrix
  and an explicit legacy-opt-out matrix passed reasoning, Italian,
  medium-context, rolling-window, and resumed-chat cases byte-for-byte.

Decision:

- PROMOTE. The broad result is positive across 31 tasks, the sole failure was
  decisively non-reproducible, low-acceptance tasks improved, and all outputs
  remained exact.
- This closes the candidate and returns the branch to a clean optimization
  checkpoint. Future cumulative or profiling work should use the promoted
  default and reserve `=0` for attribution and regression diagnosis.

## Phase 1.45: cumulative throughput after compressor prebatch promotion

User-run artifact:

- `speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901`
  at clean commit `8ee89c2`.
- Every current DSpark output matched ordinary baseline and the frozen
  historical artifact byte-for-byte.
- Baseline median: `24.92 t/s`; current DSpark median: `22.25 t/s`; ratio of
  medians `0.8929x`.
- Median paired ratio: `0.8952x`; geometric paired ratio: `0.8993x`.
- DSpark was faster on two tasks, equal on none, and slower on thirty. The
  paired range was `0.8005x..1.0029x`; the geometric gap to parity is `10.1%`.
- Against frozen commit `7b954be`, movement was `1.0416x` geometrically with
  `30/32` tasks improved and a `0.9947x` minimum.

Immediate-checkpoint comparison:

- The directly preceding cumulative artifact is
  `speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-204632`,
  after shared-Q8 promotion and before compressor prebatch promotion.
- Its geometric paired ratio was `0.8814x`; the current `0.8993x` result is
  `1.0203x` geometric movement across matched tasks.
- The median matched-task movement is `1.0209x`; `28/32` tasks improved, four
  regressed, and the movement range was `0.9513x..1.0949x`.
- Baseline and DSpark absolute throughput both moved between sessions
  (`24.23 → 24.92 t/s` baseline median and `21.43 → 22.25 t/s` runtime
  median). Do not attribute the entire cumulative movement to one promotion.
  The same-session prebatch gates remain the authoritative causal evidence.
- The initial machine snapshot included `duetexpertd` at `91.1%` CPU and
  WindowServer at `38.6%`; desktop interference remains present despite the
  balanced paired design.

Decision:

- Progress is real and the direction agrees with the isolated promotion gate,
  but exact DSpark remains below near parity. Do not make an end-to-end speedup
  claim.
- Refresh the cumulative stats-only cost audit against this exact artifact
  before selecting another Metal verifier candidate.

## Phase 1.46: cumulative cost audit repinned

Preparation:

- Updated `CUMULATIVE_SOURCE_COMMIT` in
  `speed-bench/run_dspark_humaneval_cumulative_cost_audit.py` to the clean
  Phase 1.45 commit `8ee89c2ccb8e3d4269fa3f01f1109b1e1878c37d`.
- The provenance test was updated to require that exact commit. Existing
  selection, model identity, threshold, promoted-default, clean-tree, output,
  prompt, and raw-ratio checks remain unchanged.
- A real dry run accepted the Phase 1.45 summary and printed exactly 32
  stats-only exact-runtime processes. Commands contain runtime stats and
  threshold `0.75`, with no fast verifier, oracle trace, or experimental route.
- This is an instrumented diagnostic, not a throughput benchmark. Desktop
  idleness is less critical, but the output and structural counters remain
  strict correctness gates.

User-run command:

```sh
python3 speed-bench/run_dspark_humaneval_cumulative_cost_audit.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901/summary.json \
  --confirm-ready
```

Decision boundary:

- Use the refreshed target, sidecar, residual, and width accounting to choose
  the next candidate. Do not reuse the Phase 1.39 cost split after compressor
  projection scheduling changed.

Cost result:

- User-run artifact:
  `speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512` at clean
  commit `2863d7e`.
- Every stats runtime output matched the frozen Phase 1.45 cumulative artifact
  byte-for-byte.
- Frozen baseline budget: `40.175 ms/emitted`; current DSpark budget:
  `44.694 ms/emitted`; deficit: `4.520 ms/emitted`.
- Fresh components account for `36.356 ms/emitted` target verification,
  `7.596 ms/emitted` generation sidecar, and `0.742 ms/emitted` residual.
- Target verification is `81.3%` of runtime; sidecar is `17.0%`. The
  end-to-end-calibrated target scale for parity is `0.876x`, or a `12.4%`
  reduction. Component-only accounting implies a `10.4%` reduction.
- Target workload is `0.3854` evaluations per emitted token and `2.706`
  positions per evaluation. The schedule remains 760 attempts, 684 full and
  76 partial, with zero fallbacks.
- Width 5 represents 421 evaluations and `54.7%` of target time at
  `33.359 ms/position`. Width 1 represents 600 evaluations and `18.8%` of
  target time at `40.245 ms/position`.
- Scheduler widths 0 and 1 total 502 rounds while still costing roughly
  `19.7 ms` of sidecar per round. This remains a secondary opportunity, but
  target verification is the larger and currently better-attributed surface.
- Sidecar outside the scheduler is `617.597 ms` across all processes. Prefill
  sidecar totals `1175.571 ms`; neither value is used as throughput evidence.

Comparison with Phase 1.39:

- Target cost fell from `38.298` to `36.356 ms/emitted` (`0.949x`).
- Sidecar fell from `7.881` to `7.596 ms/emitted` (`0.964x`).
- Width-5 target cost fell from `35.561` to `33.359 ms/position` (`0.938x`).
- Width-1 target cost fell only from `40.942` to `40.245 ms/position`
  (`0.983x`), consistent with compressor projection prebatch primarily
  benefiting wider verifier batches.
- The calibrated target reduction needed for parity improved from `14.6%` to
  `12.4%`, but target verification still occupies essentially the same share
  of runtime.

Decision:

- Continue optimizing exact multi-row target verification. The promotion
  worked, but did not change the dominant bottleneck class.
- Refresh the width-layer profile before selecting the next candidate because
  compressor prebatch directly changed the serial attention-tail schedule.

## Phase 1.47: post-prebatch width-layer profile repinned

Preparation:

- Updated `CUMULATIVE_COST_SOURCE_COMMIT` in
  `speed-bench/run_dspark_threshold075_width_layer_profile.py` to the clean
  Phase 1.46 cost-audit commit
  `2863d7e26efcfd3e419691b2ae3d0547952eb886`.
- The existing profiler still requires the Phase 1.45 throughput artifact and
  now cross-validates the Phase 1.46 cost artifact, including exact task-079
  width counts and every structural runtime counter.
- It profiles layers 0, 21, and 42 in three separate synchronized processes,
  groups attention-pre, serial-tail, and FFN events by actual verifier width,
  and requires byte-exact output against the frozen cumulative artifact.
- A real dry run accepted both new references and printed only the three
  intended profile commands. No prompt was materialized and no model ran.

User-run command:

```sh
python3 speed-bench/run_dspark_threshold075_width_layer_profile.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512/summary.json \
  --confirm-ready
```

Decision boundary:

- Use stable width-5 medians to choose the next verifier candidate. Widths 2
  and 3 have sparse observations and remain directional only.

Profile result:

- User-run artifact:
  `speed-bench/local-runs/post-promotion-width-layer-20260719-232840` at clean
  commit `83f3e80`.
- Every profiled output matched the frozen cumulative HumanEval artifact
  byte-for-byte.
- Stable width-5 sampled layer cost is `2.809 ms/row`: attention preparation
  `0.656`, serial attention tail `1.083`, and exact FFN `1.070 ms/row`.
- The serial tail is now the largest sampled width-5 stage, but it exceeds the
  FFN by only `0.013 ms/row`; they are effectively tied at this resolution.
- Width-5/width-2 amortization is `0.173x` for attention preparation, `0.628x`
  for the serial tail, and `0.647x` for the FFN. FFN therefore remains the
  slightly weaker-scaling stable stage.
- Widths 2 and 3 have one evaluation each, width 4 has four, and width 5 has
  twenty. Only width-5 values are used to choose the next candidate.

Comparison with the Phase 1.40 profile:

- The prior stable width-5 split was attention preparation `0.578`, serial
  tail `1.167`, and FFN `1.061 ms/row`, totaling `2.806 ms/row`.
- After compressor projection prebatch, the synchronized serial-tail sample
  fell to `1.083 ms/row`, while FFN remained approximately flat at `1.070`.
- The total remained `2.809 ms/row`. These absolute cross-session synchronized
  shifts are attribution clues, not throughput evidence.

Decision:

- Refresh the exact FFN substage profile before selecting another runtime
  candidate. The tail and FFN are tied, but FFN has slightly weaker width
  amortization and an existing exact, byte-checked profiler.

## Phase 1.48: post-prebatch exact FFN profile repinned

Preparation:

- Updated `WIDTH_LAYER_SOURCE_COMMIT` in
  `speed-bench/run_dspark_post_promotion_width_ffn_profile.py` to the clean
  Phase 1.47 profile commit
  `83f3e803e7baa4097cc8c5ff490f72b29aced06c`.
- The provenance test now requires that exact commit. Existing clean-tree,
  model identity, task selection, threshold, promoted-default, output,
  structural-counter, and width-distribution checks remain active.
- The profiler consumes the Phase 1.45 cumulative throughput artifact, Phase
  1.46 cost audit, and Phase 1.47 width-layer profile. It runs three
  synchronized processes for layers 0, 21, and 42 and separates HC, norm,
  router, shared-expert, routed-MoE, and post-HC work by verifier width.
- A real dry run accepted all three frozen references and printed exactly the
  intended three profile commands. No model inference was performed.

User-run command:

```sh
python3 speed-bench/run_dspark_post_promotion_width_ffn_profile.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512/summary.json \
  --width-layer-reference \
  speed-bench/local-runs/post-promotion-width-layer-20260719-232840/summary.json \
  --confirm-ready
```

Decision boundary:

- Use the stable width-5 FFN component shares and amortization to select one
  narrowly scoped Metal verifier candidate. Do not infer throughput from the
  synchronized profile itself.

Profile result:

- User-run artifact:
  `speed-bench/local-runs/post-promotion-width-ffn-20260720-000722` at clean
  commit `fafcddaa86af5d6fc32e86db3414c8db110c6c60`.
- Every profiled output matched the frozen cumulative HumanEval artifact
  byte-for-byte and reproduced the expected width counts `1/1/4/20`.
- At stable width 5, synchronized FFN substages total `2.001 ms/row` against
  an outer `2.067 ms/row` control, a `0.968x` reconciliation.
- Width-5 components are routed MoE `0.806 ms/row` (`40.3%`), shared gate/up
  `0.233` (`11.6%`), router `0.229` (`11.5%`), shared down `0.212` (`10.6%`),
  HC pre `0.197` (`9.9%`), HC post `0.171` (`8.6%`), and normalization `0.152`
  (`7.6%`).
- Routed MoE is stable across sampled depth at `0.267`, `0.271`, and
  `0.268 ms/row` for layers 0, 21, and 42.
- Width-5/width-2 amortization is weakest for routed MoE at `0.518x`; the next
  weakest component is shared down at `0.421x`. Width 2 has only one
  observation, so this ratio is directional; the width-5 share is stable.

Comparison with the pre-hybrid Phase 1.30 profile:

- The old FFN profile measured routed MoE at `0.865 ms/row` (`40.9%`) with
  `0.747x` width-5/width-2 amortization. The refreshed value is `0.806` and
  `0.518x`, consistent with the promoted hybrid improving but not eliminating
  the bottleneck.
- The old inner routed-MoE split measured a fully row-wise implementation and
  attributed `49.5%` to gate/up and `43.7%` to down. That split is stale:
  today's default batches gate/up while retaining exact down/sum arithmetic.
  Do not select a new kernel from the old proportions.

Decision:

- Refresh the inner routed-MoE split on the promoted hybrid before selecting
  another runtime candidate. The current batch encoder already exposes the
  required synchronized stages, so no inference code change is needed.

## Phase 1.49: promoted routed-MoE hybrid profile repinned

Preparation:

- Updated `WIDTH_FFN_SOURCE_COMMIT` in
  `speed-bench/run_dspark_post_promotion_width_routed_moe_profile.py` to the
  clean Phase 1.48 commit
  `fafcddaa86af5d6fc32e86db3414c8db110c6c60`.
- Converted the profiler from `DS4_METAL_MOE_ONE_STAGE_PROFILE`, which observes
  the retired exact row-wise path, to `DS4_METAL_MOE_STAGE_PROFILE`, which
  observes the current promoted batch routed-MoE encoder.
- The mapper now requires exactly one `gate_up`, `activation_weight`, `down`,
  and `sum` sequence inside each enclosing exact routed-MoE interval. It also
  requires the verifier width, `6 * width` expert-pair rows, six experts,
  `hybrid_exact_down_*` path, and F32 intermediate to match.
- Batch-stage timings are divided by verifier width before aggregation. The
  enclosing routed-MoE control is normalized the same way.
- A real dry run accepted the Phase 1.45 throughput, Phase 1.46 cost, Phase
  1.47 layer, and Phase 1.48 FFN references and printed exactly three profile
  commands for layers 0, 21, and 42. No model inference was performed.

User-run command:

```sh
python3 speed-bench/run_dspark_post_promotion_width_routed_moe_profile.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512/summary.json \
  --width-layer-reference \
  speed-bench/local-runs/post-promotion-width-layer-20260719-232840/summary.json \
  --width-ffn-reference \
  speed-bench/local-runs/post-promotion-width-ffn-20260720-000722/summary.json \
  --confirm-ready
```

Decision boundary:

- Use the stable width-5 promoted-hybrid stage shares to choose one bounded
  Metal candidate. Do not reuse the pre-hybrid Phase 1.31 inner proportions,
  and do not treat synchronized absolute times as throughput measurements.
