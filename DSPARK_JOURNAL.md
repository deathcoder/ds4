# DSpark Development Journal

This file is durable working memory for the DSpark-on-ds4 effort. Read it when
resuming after context compaction, switching agents, or feeling unsure why a
particular DSpark change exists.

## Current Phase

Branch: `codex/dspark-observability-0`

Phase 0.38 remains deliberately diagnostic: `--dspark FILE` validates an official
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

- Phase 0.40 needs a user-run comparison with exact batched verification. Ask the
  user to make the machine as quiet as practical, run the harness, and bring
  back `summary.md` and `results.csv`; interpret results alongside captured
  process metadata rather than assuming a fully idle host.
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
