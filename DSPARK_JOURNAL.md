# DSpark Development Journal

This file is durable working memory for the DSpark-on-ds4 effort. Read it when
resuming after context compaction, switching agents, or feeling unsure why a
particular DSpark change exists.

## Current Phase

Branch: `codex/dspark-observability-0`

Phase 0.21 remains deliberately diagnostic: `--dspark FILE` validates an official
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
- The ordinary-session hooks are not called by the CLI `--temp 0` argmax fast
  path. Use a seeded nonzero-temperature smoke when verifying this diagnostic
  path against a normal-generation baseline.
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
    experiment. It exact-verifies a draft suffix, restores its virtual target
    state, replays the accepted prefix through the ordinary target graph, and
    returns the resulting ordered token batch for immediate caller emission.
    It falls back to one target token below a verified depth of two. This is
    deliberately diagnostic rather than fast because replay evaluates accepted
    tokens a second time.
  - Added the public greedy-only session entry point
    `ds4_session_eval_dspark_greedy`; it returns zero when ordinary evaluation
    should proceed and never owns an emission queue.
  - The sidecar probe still does not emit, accept, append, generate, or
    speculate tokens.
  - Added `dspark_model`, `dspark_config`, and `dspark_ready` to `ds4_engine`.
  - `ds4_engine_open` now rejects `--mtp` plus `--dspark`, opens the DSpark
    sidecar, validates/binds it, and logs that runtime is not enabled yet.
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

An untargeted `./ds4_test` run entered its long-context GPU prefill case and was
stopped after about one minute to avoid occupying the accelerator indefinitely.
Do not treat the full test suite as passed for this phase; the focused DSpark
tests and real-sidecar correctness smokes above did pass.

Downloaded sidecar byte size:

```text
/Users/deathcodevision/dev/ds4/gguf/ds4flash-dspark.gguf
11489939840 bytes
```

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
   rg -n -- "ds4_engine_options|ds4_engine_open\\(|--mtp|ds4_help_print"
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

- How should DSpark cache state be represented for real runtime: separate
  sidecar KV buffers mirroring the official two-step cache fill/draft flow, or
  a fresh graph path that directly consumes the target hidden-state window?
- The current `DS4_DSPARK_PROBE=1` / `DS4_DSPARK_VERIFY=1` hooks preserve
  target-layer context from normal prefill/decode and build internal
  `d->draft[]` candidates, but still perform CPU readback and CPU sidecar draft
  computation for diagnostics. The greedy CLI path can now return an emitted
  batch directly, but it restores and replays that prefix for correctness.
  The next runtime improvement should reuse exact verification work or commit
  the accepted target states directly, without changing observable output.
- What queue-aware policy should server, agent, and eval use before accepting a
  multi-token batch? Their stop, tool, and forced structural tokens can alter
  the stream after the first output, so they cannot simply reuse the CLI loop.
- How should a future acceptance path preserve non-greedy sampling semantics?
  The diagnostic separates `greedy_eligible` from `stream_eligible`; committing
  multi-token prefixes is only straightforward for a greedy target stream.
- Where should DSpark runtime stats live so they are useful for development but
  cannot perturb benchmark measurements unless explicitly enabled?
- Should future CLI UX stay as `--dspark`, or eventually become a broader
  `--draft dspark` once runtime is real?
