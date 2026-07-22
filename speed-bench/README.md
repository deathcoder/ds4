## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

## DSpark baseline comparison

DSpark generation must be compared through the normal `ds4` CLI because
`ds4-bench` does not currently load a DSpark sidecar. Build first, stop other
CPU/GPU-heavy work, connect power, wait for the machine to reach a stable
thermal state, and then run the dedicated harness yourself:

```sh
make ds4
python3 speed-bench/run_dspark_comparison.py --dry-run
python3 speed-bench/run_dspark_comparison.py --confirm-idle
python3 speed-bench/run_dspark_comparison.py --dry-run --fast-verifier
python3 speed-bench/run_dspark_comparison.py --confirm-idle --fast-verifier
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --serial-ffn-ablation
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --serial-ffn-ablation
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --attention-pre-ablation
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --attention-pre-ablation
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --exact-q8-rows-ablation
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --exact-q8-rows-ablation
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --attention-suffix-ablation
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --attention-suffix-ablation
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --compressor-pair-nr4-ablation
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --compressor-pair-nr4-ablation
python3 speed-bench/run_dspark_exact_attention_suffix_profile.py --dry-run
python3 speed-bench/run_dspark_exact_attention_suffix_profile.py --confirm-ready
python3 speed-bench/run_dspark_exact_attention_tail_profile.py --dry-run
python3 speed-bench/run_dspark_exact_attention_tail_profile.py --confirm-ready
python3 speed-bench/run_dspark_exact_compressor_profile.py --dry-run
python3 speed-bench/run_dspark_exact_compressor_profile.py --confirm-ready
python3 speed-bench/run_dspark_exact_attention_transition_profile.py --dry-run
python3 speed-bench/run_dspark_exact_attention_transition_profile.py --confirm-ready
python3 speed-bench/run_dspark_exact_ffn_batch_profile.py --dry-run
python3 speed-bench/run_dspark_exact_ffn_batch_profile.py --confirm-ready
```

The runner deliberately refuses real execution without `--confirm-idle`. Treat
that flag as confirmation that you have made the machine as quiet as practical;
the captured process and thermal metadata records unavoidable interference. Its
defaults perform one warmup per mode followed by three measured pairs. Pair
order alternates baseline/runtime then runtime/baseline, with a 10-second
cooldown between processes. Each process uses greedy generation, a fixed seed,
64 generated tokens, and `speed-bench/dspark_prompt.txt`. DSpark retains a
rolling 128-token sidecar attention window, so the target prefix may be longer;
this small fixture remains the focused steady-generation microbenchmark.

Default runtime runs set only:

```text
DS4_DSPARK_GPU_RUNTIME=1
DS4_DSPARK_MULTI_COMMIT=1
DS4_DSPARK_GPU_RUNTIME_STATS=1
```

Exact multi-row verification uses batched all-layer FFN execution by default.
Set `DS4_DSPARK_EXACT_FFN_BATCH=0` only when the old serial FFN implementation
is needed as a diagnostic control. The selected-layer FFN observer also keeps
the batched runtime path non-authoritative for the observed call.

Exact multi-row verification also prepares the cache-independent attention
prefix across proposal rows by default. It batches HC mixing, attention norm,
Q-LoRA normalization, and Q/KV RoPE preparation, then runs the unchanged
cache-mutating attention tail serially in autoregressive row order. Set
`DS4_DSPARK_EXACT_ATTN_PRE_BATCH=0` only to select the old fully serial
attention path as a diagnostic control. The selected-layer attention-pre
observer keeps the batched preparation non-authoritative for the observed
call.

Exact attention preparation uses the Metal 2-5-row Q8 microbatch by default for
the Q-LoRA, KV, and final Q projections. Each proposal row retains the ordinary
decode kernel's lane assignment, block traversal, accumulation sequence, SIMD
reduction, and output layout; each quantized weight block is loaded once and
applied to all rows. Cache writes and the autoregressive attention tail are
unchanged. Set `DS4_DSPARK_EXACT_Q8_ROWS=0` only to force the legacy separate
one-token Q8 launches as a diagnostic control. Use `--exact-q8-rows-ablation`
for the uninstrumented legacy-control versus promoted-default confirmation.
Both modes clear diagnostics and runtime stats, alternate order over three
pairs, and require byte-identical output.

The exact attention suffix candidate is opt-in through
`DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH=1`. It preserves the serial, autoregressive
cache and attention core through inverse RoPE, then batches projection A and
the exact fused projection-B/HC expansion across proposal rows. A failed batch
attempt replays only that stateless suffix with the ordinary one-row kernels.
The default exact runtime remains unchanged while this candidate is evaluated.

`--fast-verifier` additionally sets
`DS4_DSPARK_FAST_BATCH_VERIFY=1`. This is deliberately opt-in: it makes the
compute-batched verifier authoritative for suffix tops and continuation logits,
while exact verification remains the fallback after recoverable fast-path
failure. Partial accepts restore the target frontier and rerun only the accepted
prefix with the same verifier. The harness still requires byte-identical output
against baseline and aborts immediately if numerical drift changes the stream.
Fast authority is currently limited to the first synchronized prompt of a
session. A resumed sync permanently suspends it for that session and uses exact
batch verification with the default exact FFN path, because broader soak
testing found a second-turn numerical divergence while one-shot output remained
identical.

`--serial-ffn-ablation` changes the two paired modes: instead of comparing
non-DSpark baseline against DSpark, it compares the serial exact control
(`DS4_DSPARK_EXACT_FFN_BATCH=0`) against the promoted default exact FFN path.
Pair order alternates serial/default then default/serial. Both sides use the
same sidecar, prompt, acceptance policy, and target verifier except for the
all-layer FFN implementation. This option is mutually exclusive with
`--fast-verifier`. The old `--exact-ffn-batch-ablation` spelling remains an
alias.

The serial-FFN ablation is an uninstrumented throughput pass. It disables runtime
stats as well as diagnostics in both modes, requires every stdout stream to be
byte-identical to the first serial-exact warmup, and reports serial/default
medians, paired ratios, and the default percentage delta. Raw streams,
metadata, pair order, and CSV/JSON/Markdown summaries use the same run directory
as the ordinary comparison.

`--attention-pre-ablation` is also an uninstrumented paired throughput pass. It
compares the fully serial attention control
(`DS4_DSPARK_EXACT_ATTN_PRE_BATCH=0`) against the promoted default exact
attention-pre path. Both modes use the same DSpark sidecar, exact all-layer FFN
path, prompt, acceptance policy, and serial cache/attention tail. Diagnostics
and runtime stats are disabled, and the runner requires every stdout stream to
remain byte-identical. Pair order alternates serial/default then default/serial.
The report includes both medians, paired ratios, and the default percentage
delta.

This mode is mutually exclusive with `--fast-verifier` and
`--serial-ffn-ablation`. It measures only the exact verifier path; the fast
verifier has its own preparation and is not part of this ablation.

`--attention-suffix-ablation` is an uninstrumented paired pass comparing the
promoted default exact runtime against the opt-in exact attention-suffix batch
candidate. Both modes use the same sidecar, attention preparation, FFN path,
prompt, and acceptance policy. Diagnostics and runtime stats are disabled,
stdout must remain byte-identical, and pair order alternates default/candidate
then candidate/default. This mode is mutually exclusive with every other
ablation and `--fast-verifier`.

The measured suffix candidate is currently slower than default exact Metal
generation. Use `run_dspark_exact_attention_suffix_profile.py` before changing
that path again. The synchronized diagnostic compares default serial attention
tail work against candidate serial core direct-write, batched projection A, and
batched fused projection-B/HC at representative early, middle, and late layers.
It also reports attention-pre and FFN control medians, requires identical
proposal schedules and byte-identical output, and rejects incomplete or unknown
stage records. Its boundaries deliberately alter scheduling, so its timings are
for attribution only and must not be reported as generation throughput.

`--compressor-pair-nr4-ablation` is an uninstrumented paired pass comparing
default exact DSpark against the existing opt-in
`DS4_METAL_COMPRESSOR_PAIR_NR4=1` projection schedule. The runner explicitly
removes any inherited NR4 setting from both environments, enables it only for
the candidate, disables diagnostics and runtime stats, alternates pair order,
and requires every output to remain byte-identical. Before benchmarking, run
the candidate-only correctness matrix:

```sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_COMPRESSOR_PAIR_NR4=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
```

That script unsets NR4 for each baseline process and enables it only for the
exact DSpark candidate, so its five output comparisons directly test the new
projection schedule rather than comparing two NR4 executions.

RB16-direct is the default one-token sparse-attention route when top-k is 512
and every compressed cache row is visible. It keeps the RB16 row block, all
selected rows, selected-row order, and online-softmax arithmetic while removing
the redundant per-thread scan and local `rows[16]` array. Every other indexed
attention case continues to use the legacy RB16 or general kernel. Set
`DS4_METAL_INDEXED_ATTN_RB16_LEGACY=1` only to force the legacy one-token route
as a correctness or performance control.

`--indexed-attention-rb16-promotion-ablation` is the uninstrumented final
confirmation pass. It compares explicit legacy RB16 against the promoted
environment-free default, clears inherited route, trace, and sparse-threshold
settings, disables diagnostics and runtime stats, alternates pair order, and
requires byte-identical output. The old
`--indexed-attention-rb16-direct-ablation` spelling remains an alias. The
default short fixture never reaches sparse indexed attention, so use the 8K
transition fixture explicitly:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --dry-run --indexed-attention-rb16-promotion-ablation \
  --prompt-file speed-bench/issue468/code_8k.txt --ctx 16384 --tokens 128
python3 speed-bench/run_dspark_comparison.py \
  --confirm-idle --indexed-attention-rb16-promotion-ablation \
  --prompt-file speed-bench/issue468/code_8k.txt --ctx 16384 --tokens 128
```

The promotion-specific correctness matrix lowers only the implementation
threshold and generates a temporary prompt long enough to cross the immutable
512-row selector frontier. It compares base output, explicit legacy RB16, and
the promoted default, and requires a promoted-default route trace:

```sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_INDEXED_ATTN_RB16_PROMOTION=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
```

After retiring deferred suffix batching, use
`run_dspark_exact_attention_tail_profile.py` to inspect the retained serial tail
without reordering its operations. It emits one-row synchronized records for
KV/cache update, compressor/indexer work, attention, inverse RoPE, projection A,
and fused projection-B/HC. The runner cross-checks those row positions against
the exact verifier's proposal batches, keeps attention-pre and FFN medians as
controls, and requires byte-identical output. These timings also change
scheduling and are attribution data, not throughput measurements.

Use `run_dspark_exact_compressor_profile.py` to split the compressed portion of
that retained tail in place. Its defaults profile ratio-128 layer 21 and
ratio-4 layer 42 over 128 generated tokens on the 8K code fixture. With the
base model's tokenizer this prompt begins near position 3997, so the longer
generation crosses the default 1024-compressed-row sparse threshold near
position 4099. The profile separates main compressor projection from recurrent
update/emit; layer 42 additionally separates indexer compressor
projection/update, query and weight preparation, score, and the full 512-row
top-k. The runner classifies emit rows from absolute positions, cross-checks
every component row against exact proposal batches, and requires byte-identical
output. Dense ratio-4 rows legitimately omit prepare/score/top-k; if sparse
mode begins during a run, all three sparse stages must cover the same valid
proposal-row subset. The report labels ratio-4 runs as dense, sparse, or a
dense-to-sparse transition. This is synchronized attribution, not a
generation-throughput benchmark.

Use `run_dspark_exact_attention_transition_profile.py` to measure only the
retained attention call while ratio-4 layer 42 crosses from dense mixed
attention into sparse indexed attention. The default 8K code fixture and
128-token generation preserve the normal sparse threshold and full 512-row
top-k contract. Every row is labeled from the branch actually selected by the
runtime (`raw`, `dense_mixed`, or `sparse_indexed`), and all mode records
together must match the exact verifier's proposal-row multiset. The default
runner rejects a result unless at least one selected layer contains both dense
and sparse records; `--allow-single-mode` is available only for deliberate
short diagnostic runs. It also requires byte-identical output and matching
attention-pre/FFN control schedules. Its immediate before/after boundaries are
identical across modes, but synchronization changes scheduling, so compare the
dense and sparse attribution within this run only and never report it as
generation throughput.

Pass `--rb16-promotion-comparison` to repeat that synchronized attribution with
explicit legacy RB16 and the promoted RB16-direct default in separate
processes. The
runner requires both variants to produce byte-identical output, identical
proposal schedules, and the same dense/sparse branch labels. Its report shows
promoted/legacy ratios for dense and sparse attention separately, with
attention-pre and FFN controls. A promoted-route win should appear in sparse
indexed attention; dense attention and both controls do not use either indexed
kernel and should remain stable. The old `--rb16-direct-comparison` spelling
remains an alias. Run this diagnostic yourself because its
synchronization points make its timings sensitive to other machine activity:

```sh
python3 speed-bench/run_dspark_exact_attention_transition_profile.py \
  --dry-run --rb16-promotion-comparison
python3 speed-bench/run_dspark_exact_attention_transition_profile.py \
  --confirm-ready --rb16-promotion-comparison
```

After an uninstrumented serial-FFN ablation, use
`run_dspark_exact_ffn_batch_profile.py` as a separate attribution pass. It
compares the same serial-exact control and default exact-FFN modes with
end-of-session runtime stats enabled in both. The profile reports target-layer,
output-head, residual target, and generation-sidecar milliseconds per emitted
token. It also checks that serial exact never selects FFN batching and reports
default exact-FFN completion counts plus verifier fallbacks. Output remains
byte-identical across modes.

This profile is intentionally instrumented and synchronized. Its printed t/s
values are context only and must not be mixed with the uninstrumented throughput
ablation. Runtime diagnostics remain disabled, so it adds clock reads and one
summary record rather than per-call hot-path logging. The default is one
alternating pair with no warmup; use `--pairs` only when diagnostic timing is too
variable to interpret.

All inherited `DS4_DSPARK_*` variables and instrumentation variables containing
`PROFILE`, `TRACE`, `DUMP`, or `TIMING` (plus `*_LOG`) are removed first. Runtime
diagnostic logs remain disabled. In the ordinary baseline/runtime comparison,
the stats option adds only clock reads and one machine-readable record when the
session closes; it reports acceptance depth, target graph calls and token
positions, exact-batch verifier outcomes, and bridge/stage/head/chain timing
split between prefill and generation. The serial-FFN ablation leaves that option
unset. Other `DS4_*` tuning variables are preserved and recorded. Every run must
produce byte-identical stdout; the harness aborts on drift. Raw stdout/stderr,
environment metadata, process and thermal snapshots, per-run CSV data, and
median/paired-speedup summaries go under the ignored
`speed-bench/local-runs/` directory.

On a resident non-streaming Metal graph, DSpark runtime verification batches up
to five proposed tokens into one exact target command stream. It retains the
normal one-token decode kernels and cache-update order. Full accepts commit the
result directly; partial accepts restore the saved target frontier and rerun
only the accepted prefix as a second exact batch. Unsupported graph shapes or
recoverable batch failures automatically use the serial verifier.

`DS4_DSPARK_FAST_VERIFY_OBSERVER=1` is a correctness-development option, not a
benchmark setting. It runs the throughput-oriented legacy batch verifier from
a snapshotted target frontier, reads its row tops and final logits, restores
state, and then lets the exact verifier remain authoritative. Each proposal
records top-token parity, final-logit drift, and fast/exact verifier time. The
comparison harness clears this option like every other inherited DSpark option.

The non-performance correctness soak is user-independent and may be run with:

```sh
./tests/dspark_fast_verifier_soak.sh
./tests/dspark_exact_ffn_batch_runtime_soak.sh
./tests/dspark_exact_attention_pre_batch_runtime_soak.sh
DS4_TEST_DSPARK_SERIAL_Q8_ROWS_RUNTIME=1 \
  ./tests/dspark_exact_attention_pre_batch_runtime_soak.sh
./tests/dspark_exact_attention_suffix_batch_runtime_soak.sh
```

It compares baseline and fast output across 64-token multi-cycle generation,
code fallback, Italian, Spanish, structured JSON, near-window context, strict
margin gating, and a resumed two-turn session. It requires fast commits where eligible,
rejects verifier/capture failures, and verifies the resumed fast-to-exact
transition.
The exact-FFN soak separately covers 64-token generation, rolling-window state,
and successful default exact-FFN verification after a resumed sync.
The exact attention-pre soak covers the same long-generation, rolling-window,
and resumed-session boundaries while requiring all 43 layers to use the
default prepared path without fallback.
The exact attention-suffix soak covers those boundaries with the opt-in suffix
candidate and requires every layer attempt to complete without serial fallback.

Useful explicit overrides include `--pairs`, `--warmups`, `--cooldown`,
`--tokens`, and `--output-dir`. Do not compare runs with different settings.
The CLI timing line excludes model loading, while the recorded wall time does
not; generation t/s from the timing line is the comparison metric.

### Cold versus warm prefill

The normal CLI comparison starts a new process for every sample. That is useful
for generation throughput, but its first prefill also includes process-local
Metal initialization, page residency, and DSpark graph setup. Use the dedicated
warm-prefill helper to separate that cold cost from steady fresh-session
prefill:

```sh
make ds4-warm-prefill-bench
python3 speed-bench/run_dspark_warm_prefill.py --dry-run
python3 speed-bench/run_dspark_warm_prefill.py --confirm-ready
```

Each child process opens one engine, records its first fresh session as `cold`,
then measures three additional fresh sessions as `warm` while the engine remains
open. Session creation and destruction are outside the timer. The warm sessions
do not reuse KV cache; only process-local model mappings, compiled pipelines,
and GPU residency survive. The outer runner alternates baseline/runtime process
order over three pairs and waits ten seconds between children by default.

The runtime mode sets only `DS4_DSPARK_GPU_RUNTIME=1` and
`DS4_DSPARK_MULTI_COMMIT=1`. Runtime stats and diagnostics are deliberately
disabled so their logging cannot affect the result. Inherited DSpark and timing
instrumentation variables are cleared in both modes. Every sample copies and
hashes the complete target logits vector; the run aborts if prompt length,
argmax, or the hash differs from the first baseline sample.

`--confirm-ready` means only that the user has made the machine as quiet and
thermally stable as practical. Process and thermal snapshots are retained with
raw child CSV/stderr, flattened samples, metadata, and cold/warm summaries under
`speed-bench/local-runs/warm-prefill-<timestamp>/`. Useful overrides are
`--pairs`, `--warmups`, `--runs`, `--cooldown`, and `--output-dir`.

The summary also reports complete child-process wall time and non-sync overhead
(child wall time minus all recorded sync durations). Use those fields when an
optimization moves setup work across the sync timer: improved first-sync t/s is
not a startup win unless child wall time or non-sync-adjusted total also falls.

### Issue 468 long-prompt comparison

Use the dedicated corpus runner to compare DSpark against baseline on the exact
three 8k prompt fixtures from the published issue-468 legacy MTP study:

```sh
make ds4
python3 speed-bench/run_dspark_issue468_comparison.py --dry-run
python3 speed-bench/run_dspark_issue468_comparison.py --confirm-ready
```

The authoritative throughput pass matches the source workload at `ctx=16384`,
128 generated tokens, temperature zero, and seed one. It deliberately omits
`--nothink`, as did the source harness. It performs one warmup per mode and
prompt, then three alternating measured pairs with a ten-second cooldown. The
runtime uses the exact DSpark verifier by default.

All inherited `DS4_*` variables are cleared from every child process. Throughput
runs enable only the GPU runtime and multi-commit, so Metal route controls,
diagnostics, and runtime statistics cannot enter their medians. The inherited
keys actually removed and the explicit child-environment policy are recorded in
metadata.

After a completed throughput comparison, use `--stats-only` for attribution
without repeating the paired benchmark. It runs one fresh uninstrumented
baseline reference and one stats-enabled exact runtime per prompt, requires
byte-identical output, omits throughput conclusions, and reports acceptance,
target-verifier, sidecar, and fallback measurements:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py \
  --confirm-ready --stats-only
```

Use `--acceptance-audit` when the question is proposal quality rather than
speed. It runs one fresh baseline and one exact audited runtime per prompt,
requires byte-identical output, and reports the paper-aligned accepted length
plus position-wise acceptance and confidence without making throughput claims:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py \
  --confirm-ready --acceptance-audit
```

To control for the paper's non-thinking generation mode on the same prompts,
run a second audit against the first audit's `summary.json`:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py \
  --confirm-ready \
  --acceptance-audit \
  --nothink \
  --acceptance-reference \
  speed-bench/local-runs/issue468-acceptance-<timestamp>/summary.json
```

The reference is accepted only when it used the opposite thinking mode and
matches the current binary/model paths, context, token count, seed, and prompt
hashes. The report adds accepted-length, verify-rate, conditional-acceptance,
and prefix-survival deltas. Non-thinking audit outputs use the
`issue468-acceptance-nothink-<timestamp>/` directory prefix.

For a corpus/domain isolation check against one of the paper's named code
benchmarks, run the frozen HumanEval acceptance study:

```sh
python3 speed-bench/run_dspark_humaneval_acceptance.py --dry-run
python3 speed-bench/run_dspark_humaneval_acceptance.py --confirm-ready
```

The default gate runs 32 evenly spaced rows from the complete pinned 164-row
DeepSpec corpus. Use `--sample-count 8` to reproduce the original pilot or
`--sample-count 164` for the complete dataset. This is an acceptance diagnostic,
not a throughput benchmark: it runs one fresh baseline/exact-runtime pair per
sample with no cooldown by default, requires byte-identical output, and omits
timing values from its result CSV and report. Aggregates pool proposal rounds.

The study uses byte-exact DeepSpec `turns[0]` content, non-thinking mode, and no
confidence scheduler. It deliberately retains the V4-Flash five-draft, greedy,
128-output-token protocol so corpus is the isolated variable. The report shows
the paper's HumanEval range (`5.38-5.64` accepted length, or `0.672-0.705`
normalized verify rate) only as a directional target; Table 1 used all 164
samples, Qwen3/Gemma4 block-seven checkpoints, temperature-1.0 rejection
sampling, and up to 2048 output tokens. Raw outputs go under
`speed-bench/local-runs/humaneval-acceptance-<count>-<timestamp>/`.

Once the 32-sample acceptance gate is complete, measure the identical workload
with the paired, uninstrumented throughput runner:

```sh
python3 speed-bench/run_dspark_humaneval_throughput.py \
  --dry-run \
  --acceptance-reference \
  speed-bench/local-runs/humaneval-acceptance-32-<timestamp>/summary.json
python3 speed-bench/run_dspark_humaneval_throughput.py \
  --confirm-ready \
  --acceptance-reference \
  speed-bench/local-runs/humaneval-acceptance-32-<timestamp>/summary.json
```

The reference is accepted only when its exact 32-row selection, pinned source,
binary and model paths, context, token count, non-thinking mode, seed, greedy
protocol, and exact verifier match the throughput workload. Two prompt pairs
warm the process-local paths with opposite mode order and are excluded. Each of
the 32 tasks then contributes one byte-equal baseline/runtime pair; pair order
alternates by task and a three-second cooldown follows every child by default.
Only the GPU runtime and multi-commit variables are enabled for DSpark. Stats,
acceptance auditing, route controls, and profilers remain disabled.

The primary result is the median of the 32 within-task DSpark/baseline ratios.
The report also gives the geometric mean, interquartile range, faster/equal/
slower task counts, and a descriptive correlation between the prior task-level
acceptance rates and measured speed ratios. Acceptance is loaded from the
separate validated audit; it is never instrumented during the throughput run.
Absolute t/s is retained for context but should be collected only by the user
with the machine as quiet and thermally stable as practical. Raw artifacts go
under `speed-bench/local-runs/humaneval-throughput-32-<timestamp>/`.

After a promoted-runtime throughput rerun, attribute exact-verifier cost on
representative low- and high-acceptance HumanEval tasks with:

```sh
python3 speed-bench/run_dspark_humaneval_exact_profile.py \
  --dry-run \
  --throughput-reference \
  speed-bench/local-runs/humaneval-throughput-32-<timestamp>/summary.json
python3 speed-bench/run_dspark_humaneval_exact_profile.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-throughput-32-<timestamp>/summary.json
```

The default tasks are `humaneval_152` (acceptance `0.528`, prior speed ratio
`0.4541x`) and `humaneval_079` (acceptance `0.839`, prior speed ratio
`0.8134x`). The runner validates both against the frozen corpus and supplied
uninstrumented throughput artifact, then profiles layers 0, 21, and 42. A
stats-only exact reference supplies emitted-token and target-evaluation counts;
each synchronized layer run keeps stats disabled and measures attention
preparation, the serial attention tail, and exact FFN. Every output must match
the prior uninstrumented runtime output byte-for-byte. The report presents
component medians per proposal row alongside synchronized totals per emitted
token, making acceptance-driven invocation amplification visible without
mislabeling profiler timings as throughput. Raw artifacts go under
`speed-bench/local-runs/humaneval-exact-profile-<timestamp>/`.

To test the official-style confidence-prefix rule without immediately repeating
the full 32-task throughput study, use the predeclared scheduler ablation:

```sh
python3 speed-bench/run_dspark_humaneval_scheduler_ablation.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-throughput-32-<timestamp>/summary.json
python3 speed-bench/run_dspark_humaneval_scheduler_ablation.py \
  --confirm-idle \
  --throughput-reference \
  speed-bench/local-runs/humaneval-throughput-32-<timestamp>/summary.json
```

The three modes are fixed K=5, threshold `0.38`, and threshold `0.455`. The two
tasks are fixed before measurement: low-acceptance `humaneval_152` and high-
acceptance `humaneval_079`. With the default three pairs and one warmup period,
the runner starts 21 exact DSpark processes: three excluded warmups and 18
measured runs. Each task uses a three-period Latin rotation so every mode
occupies every order position. All outputs must match the prior exact runtime
artifact byte-for-byte.

`DS4_DSPARK_CONFIDENCE_THRESHOLD` uses DeepSpec's strict prefix rule: select
the drafts before the first raw confidence below the threshold. DSpark runtime
defaults to threshold `0.455`; an explicit value overrides it. Threshold `0`
is the fixed-K=5 control. The complete five-row sidecar block is still computed,
and selecting K=0 uses ordinary one-token target evaluation. Invalid or
non-finite values outside `[0,1]` fail explicitly. The benchmark enables no
stats, audit, trace, diagnostics, or profiler. Its candidate/fixed paired ratios
measure actual Metal cost; offline target-position proxies are not throughput
results. Do not change the tasks or thresholds after observing this gate.

To test whether substantially more aggressive scheduling can reduce the
remaining exact-verifier cost, use the frozen eight-task gate:

```sh
python3 speed-bench/run_dspark_humaneval_aggressive_scheduler_gate.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-<timestamp>/summary.json
python3 speed-bench/run_dspark_humaneval_aggressive_scheduler_gate.py \
  --confirm-idle \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-<timestamp>/summary.json
```

The four frozen modes are ordinary target baseline, current threshold `0.455`,
threshold `0.75`, and threshold `0.85`. The eight tasks cover low acceptance,
middle acceptance, high current throughput, high profitable-round share, and a
high-acceptance adversarial control. One measured run per mode and task yields
32 measured processes; four global warmups are excluded. The four mode
positions are balanced exactly twice across the tasks.

Every mode must match the frozen exact output byte-for-byte. No runtime stats,
audit, trace, diagnostics, or profiler is enabled. A candidate passes only if
its geometric mean versus threshold `0.455` is at least `1.03x`, it wins at
least six of eight tasks, and no task is below `0.90x`. If both candidates pass,
threshold `0.85` is selected only when its geometric mean is at least `1.01x`
the threshold-`0.75` result; otherwise the lower threshold wins.

After the aggressive gate selects threshold `0.75`, confirm it on all 32 frozen
HumanEval tasks:

```sh
python3 speed-bench/run_dspark_humaneval_threshold075_throughput.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-<timestamp>/summary.json \
  --gate-reference \
  speed-bench/local-runs/humaneval-aggressive-scheduler-<timestamp>/summary.json
python3 speed-bench/run_dspark_humaneval_threshold075_throughput.py \
  --confirm-idle \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-<timestamp>/summary.json \
  --gate-reference \
  speed-bench/local-runs/humaneval-aggressive-scheduler-<timestamp>/summary.json
```

The runner validates that the aggressive gate selected `0.75`, then pairs
ordinary target baseline against exact DSpark threshold `0.75` on the same
frozen 32 tasks. Two global warmup pairs are excluded. The 64 measured
processes alternate baseline-first and runtime-first order. Every process must
match the frozen output, and no instrumentation or fast verifier is enabled.

Within-run threshold-`0.75`/baseline ratios are authoritative. The prior
threshold-`0.455` ratios are descriptive cross-run context. Broad scheduler
confirmation requires at least `1.05x` geometric task-level movement over the
prior study, at least 24 improved tasks, and no task below `0.80x` movement.
The next path remains scheduler acceptance/cost auditing only if the fresh
baseline geometric mean reaches `0.95x` or at least eight tasks become faster
than baseline. Otherwise freeze scheduler tuning at `0.75` and return to exact
Metal verifier optimization.

After freezing the scheduler, collect exact-verifier costs under the confirmed
threshold-`0.75` schedule:

```sh
python3 speed-bench/run_dspark_humaneval_threshold075_cost_audit.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-<timestamp>/summary.json
python3 speed-bench/run_dspark_humaneval_threshold075_cost_audit.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-<timestamp>/summary.json
```

The audit runs one stats-only exact DSpark process for each frozen task and
requires every output to match the completed uninstrumented threshold-`0.75`
artifact byte-for-byte. It reports target and sidecar cost per emitted token,
verifier and scheduler width economics, batch outcomes, and two target-time
parity scales:

- an end-to-end-calibrated scale that assigns the frozen measured deficit to
  target verification while keeping other costs fixed;
- a component-accounted scale based only on fresh target and sidecar timings.

The two scales must be read with the report's cross-run residual warning. No
fresh baseline, timed throughput pass, acceptance audit, oracle trace, layer
profiler, or fast verifier is enabled.

To screen a measured-cost verifier-width policy without model execution, pair
the frozen five-position confidence trace with the latest cumulative cost
audit:

```sh
python3 speed-bench/analyze_dspark_cost_aware_scheduler.py \
  --trace \
  speed-bench/local-runs/humaneval-scheduler-trace-32-20260715-165938/scheduler_trace.csv \
  --cost-summary \
  speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512/summary.json \
  --output-dir \
  speed-bench/local-runs/humaneval-cost-aware-scheduler-<timestamp>
```

The analyzer compares static threshold `0.75`, a raw-confidence policy that
maximizes expected runtime tokens returned over measured round cost, fixed
`K=2`, and a realized per-round oracle. Candidate widths are `{0,2,3,4,5}`
because ds4's `K=0` and `K=1` paths both emit one ordinary target token. For
`K>=2`, runtime progress is `max(1, min(accepted drafts, K))`; the paper's
following bonus token is not credited to the same scheduler round. Every
round is charged the full sidecar cost, since DSpark computes all five drafts
before choosing verifier width. The result is a local counterfactual with pooled
instrumented costs and frozen proposal boundaries, not a throughput forecast;
it is only a gate for implementing an opt-in runtime controller.

After the cost audit identifies multi-row exact verification as the dominant
surface, profile its scaling by actual verifier width:

```sh
python3 speed-bench/run_dspark_threshold075_width_layer_profile.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-<timestamp>/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-threshold075-cost-<timestamp>/summary.json
python3 speed-bench/run_dspark_threshold075_width_layer_profile.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-<timestamp>/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-threshold075-cost-<timestamp>/summary.json
```

The diagnostic is frozen to `humaneval_079`, target layers `0`, `21`, and
`42`, and exact verifier widths `2-5`. It runs three synchronized profile
processes, validates each process's output and width histogram against the
completed threshold-`0.75` artifacts, then groups attention preparation,
serial attention tail, and exact FFN records by their actual width. Widths two
and three have one observation each on this task, so they are directional
anchors; width five has twenty observations and is the stable optimization
target.

When that profile identifies layer `42`'s serial attention tail as the
weakest-amortizing component, split the tail by verifier width:

```sh
python3 speed-bench/run_dspark_threshold075_width_tail_profile.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512/summary.json \
  --layer-reference \
  speed-bench/local-runs/post-promotion-width-layer-20260719-232840/summary.json
python3 speed-bench/run_dspark_threshold075_width_tail_profile.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512/summary.json \
  --layer-reference \
  speed-bench/local-runs/post-promotion-width-layer-20260719-232840/summary.json
```

This is one synchronized layer-`42` process. It maps each one-row tail event
back to its enclosing proposal batch by control-stage sequence, which remains
unambiguous even if later target evaluations overlap earlier target
positions. The report separates KV/cache update, compressor/indexer,
attention, inverse RoPE, projection A, and projection B plus HC for widths
`2-5`. The current provenance pins the Phase 1.45 cumulative throughput,
Phase 1.46 cost audit, and Phase 1.47 post-prebatch width-layer profile.

After producing the current post-prebatch tail artifact, split its attention
component by the route actually selected for every proposal row:

```sh
python3 speed-bench/run_dspark_threshold075_width_attention_profile.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512/summary.json \
  --layer-reference \
  speed-bench/local-runs/post-promotion-width-layer-20260719-232840/summary.json \
  --tail-reference \
  speed-bench/local-runs/post-promotion-width-tail-20260720-123704/summary.json
python3 speed-bench/run_dspark_threshold075_width_attention_profile.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512/summary.json \
  --layer-reference \
  speed-bench/local-runs/post-promotion-width-layer-20260719-232840/summary.json \
  --tail-reference \
  speed-bench/local-runs/post-promotion-width-tail-20260720-123704/summary.json
```

This remains one synchronized layer-`42` diagnostic process. It assigns raw,
dense-mixed, and sparse-indexed attention events to the enclosing verifier
batch and reports both row share and synchronized cost share for widths `2-5`.
Use stable width-5 cost share to select the next route-specific Metal target;
the slowest route per row can still be irrelevant if it is rarely selected.
The runner is pinned through the clean Phase 1.54 tail artifact and enables no
runtime candidate or throughput pass.

When the route report confirms that the current frozen workload is entirely
dense-mixed, split the promoted fused-gather call into its three production
dispatches:

```sh
python3 speed-bench/run_dspark_threshold075_width_attention_profile.py \
  --fused-gather-stages --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-223901/summary.json \
  --cost-reference \
  speed-bench/local-runs/humaneval-cumulative-cost-20260719-225512/summary.json \
  --layer-reference \
  speed-bench/local-runs/post-promotion-width-layer-20260719-232840/summary.json \
  --tail-reference \
  speed-bench/local-runs/post-promotion-width-tail-20260720-123704/summary.json \
  --attention-reference \
  speed-bench/local-runs/post-promotion-width-attention-20260720-124940/summary.json
```

The additional diagnostic reports fused preparation, unchanged vector
FlashAttention, and unchanged reduction separately for widths `2-5`. It
requires the pinned route artifact to contain only dense-mixed rows at every
width and retains the same output and proposal-schedule gates. The stage
boundaries exist only under `DS4_METAL_FLASH_ATTN_GATHERED_PROFILE`; ordinary
execution retains the unsynchronized production command sequence.

After threshold `0.455` passes that representative gate, run the frozen
32-task baseline-versus-scheduled confirmation without changing the policy:

```sh
python3 speed-bench/run_dspark_humaneval_throughput.py \
  --dry-run --allow-dirty \
  --confidence-scheduler \
  --scheduler-reference \
  speed-bench/local-runs/humaneval-scheduler-trace-32-20260715-165938/scheduler_summary.json \
  --acceptance-reference \
  speed-bench/local-runs/humaneval-acceptance-32-20260715-121045/summary.json \
  --historical-throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260715-182840/summary.json
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

This mode accepts no free-form threshold. It validates that the supplied
scheduler artifact is the prior 32-task K=5 trace, that its `97.5%` retention
policy selected held-out median threshold `0.455`, and that the trace derives
from the supplied acceptance artifact. It then runs the original frozen
HumanEval selection and alternating pair order with scheduled DSpark in place
of fixed K=5. The two global warmup pairs are excluded, so the default schedule
starts 68 children: four warmups and 64 measured runs. Every scheduled output
must match its fresh ordinary-target baseline byte-for-byte. Results go under
`speed-bench/local-runs/humaneval-scheduler-throughput-32-<timestamp>/`.

When `--historical-throughput-reference` is supplied, the runner validates the
prior frozen task selection, scheduler threshold, protocol, model paths, and
acceptance/scheduler references. It then reports descriptive task-level
movement in DSpark t/s and DSpark/baseline ratios. These cross-run values are
context only; the new within-run paired ratio remains authoritative.

The new within-run scheduled/baseline ratios are authoritative. Phase 0.94's
fixed-K median paired ratio of `0.6840x` is useful historical context, but it
must not be divided into the new result as if both modes were measured in the
same process sequence. At that stage, promotion remained blocked on the
non-code gate below.

This historical runner explicitly pins threshold `0` when
`--confidence-scheduler` is absent, preserving its fixed-K=5 control after the
runtime default changed.

The final promotion decision used the frozen math/chat generalization gate:

```sh
python3 speed-bench/run_dspark_generalization_gate.py \
  --dry-run --allow-dirty
python3 speed-bench/run_dspark_generalization_gate.py \
  --confirm-idle
```

The corpus pins 12 exact first-turn prompts from DeepSpec commit
`005e03b81cec38b7da6399833d609ee89a2587f2`: two each from GSM8K,
MATH-500, and AIME-2025, three long-form non-code Alpaca rows, and three
MT-Bench first turns spanning writing, reasoning, and humanities. Prompt bytes,
source-line hashes, original dataset hashes, source indices, and selection rules
live in `speed-bench/dspark-generalization/`. Selection is restricted to the
same per-dataset row caps declared by DeepSpec `eval.py`; no model output or
acceptance measurement was used to select rows.

Each task runs ordinary baseline, fixed K=5 DSpark, and threshold-`0.455`
DSpark once. Mode order rotates across tasks so all three modes occupy each
position exactly four times. Three excluded warmups plus 36 measured children
make 39 processes total. Every mode must produce byte-identical output, and no
instrumentation is enabled. MT-Bench contributes only its self-contained first
turn; this is a scheduler regression gate, not a matched benchmark-score run.
The fixed-K=5 child explicitly sets threshold `0`, so the historical control
remains reproducible after default promotion.

The promotion rule is frozen before measurement: scheduled/fixed median must
exceed `1.0x` in both math and chat, at least four of six tasks must improve in
each domain, overall scheduled/fixed geometric mean must be at least `1.03x`,
and no task may fall below `0.90x`. The scheduled/baseline ratio remains the
separate end-user result. This gate passed and `0.455` is now the DSpark runtime
default; do not tune it from this gate.

To attribute the promoted runtime without repeating that throughput gate, use
the frozen cross-domain diagnostic:

```sh
python3 speed-bench/run_dspark_generalization_attribution.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/dspark-generalization-20260715-190530/summary.json
python3 speed-bench/run_dspark_generalization_attribution.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/dspark-generalization-20260715-190530/summary.json
```

The supplied reference is validated before execution. The four tasks are
frozen as the lowest and highest scheduled/baseline result in each domain:
`math500_00166`, `gsm8k_00333`, `mt_bench_00075`, and `alpaca_00115`.
Each task runs one stats-enabled promoted-default DSpark process with no
threshold override, and its output must match the previously validated
scheduled output byte-for-byte. No fresh baseline, fixed-K process, acceptance
audit, trace, or layer profile is run.

The report omits diagnostic t/s and separates progress per proposal, proposal
rounds and target evaluations per emitted token, verified positions per target
evaluation, synchronized target cost, and sidecar cost. It also reports the
low/high amplification ratio within math and chat. Use those component shares
to choose the next optimization; do not combine synchronized component time
with the historical uninstrumented latency as though they came from one timing
boundary.

The older `--stats-pass` mode remains available when a single invocation should
run the full throughput comparison and then append one instrumented runtime
sample per prompt:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py \
  --confirm-ready --stats-pass
```

Every runtime and repeated baseline output must be byte-identical to that
prompt's first baseline output. Raw streams, paired rows, machine snapshots,
metadata, summaries, and optional stats go under the ignored
`speed-bench/local-runs/issue468-<timestamp>/` directory. Stats-only runs use
`speed-bench/local-runs/issue468-stats-<timestamp>/` and write `runs.csv`,
`stats.csv`, `summary.json`, and `summary.md`, but no `throughput.csv`.

The fixtures and SHA-256 provenance live in `speed-bench/issue468/`, along with
a frozen copy of the published MTP result table. The generated report compares
percentage changes from each implementation's own baseline. Absolute t/s is
not comparable across the two machines, and legacy MTP and DSpark are different
drafters; the cross-study columns are contextual relative improvements, not a
controlled head-to-head measurement.

`--fast-verifier` is retained for correctness research, but is not a valid
throughput setting for this corpus. Long-prompt tracing found and corrected two
compressed-attention policy mismatches, but the prefill-style batch verifier
still commits numerically approximate target/cache state. On `synthesis_8k`,
every shadow proposal matched exact row tops when started from an exact
frontier, yet fast authority accumulated enough state drift to change output.
Exact DSpark output matched baseline. The fast verifier must use numerically
exact batched decode semantics before its long-prompt speed can be reported;
confidence thresholds and larger correctness samples cannot prove approximate
state safe.

`--exact-head-batch` is a separate, correctness-oriented microbatch experiment.
It keeps every target layer and cache update on the exact verifier, batches only
the `n-1` intermediate output heads used for draft acceptance, and always runs
the final continuation head through the original serial exact path. It is
mutually exclusive with `--fast-verifier`:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py \
  --dry-run --exact-head-batch
```

The component passed byte-identity checks across all three long fixtures and
176 shadow comparisons of batched versus serial row tops. No performance claim
has been made for it yet. The first issue-468 run with this option compared it
against non-DSpark baseline, which reconfirmed that the exact verifier is slower
but did not isolate the output-head change.

Use the dedicated direct ablation to compare ordinary exact DSpark against the
same exact path with only intermediate output-head batching enabled:

```sh
python3 speed-bench/run_dspark_exact_head_ablation.py --dry-run
python3 speed-bench/run_dspark_exact_head_ablation.py --confirm-ready
```

The default run is one 64-token pair over `code_8k`, normally about one to two
minutes. Both modes enable stats, so their t/s values are diagnostic context,
not throughput results. The report instead splits target-verifier time into
exact layers, batched heads, serial heads, and residual overhead. It also
requires byte-identical output and reports successful head batches. Codex does
not run this command; the user starts it when the machine is ready.

The promoted exact prefix-checkpoint path avoids replaying an accepted partial
proposal after the exact verifier has already computed it. It snapshots the
small compressor/indexer frontier after each proposal row, retains the
corresponding exact logits and target hidden-state capture, and restores only
the accepted prefix. The raw SWA cache can leave later speculative rows as
invisible append-only data. Any unavailable checkpoint state restores the
original frontier and uses the existing exact replay path.

Run the byte-exact fixed-K correctness matrix first:

```sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_PREFIX_CHECKPOINT=default \
DS4_TEST_DSPARK_CONFIDENCE_THRESHOLD=0 \
  ./tests/dspark_gpu_candidates_correctness.sh
```

Then, when the machine is ready for a paired uninstrumented measurement:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-prefix-checkpoint-ablation \
  --prompt-file speed-bench/issue468/code_8k.txt \
  --ctx 16384 \
  --tokens 64 \
  --confirm-idle
```

The reference mode explicitly sets
`DS4_DSPARK_EXACT_PREFIX_CHECKPOINT=0`; the candidate is the ordinary promoted
exact runtime with no override. Both modes use the promoted confidence
scheduler, omit runtime stats and diagnostics, alternate order, and must
produce byte-identical output. Codex does not run this timed command.

To attribute checkpoint behavior without another throughput pass:

```sh
python3 speed-bench/run_dspark_humaneval_checkpoint_attribution.py \
  --dry-run --allow-dirty \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json
python3 speed-bench/run_dspark_humaneval_checkpoint_attribution.py \
  --confirm-ready \
  --throughput-reference \
  speed-bench/local-runs/humaneval-scheduler-throughput-32-20260716-103542/summary.json
```

The four frozen tasks cover low acceptance, the best current speed ratio, and
two of the largest checkpoint-era gains. The runner starts four stats-enabled
exact-runtime processes and reuses each prior runtime output as its byte-exact
reference. It reports checkpoint attempts, successful restores, replay
fallbacks, exact target rows whose replay was avoided, and a structural legacy
target-position proxy. It also reports confidence-selected scheduler width,
committed progress and sidecar time at each selected width, plus target
evaluation count and synchronized time at each actual verifier width. Sidecar
work consumed outside multi-commit or left after the final emitted token is
reported separately rather than charged to a scheduler width. Diagnostic t/s
is omitted, and the proxy and synchronized component timings are not speed
predictions. No fresh baseline, throughput pass, acceptance audit, trace, or
layer profiler is run.

Use the synchronized exact-runtime profile to attribute the remaining promoted
target-layer work:

```sh
python3 speed-bench/run_dspark_exact_layer_profile.py --dry-run
python3 speed-bench/run_dspark_exact_layer_profile.py --confirm-ready
```

The default profile inspects the model layer count, then runs one uninstrumented
exact reference followed by exact DSpark runs profiling the first, middle, and
last layer over 32 generated tokens. For V4 Flash those layers are 0, 21, and
42. Explicit `--layers` values are rejected before execution when they fall
outside the inspected model range. Every profiled output must match the
reference byte-for-byte.

At the selected layer, the diagnostic measures the three authoritative
post-promotion components: batched attention preparation, the unchanged serial
cache/attention tail across proposal rows, and the exact batched FFN. It fences
before the layer and after each component, so values include synchronization
overhead and change normal scheduling. One-token verifier calls are ignored;
every retained multi-row batch must have all three records with the same start
and width. The report normalizes each batch by its proposal rows before taking
component medians. Use the relative component ordering only to select an
implementation target; these are not additive production timings, throughput
results, or a speedup claim. Raw streams, component rows, metadata, and the
summary are written under ignored `speed-bench/local-runs/layer-profile-*`.
An interrupted run can reuse matching reference and layer files with
`--resume-dir`; the harness validates the retained command, prompt, context,
token count, and component contract before skipping completed layers.

The default-off exact attention row-view cache removes host-side object churn
without changing target arithmetic or Metal command order. The ordinary exact
tail creates temporary views over the same batch tensors for every proposal
row of every target layer. With
`DS4_DSPARK_EXACT_ATTN_ROW_VIEWS=1`, those fixed views are created once per
exact verifier call and reused through all 43 layers.

Run the byte-exact correctness matrix before timing:

```sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_ATTN_ROW_VIEWS=1 \
  ./tests/dspark_gpu_candidates_correctness.sh
```

Diagnostics require every multi-row verifier call to report a complete cache
with all expected layer-row uses and no fallback. Then, when the machine is
ready, run the paired uninstrumented ablation:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-row-views-ablation \
  --confirm-idle
```

The reference is ordinary exact DSpark; only the candidate sets
`DS4_DSPARK_EXACT_ATTN_ROW_VIEWS=1`. Runtime stats and diagnostics are disabled,
order alternates by pair, and every output must match byte-for-byte. Codex does
not run this timed command.

The promoted exact attention-output NR4 path targets the two retained serial
Q8 projections after attention. The legacy kernels compute two adjacent output
rows per threadgroup. The default projection A and fused projection B plus HC
expansion kernels compute four adjacent rows per threadgroup instead.

This changes activation reuse and threadgroup count only. Every output row
keeps the ordinary Q8 block traversal, scalar accumulation order, SIMD
reduction, threadgroup reduction, and HC expansion arithmetic. The candidate
is restricted to dimensions divisible by four and otherwise uses the existing
two-row kernels.

`DS4_DSPARK_EXACT_ATTN_OUT_NR4=0` restores the legacy NR2 kernels. Run both
byte-exact correctness controls before timing:

```sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR4=1 \
  ./tests/dspark_gpu_candidates_correctness.sh

DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR4=0 \
  ./tests/dspark_gpu_candidates_correctness.sh
```

The explicit NR4 correctness control enables a trace that must confirm both
projection A and projection B plus HC used their NR4 kernels. The trace flag
is not enabled by the throughput harness. When the machine is ready, run:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-output-nr4-ablation \
  --confirm-idle
```

The reference sets `DS4_DSPARK_EXACT_ATTN_OUT_NR4=0`; the candidate is ordinary
exact DSpark with the promoted NR4 default and no NR environment override.
Both modes are Metal-only, paired, uninstrumented, and must produce
byte-identical output. Codex does not run this timed command.

The next bounded attention-output experiment extends the same activation reuse
from four to eight adjacent output rows. NR8 is default-off and leaves promoted
NR4 unchanged unless `DS4_DSPARK_EXACT_ATTN_OUT_NR8=1` is set. Run the explicit
NR8 and ordinary NR4 correctness controls first:

```sh
DS4_TEST_DSPARK_MODE=runtime \
DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR8=1 \
  ./tests/dspark_gpu_candidates_correctness.sh

DS4_TEST_DSPARK_MODE=runtime \
  ./tests/dspark_gpu_candidates_correctness.sh
```

The NR8 control requires traces from both projection A and projection B plus
HC. Then, when the machine is ready, run the paired uninstrumented gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --exact-attention-output-nr8-ablation \
  --confirm-idle
```

The reference is ordinary promoted NR4. Only the candidate enables NR8.
Runtime stats and traces are disabled in both timed modes, order alternates by
pair, and every output must match byte-for-byte.

To attribute the retained one-row dense-mixed attention route before changing
its implementation, run:

```sh
python3 speed-bench/run_dspark_dense_mixed_flash_profile.py \
  --dry-run \
  --allow-dirty
python3 speed-bench/run_dspark_dense_mixed_flash_profile.py \
  --confirm-ready
```

This is a synchronized diagnostic, not a throughput benchmark. It runs the
8K transition fixture once as an uninstrumented reference and once with
layer-42 exact-attention plus gathered-FlashAttention boundaries. Every output
must match byte-for-byte. The report separates raw-ring linearization, raw and
compressed cache copies, mask work, padding, split-K attention, and the final
reduction for each dense-mixed row.

After the dense-mixed fused-gather candidate passes the byte-exact correctness
matrix, run its paired throughput gate:

```sh
python3 speed-bench/run_dspark_comparison.py \
  --dense-mixed-fused-gather-ablation \
  --dry-run \
  --allow-dirty
python3 speed-bench/run_dspark_comparison.py \
  --dense-mixed-fused-gather-ablation \
  --confirm-idle
```

The legacy reference uses separate raw-ring linearization, raw/compressed
conversion, mask fill, tail padding, FlashAttention, and reduction dispatches.
The promoted default fuses the preparation work into one kernel, then uses the
unchanged FlashAttention and reduction kernels. The explicit rollback switch
is `DS4_METAL_DENSE_MIXED_GATHERED_LEGACY=1`; the historical
`DS4_METAL_DENSE_MIXED_DIRECT=1` switch still forces fused gather. Both modes
use exact Metal verification with runtime stats, traces, and diagnostic
boundaries disabled, and every output must match byte-for-byte. The older
`--dense-mixed-direct-ablation` spelling remains an alias.

To confirm the candidate on the frozen threshold-0.75 HumanEval workload:

```sh
python3 speed-bench/run_dspark_humaneval_dense_mixed_direct.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --dry-run \
  --allow-dirty
python3 speed-bench/run_dspark_humaneval_dense_mixed_direct.py \
  --throughput-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-idle
```

This runs 32 gathered/fused-gather pairs with alternating order plus two
excluded warmup pairs. Both modes use threshold `0.75`; only the fused-gather
mode enables the candidate. All 68 processes are uninstrumented and must match
the frozen exact HumanEval outputs byte-for-byte. Run this broad gate only
after the short fused-gather ablation remains positive.

If that gate passes every aggregate criterion but fails only because
`humaneval_121` is the sole task below the frozen `0.95x` floor, adjudicate the
single pair with the predeclared replicated gate:

```sh
python3 speed-bench/run_dspark_humaneval_dense_mixed_outlier.py \
  --confirmation-reference \
  speed-bench/local-runs/humaneval-dense-mixed-direct-32-20260717-080040/summary.json \
  --dry-run \
  --allow-dirty
python3 speed-bench/run_dspark_humaneval_dense_mixed_outlier.py \
  --confirmation-reference \
  speed-bench/local-runs/humaneval-dense-mixed-direct-32-20260717-080040/summary.json \
  --confirm-idle
```

The adjudicator refuses any broad artifact that failed another criterion or
has a different set of sub-floor tasks. It runs six measured pairs with
exactly balanced order plus two excluded warmup pairs. Promotion requires a
median and geometric paired ratio of at least `1.02x`, at least four wins, and
at least five of six pairs at or above the original `0.95x` task floor. The
original 32-task result remains a formal failure; this run determines whether
its sole negative pair represents a reproducible task regression.

The six-pair adjudication passed on `humaneval_121`: fused gather won `6/6`,
with a `1.0959x` median paired ratio and `1.1078x` geometric mean. Together
with the broad gate's `1.0949x` geometric mean, `31/32` initial wins, and
byte-exact outputs, this promotes fused gather to the ordinary Metal
dense-mixed default. Use `DS4_METAL_DENSE_MIXED_GATHERED_LEGACY=1` only for
rollback or controlled comparisons.

To reassess end-to-end throughput after the fused-gather promotion, run the
same frozen threshold-`0.75` HumanEval workload against the ordinary target
baseline:

```sh
python3 speed-bench/run_dspark_humaneval_cumulative_throughput.py \
  --historical-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --dry-run \
  --allow-dirty
python3 speed-bench/run_dspark_humaneval_cumulative_throughput.py \
  --historical-reference \
  speed-bench/local-runs/humaneval-threshold075-throughput-32-20260716-155112/summary.json \
  --confirm-idle
```

The runner validates the old clean threshold-`0.75` artifact at commit
`7b954be`, including all 32 task pairs, hashes, exact outputs, model paths,
selection, and protocol. It then runs fresh baseline/current-runtime pairs
with alternating order and two excluded warmup pairs. Runtime explicitly pins
threshold `0.75` but otherwise uses promoted defaults: neither the fused-gather
force switch nor the legacy-gathered rollback switch is present. All current
outputs must match both ordinary baseline and the frozen historical output
byte-for-byte.

Current within-run paired ratios decide end-to-end performance. Cross-run
movement from the historical `0.8634x` geometric result is descriptive. The
predeclared movement gate requires at least `1.05x` geometric movement, at
least 24 improved tasks, and no task below `0.90x` movement. The independent
outcome bands are below near parity under `0.95x`, near parity from `0.95x` to
below `1.00x`, and parity or speedup at `1.00x` or above. No instrumentation is
enabled during this gate.

## Ordinary decode: pinned upstream main versus the current branch

Use this separate gate to determine whether the DSpark branch also changed
ordinary Metal decoding. It compares the current binary with a detached,
clean, pre-DSpark `origin/main` worktree pinned at
`80ebbc396aee40eedc1d829222f3362d10fa4c6c`. The prepared worktree is
`../ds4-master-baseline`, and its `ds4` binary must be built from that exact
commit. The runner refuses a different upstream commit or tracked changes in
either source tree.

The workload is the same deterministic 32-task HumanEval selection and frozen
outputs used by the latest cumulative study. Both modes are ordinary
non-thinking Metal decode: no sidecar, DSpark runtime, stats, traces,
diagnostics, or profiler. Every `DS4_*` environment variable is cleared, both
binaries must reproduce the frozen output byte-for-byte, measured order
alternates by task, and two balanced warmup pairs are excluded. The complete
run is 68 model processes. Each binary runs from its own source worktree
because ds4 compiles relative `metal/*.metal` sources at startup.

When the machine is ready, run:

```sh
python3 speed-bench/run_ds4_master_baseline_comparison.py \
  --output-reference \
  speed-bench/local-runs/humaneval-cumulative-throughput-32-20260719-171040/summary.json \
  --confirm-idle
```

The predeclared meaningful-progress gate requires a geometric current/upstream
paired ratio of at least `1.01x`, at least `24/32` tasks faster, and no task
below `0.95x`. The paired result answers whether this branch improved ordinary
decode; it must not be multiplied into a DSpark/baseline ratio from another
session.

## oMLX DeepSeek V4 comparison

This comparison pins oMLX commit
`a20d60de2e843395819969e61d8845d2497c49f0` and the
`Jundot/DeepSeek-V4-Flash-oQ2e-mtp` checkpoint at revision
`f42b63224cfed5cff40185004b77d7ff935a6c47`. The checkpoint is about
91.2 GiB and contains oMLX's native full MTP transformer block. It is not a
DSpark sidecar: this is an end-to-end engine and quantization comparison, not
an engine-only comparison against ds4's IQ2XXS GGUF.

The pinned source environment must include oMLX's optional native DeepSeek
expert kernels. A plain editable install silently uses the slower generic
fallback. The prepared checkout has already been built and ABI-checked. To
rebuild it after recreating the environment:

```sh
cd /Users/deathcodevision/dev/local-inference-lab/omlx
uv pip install --python .venv/bin/python \
  "cmake>=3.27" "nanobind==2.13.0" "setuptools>=61" wheel
PATH="$PWD/.venv/bin:$PATH" \
CMAKE_ARGS="-DPython_EXECUTABLE=$PWD/.venv/bin/python" \
OMLX_WITH_CUSTOM_KERNEL=1 \
  .venv/bin/python setup.py build_ext --inplace --force
.venv/bin/python -c \
  'from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())'
```

Download the model into the pinned oMLX environment when enough disk space is
available:

```sh
mkdir -p /Users/deathcodevision/dev/local-inference-lab/omlx-models/Jundot
/Users/deathcodevision/dev/local-inference-lab/omlx/.venv/bin/hf download \
  Jundot/DeepSeek-V4-Flash-oQ2e-mtp \
  --revision f42b63224cfed5cff40185004b77d7ff935a6c47 \
  --local-dir /Users/deathcodevision/dev/local-inference-lab/omlx-models/Jundot/DeepSeek-V4-Flash-oQ2e-mtp
```

After the download exits successfully, validate the architecture, native MTP
tensor directory, and presence of every indexed shard without loading the
model:

```sh
/Users/deathcodevision/dev/local-inference-lab/omlx/.venv/bin/python \
  speed-bench/run_omlx_humaneval_mode.py \
  --mode baseline \
  --validate-only
```

The initial sweep uses eight deterministic HumanEval tasks. Each process runs
one excluded warmup, then the same eight measured tasks with caches disabled,
one active request, non-thinking chat, greedy decoding, and 128 output tokens.
Run each mode only while the machine is as idle as practical:

```sh
OMLX_PY=/Users/deathcodevision/dev/local-inference-lab/omlx/.venv/bin/python

$OMLX_PY speed-bench/run_omlx_humaneval_mode.py --mode baseline --confirm-idle
$OMLX_PY speed-bench/run_omlx_humaneval_mode.py --mode mtp1 --confirm-idle
$OMLX_PY speed-bench/run_omlx_humaneval_mode.py --mode mtp2 --confirm-idle
$OMLX_PY speed-bench/run_omlx_humaneval_mode.py --mode mtp3 --confirm-idle
```

`mtp1` is fixed depth one. `mtp2` and `mtp3` set adaptive maximum depths two
and three; oMLX's online controller may choose a shallower depth per cycle.
The first sweep is diagnostic because each mode occupies one process-order
position. Analyze it by passing the four generated `summary.json` paths:

```sh
python3 speed-bench/analyze_omlx_humaneval_sweep.py \
  --baseline /path/to/baseline/summary.json \
  --mtp1 /path/to/mtp1/summary.json \
  --mtp2 /path/to/mtp2/summary.json \
  --mtp3 /path/to/mtp3/summary.json
```

Every speculative mode must match oMLX baseline byte-for-byte within each
task. Across oMLX and ds4, compare paired throughput and token counts rather
than output bytes because the engines use different target quantizations and
model formats. Confirm the winning oMLX mode with balanced process ordering
before making a performance claim or selecting a mechanism to port.
