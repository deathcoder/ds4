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

All inherited DSpark and timing/logging variables are cleared. Throughput runs
enable only the GPU runtime and multi-commit, so no runtime statistics enter
their medians. Add `--stats-pass` to run one separate
instrumented runtime sample per prompt after all throughput measurements:

```sh
python3 speed-bench/run_dspark_issue468_comparison.py \
  --confirm-ready --stats-pass
```

Every runtime and repeated baseline output must be byte-identical to that
prompt's first baseline output. Raw streams, paired rows, machine snapshots,
metadata, summaries, and optional stats go under the ignored
`speed-bench/local-runs/issue468-<timestamp>/` directory.

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
