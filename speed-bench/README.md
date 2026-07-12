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

`--fast-verifier` additionally sets
`DS4_DSPARK_FAST_BATCH_VERIFY=1`. This is deliberately opt-in: it makes the
compute-batched verifier authoritative for suffix tops and continuation logits,
while exact verification remains the fallback after recoverable fast-path
failure. Partial accepts restore the target frontier and rerun only the accepted
prefix with the same verifier. The harness still requires byte-identical output
against baseline and aborts immediately if numerical drift changes the stream.
Fast authority is currently limited to the first synchronized prompt of a
session. A resumed sync permanently suspends it for that session and uses exact
batch verification, because broader soak testing found a second-turn numerical
divergence while one-shot output remained identical.

All inherited `DS4_DSPARK_*` variables and instrumentation variables containing
`PROFILE`, `TRACE`, `DUMP`, or `TIMING` (plus `*_LOG`) are removed first. Runtime
diagnostic logs remain disabled. The stats option adds only clock reads and one
machine-readable record when the session closes; it reports acceptance depth,
target graph calls and token positions, exact-batch verifier outcomes, and
bridge/stage/head/chain timing split between prefill and generation. Other
`DS4_*` tuning variables are preserved and recorded. Every run must produce
byte-identical stdout; the harness aborts on drift. Raw stdout/stderr,
environment metadata, process and thermal snapshots, per-run CSV data, and
median/paired-speedup plus runtime-efficiency summaries go under the ignored
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
```

It compares baseline and fast output across 64-token multi-cycle generation,
code fallback, Italian, Spanish, structured JSON, near-window context, strict
margin gating, and a resumed two-turn session. It requires fast commits where eligible,
rejects verifier/capture failures, and verifies the resumed fast-to-exact
transition.

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
runtime uses the fast authoritative DSpark verifier by default; pass
`--exact-verifier` only for a separate exact-verifier study.

All inherited DSpark and timing/logging variables are cleared. Throughput runs
enable only the GPU runtime, multi-commit, and fast verifier, so no runtime
statistics enter their medians. Add `--stats-pass` to run one separate
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
