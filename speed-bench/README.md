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
```

The runner deliberately refuses real execution without `--confirm-idle`. Treat
that flag as confirmation that you have made the machine as quiet as practical;
the captured process and thermal metadata records unavoidable interference. Its
defaults perform one warmup per mode followed by three measured pairs. Pair
order alternates baseline/runtime then runtime/baseline, with a 10-second
cooldown between processes. Each process uses greedy generation, a fixed seed,
64 generated tokens, and `speed-bench/dspark_prompt.txt`; the rendered prompt
plus generation stays inside the current 128-token DSpark sidecar window.

Runtime runs set only:

```text
DS4_DSPARK_GPU_RUNTIME=1
DS4_DSPARK_MULTI_COMMIT=1
DS4_DSPARK_GPU_RUNTIME_STATS=1
```

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

Useful explicit overrides include `--pairs`, `--warmups`, `--cooldown`,
`--tokens`, and `--output-dir`. Do not compare runs with different settings.
The CLI timing line excludes model loading, while the recorded wall time does
not; generation t/s from the timing line is the comparison metric.
