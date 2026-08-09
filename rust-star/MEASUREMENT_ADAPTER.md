# DwarfStar Measurement Adapter

Schema: `rust-star-engine-measurement-v1`.

`measure_dwarfstar.py` is the first engine-side boundary for the future paired
runner. Each invocation launches exactly one fresh `ds4-bench` process for one
context frontier and writes a self-contained evidence directory.

It deliberately does not orchestrate A/B ordering or retries yet. Keeping the
single-engine adapter separate lets the paired runner eventually invoke the
pinned DwarfStar oracle and Rust Star through equivalent process boundaries.

## Usage

```sh
python3 rust-star/measure_dwarfstar.py \
  --executable /path/to/pinned-dwarfstar/ds4-bench \
  --model /path/to/model.gguf \
  --prompt /path/to/rust_star_oracle_prompt.txt \
  --context 262144 \
  --gen-tokens 128 \
  --output /path/to/results/oracle-ctx-262144-run-01
```

The output directory must be new or empty. It receives:

- `measurement.json`: normalized metrics and checksummed artifact references;
- `benchmark.csv`: the original DwarfStar row;
- `stdout.log` and `stderr.log`: logs with the executable, model, prompt, and
  temporary paths replaced by stable placeholders.

The adapter records externally measured complete-process wall time and peak
resident memory. Prefill/generation intervals are derived from DwarfStar's
reported token counts and rates. `process_overhead_ms` is the residual outside
those intervals; it is not labeled as pure model-load time.

At a terminal single frontier DwarfStar does not create the session snapshot
whose serialized size populates `kvcache_bytes`. The adapter records `null` in
that case rather than claiming that live KV memory is zero.

Run this adapter as a fresh Python process for every measurement. Its peak-RSS
collection relies on having exactly one child process in that adapter process.
The later paired runner will enforce this automatically.
