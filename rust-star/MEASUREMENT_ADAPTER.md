# Engine Measurement Adapters

Schema: `rust-star-engine-measurement-v1`.

The engine-neutral form of this schema is defined in
`ENGINE_MEASUREMENT_FORMAT.md`.

`measure_dwarfstar.py` and `measure_ruststar.py` are the two engine-side
boundaries for the paired runner. Each invocation launches exactly one fresh
engine process for one context frontier and writes a self-contained evidence
directory.

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

The adapters record externally measured complete-process wall time and peak
resident memory. Prefill/generation intervals are derived from DwarfStar's
reported token counts and rates. `process_overhead_ms` is the residual outside
those intervals; it is not labeled as pure model-load time.

At a terminal single frontier DwarfStar does not create the session snapshot
whose serialized size populates `kvcache_bytes`. The adapter records `null` in
that case rather than claiming that live KV memory is zero.

Run either adapter as a fresh Python process for every measurement. Peak-RSS
collection relies on having exactly one child process in that adapter process.
The paired runner enforces this automatically.

Rust Star exposes its initial eligible boundary as:

```sh
python3 rust-star/measure_ruststar.py \
  --executable /path/to/rust-star \
  --model /path/to/model.gguf \
  --context 2048 \
  --gen-tokens 128 \
  --output /path/to/results/rust-star-ctx-2048-run-01
```

The native producer executes its 2K prefill without decoding diagnostic output
fixtures, allocating host boundary tensors, or copying transformer boundaries
to the host. It then times a 128-token loop without generation tensor
collection and checks the selected-token transcript afterward. The independent
C0 prefill command remains the mandatory regression gate. The adapter accepts
the raw record only when both collection flags are false, the transcript is
exact, and the declared generation schedule matches the protocol.
