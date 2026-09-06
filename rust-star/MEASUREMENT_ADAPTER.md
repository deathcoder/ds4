# Engine Measurement Adapters

Schema: `rust-star-engine-measurement-v1`.

The engine-neutral form of this schema is defined in
`ENGINE_MEASUREMENT_FORMAT.md`.

`measure_dwarfstar.py` and `measure_ruststar.py` are the two engine-side
boundaries for the paired runner. Each invocation launches exactly one fresh
engine process for one context frontier and writes a self-contained evidence
directory.

The adapters do not orchestrate A/B ordering or retries. The paired runner
invokes them through equivalent fresh-process boundaries.

## Sampling-inclusive oracle timing

The DwarfStar CSV must declare `sampling_in_interval=1`. The corrected driver
charges the initial argmax to prefill, then measures each decode evaluation
through selection and commitment of its output token. First/steady timing and
complete generation therefore include sampling, as required by the protocol.
Legacy `ds4-bench` steady timers excluded argmax and are rejected by this adapter;
retain old results as historical measurements under their original semantics.

Build the accepted oracle-v4 source, which pins both the top-k synchronization
repair and the corrected benchmark driver:

```sh
python3 rust-star/build_benchmark_oracle.py \
  --oracle-version oracle-v4 \
  --output rust-star/.work/sampling-inclusive-oracle
```

The builder records original/overlaid source hashes and the executable hash.
Pin that provenance and the oracle-v4 source commit in the paired plan. The
default is v4; `--oracle-version oracle-v3` remains an explicit historical
reproduction option. The builder never replaces an existing oracle binary.
Before timing a new driver, `--dump-frontier-logits-dir` can capture prefill and
every decoded full-logit row for exact comparison; these diagnostic runs are
not performance samples. Normal adapter commands never enable dumping.

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

The native producer supports the validated 2K and 32K frontiers. It runs
prefill without decoding diagnostic output
fixtures, allocating host boundary tensors, or copying transformer boundaries
to the host. It then times a 128-token loop without generation tensor
collection and checks each selection after its timed interval. A failed
selection triggers a readback of the same completed logits for failure details;
failed runs cannot produce an eligible measurement. Passing runs keep the
eight-byte GPU selection readback. The independent full-output C0 command
checks production GPU argmax against the same logits and remains mandatory. The adapter accepts
the raw record only when both collection flags are false, the transcript is
exact, and the declared generation schedule matches the protocol.
