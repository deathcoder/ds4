# Rust Star Engine Measurement Contract

Schema: `rust-star-engine-measurement-v1`.

This is the shared process boundary between the checkpointed paired runner and
any inference engine. Both engine adapters implement it. Rust Star first writes
the narrower `rust-star-engine-run-v1` record; `measure_ruststar.py`
independently validates that record and adds complete-process wall time, peak
RSS, sanitized logs, and checksummed artifacts.

## Invocation boundary

The paired plan provides an argument array containing these full-argument
placeholders exactly once:

- `{context}`: the exact prefill frontier;
- `{gen_tokens}`: committed greedy tokens to generate; and
- `{output}`: a new engine-evidence directory.

The runner substitutes those arguments without a shell and launches the adapter
in a fresh process group. A successful adapter exits zero and writes
`{output}/measurement.json`. It may write additional logs or artifacts beneath
that directory, but must not write outside its declared result directory except
for normal engine caches explicitly recorded in runtime configuration.

## Successful measurement

```json
{
  "schema": "rust-star-engine-measurement-v1",
  "engine": "rust-star",
  "status": "passed",
  "context": 262144,
  "gen_tokens": 128,
  "metrics": {
    "ctx_tokens": 262144,
    "prefill_tokens": 262144,
    "gen_tokens": 128,
    "gen_steady_tokens": 127,
    "kvcache_bytes": null,
    "process_peak_bytes": 1,
    "prefill_tps": 1.0,
    "prefill_ms": 1.0,
    "gen_tps": 1.0,
    "gen_ms": 1.0,
    "gen_first_ms": 1.0,
    "gen_steady_tps": 1.0,
    "gen_steady_ms": 1.0,
    "process_wall_ms": 1.0,
    "process_overhead_ms": 0.0
  }
}
```

The example values show types, not realistic timing consistency. Timers use
milliseconds and rates use committed tokens per second. `gen_first_ms` is the
first decode step. `gen_steady_*` covers the remaining `gen_tokens - 1` steps.
`process_wall_ms` spans the complete adapter/engine invocation.

`kvcache_bytes` may be `null` only when the engine cannot expose a comparable
measurement. Any non-null value must state its semantics in the engine runtime
configuration. Optional nonnegative `model_load_ms` is allowed only when backed
by a real engine timer.

For Rust Star, device work must be synchronized before stopping each interval.
Command encoding, model execution, GPU argmax sampling, selection readback, and
committing the token belong inside the generation intervals. Rust Star's timed
path reads back an eight-byte top-1 result instead of the complete logit row;
the independent C0 controls retain full-logit readback and comparison.
Correctness dumping and profiler capture must remain disabled in timed mode.

The Rust Star adapter accepts an engine run only when its raw timing metadata
also states that prefill and generation correctness collection were disabled,
the exact oracle transcript matched after timing, and the expected 44 command
buffers plus two host waits were used per generated token. An engine may emit
an ineligible raw record with a blocker for development, but the adapter must
turn it into a failed measurement rather than forwarding its rates.

## Failure measurement

An adapter should preserve local evidence, write `status: "failed"` with a
short non-sensitive diagnostic when possible, and exit nonzero. The paired
runner records the attempt but does not retry it automatically. Capacity and
engine failures remain blocked evidence; only a deliberately documented
external-event invalidation permits a full-pair retry.

Do not put credentials, inherited environment dumps, private absolute paths, or
hardware serial/UUID values in measurement artifacts.
