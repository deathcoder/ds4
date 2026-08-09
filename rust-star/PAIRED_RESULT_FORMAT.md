# Rust Star Paired Result Format

Schemas: `rust-star-paired-raw-v1` and `rust-star-paired-summary-v1`.

The raw JSON file is the interchange boundary between benchmark runners and the
offline paired aggregator. It records every attempted oracle/candidate pair.
The runner may be Python, Rust, or another narrow host tool; changing the runner
must not change this schema silently.

Required top-level fields:

- `schema`: `rust-star-paired-raw-v1`;
- `protocol`: `rust-star-paired-benchmark-v1`;
- `correctness_class`: one of `C0`, `C1`, `C2`, or `C3`;
- `host_manifest_sha256`: the privacy-filtered target-host manifest used for
  the complete timed block;
- `correctness_manifest_sha256`: the manifest that supports the declared
  correctness classification for these exact engine/model configurations;
- `configuration`: contexts, predeclared repetitions, generation length,
  sampling rule, and primary metric;
- `oracle` and `candidate`: source commit and tree, executable/model/prompt
  SHA-256 values, backend, and explicit build/runtime configuration objects;
  and
- `pairs`: every predeclared context/repetition pair, including invalid pairs.

Each pair contains:

- `context`, `repetition`, positive `attempt`, and scheduled `order` (`AB` on
  odd repetitions, `BA` on even repetitions);
- `valid` and, when false, a non-empty `invalid_reason`; and
- DwarfStar-compatible `oracle` and `candidate` metric rows when valid.

Attempts for a context/repetition are numbered contiguously from one. Earlier
attempts may be invalid, but the final attempt must be the single valid pair.
This preserves external-event failures while still completing every
predeclared measurement. First attempts are serialized in execution order:
repetition-major, with contexts ascending on odd repetitions and descending on
even repetitions. Retry attempts remain in their actual chronological position.

Build and runtime configuration objects must be explicit allowlists. They must
not contain inherited environment dumps, credentials, absolute private paths,
serial numbers, or other unrelated machine identity.

The model and prompt SHA-256 must match between engines. A valid metric row
contains:

- integer `ctx_tokens`, `prefill_tokens`, `gen_tokens`, `gen_steady_tokens`,
  `kvcache_bytes`, and `process_peak_bytes`;
- positive finite `prefill_tps`, `prefill_ms`, `gen_tps`, `gen_ms`,
  `gen_first_ms`, `gen_steady_tps`, `gen_steady_ms`, and `process_wall_ms`; and
- nonnegative finite `model_load_ms`.

`summarize_paired_benchmark.py` rejects duplicate/missing pairs, identity drift,
wrong ordering, inconsistent token counts, and non-finite metrics. Invalid
pairs are retained in the summary but excluded from distributions.

For each context, the summary reports raw-engine median/MAD/min/max and
candidate/oracle ratios computed within each valid pair. This prevents a ratio
from being constructed out of unrelated favorable runs.

Run:

```sh
python3 rust-star/summarize_paired_benchmark.py raw.json \
  --json summary.json
```

Only a `C0` file is marked `headline_eligible`; experimental classifications
remain aggregatable but explicitly ineligible for the default-path headline.
