# DSpark Dense-Mixed Decode Contract

This document freezes the byte-exact input contract of the promoted Metal
dense-mixed fused-gather path. It is the reference for any candidate that
removes `kernel_dsv4_dense_mixed_prepare_f16` and reads the raw and compressed
caches directly.

## Eligibility

The promoted path is selected only when all of these conditions hold:

- compressed rows are present;
- no compressed-row mask is active;
- `head_dim == 512`;
- inverse RoPE is not requested;
- the fused-gather route has not been explicitly rolled back.

The enclosing API additionally requires at least one raw row, a valid raw-ring
start, `raw_cap >= n_raw`, and `n_raw + n_comp <= 8192`.

## Logical Rows

Let:

```text
n_keys = n_raw + n_comp
n_rows = ceil(n_keys / 32) * 32
```

The prepared cache is row-major F16 with shape `[n_rows, 512]`. Logical row
`r` is defined as follows:

| row range | source | source row | F16 value |
|---|---|---|---|
| `0 <= r < n_raw` | FP32 raw ring | `(raw_start + r) % raw_cap` | `(half)raw[col]` |
| `n_raw <= r < n_keys` | compressed cache | `r - n_raw` | F16 load, or `(half)` FP32 load |
| `n_keys <= r < n_rows` | padding | none | `0.0h` |

Tensor offsets are applied when Metal buffers are bound. The row strides seen
by the prepare kernel are `512 * sizeof(float)` for raw rows and either
`512 * sizeof(half)` or `512 * sizeof(float)` for compressed rows.

Raw row zero is the oldest retained row, not necessarily physical ring row
zero. Compressed rows are appended after every raw row and retain their stored
order. The same prepared F16 buffer is bound as both K and V.

## Conversion Boundary

Every FP32 raw or compressed scalar is converted with Metal's ordinary
`(half)` cast before FlashAttention consumes it. A direct reader must not feed
the original FP32 value into the dot product and cast later. F16 compressed
rows are loaded as F16 without an intervening FP32 computation.

## Mask And Padding

The prepared mask has one F16 value per logical row:

```text
mask[r] = 0.0h       when r < n_keys
mask[r] = -MAXHALF   when n_keys <= r < n_rows
```

The vector kernel receives `ne11 = n_rows`, `has_mask = true`, and
`has_kvpad = false`. Padding therefore remains inside the ordinary 32-row
chunk schedule and is skipped by the mask test. A direct reader must preserve
that schedule; shortening `ne11` to `n_keys` or using the generic tail-pad path
would change the execution contract.

## FlashAttention Shape

- Query: FP32 `[n_head, 512]`.
- K and V: the same logical F16 `[n_rows, 512]` cache.
- Cache rows are shared by all query heads.
- Scale: `1 / sqrt(512)`.
- Per-head sinks are enabled.
- Bias and logit softcap are disabled.
- Split-K chunk width is 32 rows; production uses `NWG=32`.
- The promoted split-K reduction and its partial layout are authoritative.

## Split-Source Candidate Rules

A prepare-free candidate may change only where each F16 K/V scalar and mask
scalar are loaded. It must retain:

1. the logical-row mapping above;
2. the F16 conversion boundary;
3. identical K and V values;
4. `n_rows` chunk traversal and padded-row masks;
5. query loading, chunk assignment, softmax partial arithmetic, sink handling,
   temporary layout, and final reduction order.

The previously rejected sequential direct-attention kernel is not a valid
starting point because it changed the attention reduction arithmetic. The
candidate should instead specialize the current split-K vector kernel's source
loads while leaving its arithmetic body and reduction kernel unchanged.

## Required Gates

Before any throughput run:

- synthetic contract tests must cover raw-ring wraparound, F16 and FP32
  compressed rows, exact multiples of 32, and padded tails;
- the runtime correctness matrix must engage the candidate and match every
  reference byte-for-byte;
- a dedicated parity diagnostic should compare prepared K/V and masks against
  the candidate's logical source view before target output is considered.

Only after those gates pass should a three-pair uninstrumented ablation be
prepared for user execution.
