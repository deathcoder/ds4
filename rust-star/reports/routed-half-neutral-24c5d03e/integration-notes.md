# Routed half-input integration notes (preparation only)

Do not apply before local exactness and selection plus a new frozen integration
plan. Current runtime and whole-engine block remain unchanged.

- Existing production tail uses `native_attention_half` for nonzero native
  continuation chunks. It already allocates 256 MiB heads-half scratch at
  4K rows under `prefill_transient_tail_attention_heads_half`.
- Its final reader is the grouped attention-output projection. Later FFN
  ingress/router/map steps do not read that buffer. A routed input conversion
  needs only 4096 * 4096 * 2 = 32 MiB after those readers complete.
- Reuse `heads_half_buffer` only under the same nonzero native guard and an
  explicit capacity/ownership check. Keep the existing prefill transient
  cleanup before decode. No new scratch allocation is needed for this scope.
- Insert conversion after the routed-map/profile-boundary-4 marker and before
  routed gate/up, so the conversion belongs to the reported gate/up cost.
  The prepared isolated chain uses that same map/conversion/gate order.
- New kernel keeps native output-tile-first grid (32, workcap, 1), 128 threads
  and 16384-byte threadgroup memory. Only its input byte strides are halved;
  routing, quantized weights, activation arguments and output strides stay.
- A new lazily built pipeline plus appended Metal source is needed; imported
  kernels, bootstrap, prefix and decode paths must restore byte-for-byte.
- Full normal 32K/128 C0, repeated checks sensitive to prefill-state handoff,
  strict profile coverage, fixed paired engine/decode/RSS acceptance and
  separately reported peak memory footprint remain required for retention.
