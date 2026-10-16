Next candidate after profile audit: native attention-output low projection with
existing dense Q8 matmul, eight aligned group dispatches and direct strided stores.

Observed profile: attention_output 36.173 attributed GPU seconds (36.75% of native
tail), routed_gate_up 34.285, routed_down 19.267. Routed map only 0.332; defer it.

Current grouped low kernel shares the dense kernel's 64x32x32 tile geometry,
Q8 dequantization, half input staging, ordered FP32 SIMD matrix accumulations.
Its output epilogue stores to threadgroup memory, waits, then scatters using ids.
All eight output groups are dense with known token-major row order. Dense matmul
can use the physical output row pitch 8192 in ne0, but dispatch only the aligned
1024-column group width (16 Y threadgroups). Each group binds src/dst offsets.

Address proof for group g (0..8), row t (0..4096), projected channel c (0..1024):
- input FP32 index = t*32768 + g*4096 + k, k in 0..4096;
- weight byte offset = g*1024*(4096/32*34) + c*(4096/32*34);
- destination FP32 index = t*8192 + g*1024 + c.
- dense kernel dispatch (128,16,1), 128 threads, shared bytes6144;
  ne1=4096, ne0=8192 physical pitch, ne00=4096, ne12=1, r2=r3=1.
- all tiles are complete, so using physical pitch for ne0 cannot admit any
  out-of-range source row: explicit Y extent limits c to1023, bounds checks off.
- destination slices are disjoint and cover exactly all 4096*8192 FP32 elements.

Native low stage changes map+one grouped dispatch into eight dense dispatches,
net +6 tail dispatches: 25->31, odd total47+padding->53+padding. No weight binding
count changes. Legacy controls keep their map/grouped kernel. Group ids/map
scratch and host initialization can be omitted for native calls.

Do not assume bit equality from the source proof: require six live gates and
measured paired performance. Freeze candidate before any live run. No kernel
source, quantization, oracle, or production SSD/CUDA source change is needed.
