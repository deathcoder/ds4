#include <metal_stdlib>
using namespace metal;
struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
};

kernel void kernel_topk_stream512(
        constant ds4_metal_args_argsort & args [[buffer(0)]],
        device const float *scores [[buffer(1)]],
        device uint32_t *selected [[buffer(2)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    constexpr uint stream_cap = 1024u;
    constexpr uint keep_count = 512u;
    threadgroup uint64_t keys[stream_cap];
    threadgroup atomic_uint count;
    threadgroup uint64_t threshold;

    if (token >= (uint)args.ne01) return;
    if (tid == 0u) {
        atomic_store_explicit(&count, 0u, memory_order_relaxed);
        threshold = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float *row = scores + (uint64_t)token * (uint)args.ne00;
    const uint n_comp = (uint)args.ne00;
    const uint start = (uint)(((uint64_t)(token + 1u) * 0x9E3779B9u) % n_comp);

    for (uint base = 0u; base < n_comp; base += ntg) {
        const uint i = base + tid;
        if (i < n_comp) {
            uint col = start + i;
            if (col >= n_comp) col -= n_comp;
            const uint bits = as_type<uint>(row[col]);
            const uint ordered = (bits & 0x80000000u) ? ~bits :
                                                          (bits ^ 0x80000000u);
            const uint64_t key = ((uint64_t)ordered << 32u) |
                                 (uint64_t)(0xffffffffu - col);
            if (key > threshold) {
                const uint pos = atomic_fetch_add_explicit(
                    &count, 1u, memory_order_relaxed);
                keys[pos] = key;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Test-only scheduling skew: delay half the SIMD groups before they
        // sample the shared count. This does not change the selection values.
        if ((tid / 32u) % 2u != 0u) {
            volatile uint delay = tid;
            for (uint k = 0; k < 128u; ++k) delay = delay*1664525u+1013904223u;
        }
        const uint used = atomic_load_explicit(&count, memory_order_relaxed);
        if (used > stream_cap - ntg) {
            for (uint j = used + tid; j < stream_cap; j += ntg) keys[j] = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k = 2u; k <= stream_cap; k <<= 1u) {
                for (uint j = k >> 1u; j > 0u; j >>= 1u) {
                    for (uint p = tid; p < stream_cap; p += ntg) {
                        const uint other = p ^ j;
                        if (other > p) {
                            const uint64_t a = keys[p];
                            const uint64_t b = keys[other];
                            const bool descending = (p & k) == 0u;
                            if ((descending && a < b) || (!descending && a > b)) {
                                keys[p] = b;
                                keys[other] = a;
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            if (tid == 0u) {
                threshold = keys[keep_count - 1u];
                atomic_store_explicit(&count, keep_count, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint used = atomic_load_explicit(&count, memory_order_relaxed);
    for (uint j = used + tid; j < stream_cap; j += ntg) keys[j] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2u; k <= stream_cap; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint p = tid; p < stream_cap; p += ntg) {
                const uint other = p ^ j;
                if (other > p) {
                    const uint64_t a = keys[p];
                    const uint64_t b = keys[other];
                    const bool descending = (p & k) == 0u;
                    if ((descending && a < b) || (!descending && a > b)) {
                        keys[p] = b;
                        keys[other] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    device uint32_t *out = selected + (uint64_t)token * keep_count;
    for (uint i = tid; i < keep_count; i += ntg) {
        out[i] = 0xffffffffu - (uint)keys[i];
    }
}
