#include <metal_stdlib>
using namespace metal;

struct ds4_metal_args_dsv4_indexer_scores_fused {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
};

kernel void kernel_dsv4_indexer_scores_tiled(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    // Q/K are staged as half but the dot accumulator and final score remain
    // float. This is the one intentional precision tradeoff in the indexer:
    // the indexer only ranks compressed rows for top-k selection, and long
    // context profiling shows this score matrix dominates the prefill slope.
    threadgroup half *qtg = (threadgroup half *)shared; // [8][128]
    threadgroup half *ktg = qtg + TM*D;                 // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    // Stage compressed index rows once. Edge columns are zeroed so the matrix
    // loads below can stay regular; guarded stores discard them.
    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = half(row[d]);
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        // Stage Q for the eight-token tile. Each 8x8 matrix load below reads a
        // contiguous depth block from this layout.
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            half v = half(0.0f);
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}
