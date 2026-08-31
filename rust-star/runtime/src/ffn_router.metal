#include <metal_stdlib>
using namespace metal;

// Exact one-token Flash router kernels imported from DwarfStar's
// metal/unary.metal and metal/dsv4_misc.metal.
struct rust_star_unary_args {
    int ne00, ne01, ne02, ne03;
    ulong nb00, nb01, nb02, nb03;
    int ne0, ne1, ne2, ne3;
    ulong nb0, nb1, nb2, nb3;
    float slope, scale, bias, val, min, max;
};

kernel void kernel_dsv4_softplus_sqrt_f32_4(
        constant rust_star_unary_args &args,
        device const char *src,
        device char *dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int k0 = tgpig.x/args.ne01;
    const int i01 = tgpig.x - k0*args.ne01;
    const int i0 = k0*ntg.x + tpitg.x;
    if (i0 >= args.ne0) return;
    device const float4 *s = (device const float4 *)(src + i01*args.nb01);
    device float4 *d = (device float4 *)(dst + i01*args.nb1);
    const float4 x = s[i0];
    const float4 sp = select(log(1.0f + exp(x)), x, x > 20.0f);
    d[i0] = sqrt(sp);
}

struct rust_star_router_select_one_args {
    uint has_bias;
    uint hash_mode;
    uint use_token_buffer;
    uint token;
    uint hash_rows;
};

kernel void kernel_dsv4_router_finalize_one(
        constant rust_star_router_select_one_args &args,
        device const float *probs,
        device const float *bias,
        device const int *hash,
        device const int *tokens,
        device int *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256) return;
    threadgroup float *sel_scores = scratch;
    threadgroup int *idx = (threadgroup int *)(scratch + 256);
    const float p = probs[tid];
    sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    idx[tid] = (int)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            device const int *src = hash + row*6u;
            for (uint i = 0; i < 6; i++) selected[i] = src[i];
        }
    } else {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int tmp = idx[tid]; idx[tid] = idx[other]; idx[other] = tmp;
                        }
                    } else if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                        const int tmp = idx[tid]; idx[tid] = idx[other]; idx[other] = tmp;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid < 6) selected[tid] = idx[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void kernel_dsv4_router_weights_one(
        device const char *probs,
        device const char *selected,
        device char *weights,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= 6) return;
    device const float *p = (device const float *)probs;
    device const int *s = (device const int *)selected;
    float sum = 0.0f;
    for (uint i = 0; i < 6; i++) sum += p[s[i]];
    sum = max(sum, 6.103515625e-5f);
    device float *w = (device float *)weights;
    w[tid] = p[s[tid]]/sum*1.5f;
}

// Exact decomposed M1 batch-router schedule. DwarfStar's generic batch path
// stores each stage separately; keeping these as individual kernels preserves
// the same FP32 rounding boundaries before routed-expert execution.
// The long-prefill bootstrap processes one aligned 4K chunk. Keep every
// decomposed router stage live across the full chunk rather than silently
// leaving rows 2048..4095 untouched.
constant uint rust_star_router_max_batch_rows = 4096u;

kernel void rust_star_router_softplus_batch(
        device const float *src,
        device float *dst,
        uint index [[thread_position_in_grid]]) {
    if (index >= rust_star_router_max_batch_rows*256u) return;
    const float x = src[index];
    dst[index] = select(log(1.0f + exp(x)), x, x > 20.0f);
}

kernel void rust_star_router_sqrt_batch(
        device const float *src,
        device float *dst,
        uint index [[thread_position_in_grid]]) {
    if (index >= rust_star_router_max_batch_rows*256u) return;
    dst[index] = sqrt(src[index]);
}

kernel void rust_star_router_hash_rows_batch(
        device const int *hash,
        device const uint *tokens,
        device int *selected,
        uint row [[thread_position_in_grid]]) {
    if (row >= rust_star_router_max_batch_rows) return;
    const uint token = min(tokens[row], 129279u);
    for (uint slot = 0; slot < 6u; slot++) {
        selected[row*6u + slot] = hash[token*6u + slot];
    }
}

kernel void rust_star_router_topk_rows_batch(
        device const float *probs,
        device const float *bias,
        device int *selected,
        uint row [[thread_position_in_grid]]) {
    if (row >= rust_star_router_max_batch_rows) return;
    int top[6] = {-1, -1, -1, -1, -1, -1};
    device const float *row_probs = probs + row*256u;
    for (uint expert = 0; expert < 256u; expert++) {
        const float score = row_probs[expert] + bias[expert];
        for (uint slot = 0; slot < 6u; slot++) {
            if (top[slot] < 0 ||
                score > row_probs[(uint)top[slot]] + bias[(uint)top[slot]]) {
                for (uint move = 5u; move > slot; move--) {
                    top[move] = top[move - 1u];
                }
                top[slot] = (int)expert;
                break;
            }
        }
    }
    for (uint slot = 0; slot < 6u; slot++) {
        selected[row*6u + slot] = top[slot];
    }
}

kernel void rust_star_router_gather_weights_batch(
        device const float *probs,
        device const int *selected,
        device float *weights,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= 6u || index.y >= rust_star_router_max_batch_rows) return;
    weights[index.y*6u + index.x] =
        probs[index.y*256u + (uint)selected[index.y*6u + index.x]];
}

kernel void rust_star_router_clamp_sums_batch(
        device const float *src,
        device float *dst,
        uint row [[thread_position_in_grid]]) {
    if (row >= rust_star_router_max_batch_rows) return;
    dst[row] = clamp(src[row], 6.103515625e-5f, INFINITY);
}

kernel void rust_star_router_divide_batch(
        device const float *weights,
        device const float *sums,
        device float *out,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= 6u || index.y >= rust_star_router_max_batch_rows) return;
    const uint offset = index.y*6u + index.x;
    out[offset] = weights[offset] / sums[index.y];
}

kernel void rust_star_router_scale_batch(
        device const float *weights,
        device float *out,
        uint index [[thread_position_in_grid]]) {
    if (index >= rust_star_router_max_batch_rows*6u) return;
    out[index] = weights[index] * 1.5f;
}
