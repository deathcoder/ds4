#include <metal_stdlib>
using namespace metal;

// Marks composite runtime libraries that include the attention/RoPE types.
// Shared upstream sources such as flash_attn.metal are also compiled into the
// MoE-only library, where those types deliberately do not exist.
#define RUST_STAR_ATTENTION_INGRESS 1

#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];

// `flash_attn.metal` is compiled before `dense.metal` in DwarfStar and uses
// this exact definition through a forward declaration. Rust Star imports only
// the connected dense kernels, so retain the required F16 matrix decoder here.
template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

struct ds4_metal_args_get_rows {
    int ne00t; int ne00;
    ulong nb01; ulong nb02; ulong nb03;
    int ne10;
    ulong nb10; ulong nb11; ulong nb12;
    ulong nb1; ulong nb2; ulong nb3;
};

template<typename T0, typename T>
kernel void kernel_get_rows_f(
        constant ds4_metal_args_get_rows & args,
        device const char * src0, device const char * src1, device char * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int iw0 = tgpig.x/args.ne10;
    const int i10 = tgpig.x%args.ne10;
    const int i11 = tgpig.y;
    const int i12 = tgpig.z;
    const int r = ((const device int *)(src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];
    const int i02 = i11;
    const int i03 = i12;
    auto psrc = (const device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + r*args.nb01);
    auto pdst = (device T *)(dst + i12*args.nb3 + i11*args.nb2 + i10*args.nb1);
    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        pdst[ind] = psrc[ind];
        break;
    }
}

typedef decltype(kernel_get_rows_f<float, float>) get_rows_f_t;
template [[host_name("kernel_get_rows_f16")]] kernel get_rows_f_t kernel_get_rows_f<half, float>;

struct ds4_metal_args_repeat {
    int ne00; int ne01; int ne02; int ne03;
    ulong nb00; ulong nb01; ulong nb02; ulong nb03;
    int ne0; int ne1; int ne2; int ne3;
    ulong nb0; ulong nb1; ulong nb2; ulong nb3;
};

template<typename T>
kernel void kernel_repeat(
        constant ds4_metal_args_repeat & args,
        device const char * src0, device char * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;
    const int i03 = i3%args.ne03;
    const int i02 = i2%args.ne02;
    const int i01 = i1%args.ne01;
    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
    device char * dst_ptr = dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1;
    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i00 = i0%args.ne00;
        *((device T *)(dst_ptr + i0*args.nb0)) = *((device T *)(src0_ptr + i00*args.nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;
template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;

struct ds4_metal_args_norm {
    int ne00; int ne00_t;
    ulong nb1; ulong nb2; ulong nb3;
    float eps;
    int nef1[3]; int nef2[3]; int nef3[3];
    ulong nbf1[3]; ulong nbf2[3]; ulong nbf3[3];
};

template <typename T, short F>
kernel void kernel_rms_norm_fuse_impl(
        constant ds4_metal_args_norm & args,
        device const char * src0, device const char * src1_0,
        device const char * src1_1, device char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    if (sgitg == 0) shmem_f32[tiisg] = 0.0f;
    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    device const T * x = (device const T *)(src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);
    device const T * f0 = (device const T *)(src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *)(src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);
    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) sumf += dot(x[i00], x[i00]);
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) shmem_f32[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);
    const float mean = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);
    device T * y = (device T *)(dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) y[i00] = x[i00]*scale;
        if (F == 2) y[i00] = (x[i00]*scale)*f0[i00];
        if (F == 3) y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float4, 1>) kernel_rms_norm_fuse_t;
template [[host_name("kernel_rms_norm_f32_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 1>;
template [[host_name("kernel_rms_norm_f32_weighted_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 2>;

struct ds4_metal_args_qkv_rms_norm {
    int q_n; int q_n4; int kv_n; int kv_n4;
    ulong q_row_stride; ulong kv_row_stride;
    float eps;
};

kernel void kernel_dsv4_qkv_rms_norm_f32_4(
        constant ds4_metal_args_qkv_rms_norm & args,
        device const float4 * q_src, device const float4 * q_weight,
        device float4 * q_dst, device const float4 * kv_src,
        device const float4 * kv_weight, device float4 * kv_dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    if (sgitg == 0) shmem_f32[tiisg] = 0.0f;
    const uint row = tgpig.x;
    const bool kv_task = tgpig.y != 0;
    const int n = kv_task ? args.kv_n : args.q_n;
    const int n4 = kv_task ? args.kv_n4 : args.q_n4;
    const ulong row_stride4 = (kv_task ? args.kv_row_stride : args.q_row_stride)/sizeof(float4);
    device const float4 *x = kv_task ? kv_src + row*row_stride4 : q_src + row*row_stride4;
    device const float4 *w = kv_task ? kv_weight : q_weight;
    device float4 *y = kv_task ? kv_dst + row*row_stride4 : q_dst + row*row_stride4;
    float sumf = 0.0f;
    for (int i = tpitg.x; i < n4; i += ntg.x) sumf += dot(x[i],x[i]);
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) shmem_f32[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = simd_sum(shmem_f32[tiisg]);
    const float norm_scale = 1.0f/sqrt(sumf/float(n) + args.eps);
    for (int i = tpitg.x; i < n4; i += ntg.x) y[i] = (x[i]*norm_scale)*w[i];
}

struct ds4_metal_args_mul_mv {
    int ne00; int ne01; int ne02;
    ulong nb00; ulong nb01; ulong nb02; ulong nb03;
    int ne10; int ne11; int ne12;
    ulong nb10; ulong nb11; ulong nb12; ulong nb13;
    int ne0; int ne1; int nr0; short r2; short r3;
};

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32, float sumf[NR0], const int r0,
        const int ne01, ushort tiisg, ushort sgitg, threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;
    threadgroup float * shmem_f32[NR0];
    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW*row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) dst_f32[r0 + row] = tot;
    }
}

template<short NR0>
void kernel_mul_mv_f16_f32_4_impl(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0, device const char * src1, device char * dst,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = 32;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = 4;
    const int nb = args.ne00/NB;
    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const float * y = (device const float *)(src1 + offset1);
    device const float4 * y4 = (device const float4 *)(src1 + offset1);
    device const half * ax[NR0];
    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        ax[row] = (device const half *)(src0 + offset0);
        ax4[row] = (device const half4 *)(src0 + offset0);
    }
    float sumf[NR0] = {0.f};
    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;
    float4 yl4[NF4];
    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;
    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) yl4[i] = yb4[i];
        for (short row = 0; row < NR0; row++) {
            device const half4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) sumq += dot(float4(xb4[i]), float4(yl4[i]));
            sumf[row] += sumq;
        }
        yb4 += NSG*NF*NW/4;
    }
    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) sumf[row] += ax[row][i]*y[i];
    }
    device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

[[host_name("kernel_mul_mv_f16_f32_4")]]
kernel void kernel_mul_mv_f16_f32_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0, device const char * src1, device char * dst,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_f16_f32_4_impl<2>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// Exact DwarfStar M1 compressor pair path. The two projections retain the
// original per-matrix reduction order while sharing the activation stream.
template<short NR0>
void kernel_mul_mv_f16_f32_pair_4_impl(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_a, device const char * src0_b,
        device const char * src1, device char * dst_a, device char * dst_b,
        threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = 32;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = 4;
    const int nb = args.ne00/NB;
    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const float * y = (device const float *)(src1 + offset1);
    device const float4 * y4 = (device const float4 *)(src1 + offset1);
    device const half * ax_a[NR0];
    device const half4 * ax4_a[NR0];
    device const half * ax_b[NR0];
    device const half4 * ax4_b[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 +
            (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        ax_a[row] = (device const half *)(src0_a + offset0);
        ax4_a[row] = (device const half4 *)(src0_a + offset0);
        ax_b[row] = (device const half *)(src0_b + offset0);
        ax4_b[row] = (device const half4 *)(src0_b + offset0);
    }
    float sum_a[NR0] = {0.f};
    float sum_b[NR0] = {0.f};
    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;
    float4 yl4[NF4];
    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;
    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) yl4[i] = yb4[i];
        for (short row = 0; row < NR0; row++) {
            device const half4 * xb4_a = ax4_a[row] + (ib*NB + il*NF)/4;
            device const half4 * xb4_b = ax4_b[row] + (ib*NB + il*NF)/4;
            float suma = 0.f;
            float sumb = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                const float4 yv = float4(yl4[i]);
                suma += dot(float4(xb4_a[i]), yv);
                sumb += dot(float4(xb4_b[i]), yv);
            }
            sum_a[row] += suma;
            sum_b[row] += sumb;
        }
        yb4 += NSG*NF*NW/4;
    }
    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            const float yi = y[i];
            sum_a[row] += ax_a[row][i]*yi;
            sum_b[row] += ax_b[row][i]*yi;
        }
    }
    device float * dst_a_f32 = (device float *)dst_a +
        (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_b_f32 = (device float *)dst_b +
        (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    helper_mv_reduce_and_write<NR0>(dst_a_f32, sum_a, r0, args.ne01, tiisg, sgitg, shmem);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    helper_mv_reduce_and_write<NR0>(dst_b_f32, sum_b, r0, args.ne01, tiisg, sgitg, shmem);
}

kernel void kernel_mul_mv_f16_f32_pair_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_a, device const char * src0_b,
        device const char * src1, device char * dst_a, device char * dst_b,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (args.nr0 == 4) {
        kernel_mul_mv_f16_f32_pair_4_impl<4>(args, src0_a, src0_b, src1,
            dst_a, dst_b, shmem, tgpig, tiisg, sgitg);
    } else {
        kernel_mul_mv_f16_f32_pair_4_impl<2>(args, src0_a, src0_b, src1,
            dst_a, dst_b, shmem, tgpig, tiisg, sgitg);
    }
}

struct rust_star_compressor_pack_args {
    uint width;
    uint head_dim;
    uint ratio;
};

// Reproduces the strided-to-contiguous copies that precede the legacy
// compressor softmax graph. Ratio-4 has two four-row lanes; ratio-128 has one
// 128-row lane. Output keeps the pooled rows contiguous for each head element.
kernel void rust_star_compressor_pack_ratio4_one(
        constant rust_star_compressor_pack_args & args,
        device const float * state_kv,
        device const float * state_score,
        device float * packed_kv,
        device float * packed_score,
        uint gid [[thread_position_in_grid]]) {
    const uint rows = args.ratio == 4u ? 8u : args.ratio;
    const uint total = args.head_dim*rows;
    if (gid >= total) return;
    const uint col = gid/rows;
    const uint row = gid - col*rows;
    const uint src = args.ratio == 4u
        ? (row < 4u
            ? row*args.width + col
            : row*args.width + args.head_dim + col)
        : row*args.width + col;
    packed_kv[gid] = state_kv[src];
    packed_score[gid] = state_score[src];
}

kernel void rust_star_mul_f32(
        device const float * a,
        device const float * b,
        device float * dst,
        uint gid [[thread_position_in_grid]]) {
    dst[gid] = a[gid]*b[gid];
}

struct rust_star_compressor_score_batch_args {
    uint width;
    uint ratio;
    uint pos0;
    uint n_tokens;
};

kernel void rust_star_compressor_score_ape_f16_batch(
        constant rust_star_compressor_score_batch_args & args,
        device const float * score,
        device const half * ape,
        device float * dst,
        uint gid [[thread_position_in_grid]]) {
    const ulong total = (ulong)args.n_tokens*args.width;
    if ((ulong)gid >= total) return;
    const uint token = gid/args.width;
    const uint col = gid-token*args.width;
    const uint ape_row = (args.pos0+token)%args.ratio;
    dst[gid] = score[gid] + float(ape[(ulong)ape_row*args.width+col]);
}

struct rust_star_compressor_pack_batch_args {
    uint head_dim;
    uint n_comp;
    uint replay;
    uint n_threads;
};

struct rust_star_compressor_state_reset_args {
    uint elements;
};

// DwarfStar rebuilds the recurrent ratio-4 state after batched prefill. Rows
// 0--3 are populated separately from small-batch prompt-tail projections;
// rows 4--7 remain empty for the next four-token compression window.
kernel void rust_star_compressor_state_reset_ratio4(
        constant rust_star_compressor_state_reset_args & args,
        device float * state_kv,
        device float * state_score,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.elements) return;
    state_kv[gid] = 0.0f;
    state_score[gid] = -INFINITY;
}

struct rust_star_compressor_state_set_tail4_args {
    uint width;
    uint pos0;
};

kernel void rust_star_compressor_state_set_tail4(
        constant rust_star_compressor_state_set_tail4_args & args,
        device const float * kv,
        device const float * score,
        device const half * ape,
        device float * state_kv,
        device float * state_score,
        uint gid [[thread_position_in_grid]]) {
    const ulong total = 4ul*args.width;
    if ((ulong)gid >= total) return;
    const uint row = gid/args.width;
    const uint col = gid-row*args.width;
    const uint ape_row = (args.pos0+row)%4u;
    state_kv[gid] = kv[gid];
    state_score[gid] = score[gid] + float(ape[(ulong)ape_row*args.width+col]);
}

kernel void rust_star_compressor_pack_ratio4_batch(
        constant rust_star_compressor_pack_batch_args & args,
        device const uint * kv,
        device const uint * score,
        device const uint * state_kv,
        device const uint * state_score,
        device uint * packed_kv,
        device uint * packed_score,
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (group.x >= args.n_comp || group.y >= 8u ||
        args.head_dim == 0u || args.n_threads == 0u) return;
    const uint plane = group.x;
    const uint row = group.y;
    const ulong input_stride = 2ul*args.head_dim;
    const ulong dst_row = ((ulong)plane*8u+row)*args.head_dim;
    for (uint col = tid; col < args.head_dim; col += args.n_threads) {
        const ulong dst = dst_row+col;
        if (row >= 4u) {
            const uint token = plane*4u+(row-4u);
            const ulong src = (ulong)token*input_stride+args.head_dim+col;
            packed_kv[dst] = kv[src];
            packed_score[dst] = score[src];
        } else if (plane != 0u) {
            const uint token = (plane-1u)*4u+row;
            const ulong src = (ulong)token*input_stride+col;
            packed_kv[dst] = kv[src];
            packed_score[dst] = score[src];
        } else if (args.replay != 0u) {
            const ulong src = (ulong)row*input_stride+col;
            packed_kv[dst] = state_kv[src];
            packed_score[dst] = state_score[src];
        } else {
            packed_kv[dst] = 0u;
            packed_score[dst] = 0xff800000u;
        }
    }
}

struct rust_star_compressor_pool_batch_args {
    uint head_dim;
    uint n_comp;
};

kernel void rust_star_compressor_softmax_pool_batch(
        constant rust_star_compressor_pool_batch_args & args,
        device const float * kv,
        device const float * score,
        device float * dst,
        uint gid [[thread_position_in_grid]]) {
    const ulong total = (ulong)args.head_dim*args.n_comp;
    if ((ulong)gid >= total) return;
    const uint id = gid%args.head_dim;
    const uint plane = gid/args.head_dim;
    const ulong base = ((ulong)plane*8u)*args.head_dim+id;
    float max_s = -INFINITY;
    for (uint row = 0; row < 8u; row++) {
        max_s = max(max_s, score[base+(ulong)row*args.head_dim]);
    }
    float sum = 0.0f;
    float acc = 0.0f;
    for (uint row = 0; row < 8u; row++) {
        const ulong index = base+(ulong)row*args.head_dim;
        const float weight = exp(score[index]-max_s);
        sum += weight;
        acc += kv[index]*weight;
    }
    dst[gid] = acc/sum;
}

struct rust_star_compressor_softmax_pool_strided_args {
    long ne00;
    long ne01;
    long ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    long ne0;
    long ne1;
    ulong nb0;
    ulong nb1;
};

// Exact full-prefill ratio-128 path. This intentionally preserves the scalar
// reduction order of DwarfStar's kernel_dsv4_softmax_pool implementation.
kernel void rust_star_compressor_softmax_pool_strided(
        constant rust_star_compressor_softmax_pool_strided_args & args,
        device const char * kv,
        device const char * score,
        device char * dst,
        uint gid [[thread_position_in_grid]]) {
    const long n = args.ne0*args.ne1;
    if ((long)gid >= n) return;

    const long id = gid%args.ne0;
    const long ic = gid/args.ne0;
    float max_s = -INFINITY;
    for (long row = 0; row < args.ne00; row++) {
        const float value = *((device const float *)(
            score + row*args.nb10 + id*args.nb11 + ic*args.nb12));
        max_s = max(max_s, value);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (long row = 0; row < args.ne00; row++) {
        const float value = *((device const float *)(
            score + row*args.nb10 + id*args.nb11 + ic*args.nb12));
        const float weight = exp(value-max_s);
        const float projected = *((device const float *)(
            kv + row*args.nb00 + id*args.nb01 + ic*args.nb02));
        sum += weight;
        acc += projected*weight;
    }
    *((device float *)(dst + id*args.nb0 + ic*args.nb1)) = acc/sum;
}

struct ds4_metal_args_dsv4_hc_split_weighted_sum_norm {
    long n_embd; int n_hc; int sinkhorn_iters; long n_rows; long mix_hc;
    ulong nb_mix1; ulong nb_split1;
    ulong nb_x0; ulong nb_x1; ulong nb_x2;
    ulong nb0; ulong nb1; ulong nb_norm1;
    float eps; float norm_eps;
};

kernel void kernel_dsv4_hc_split_weighted_sum_norm4(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & args,
        device const char * mixes, device const float * scale,
        device const float * base, device const char * x,
        device char * split, device char * dst,
        device const char * norm_weight, device char * norm_dst,
        threadgroup float * shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if ((long)row >= args.n_rows || args.n_hc != 4 || (args.n_embd & 3) != 0) return;
    const uint n_embd = uint(args.n_embd);
    const uint n4 = n_embd >> 2;
    threadgroup float4 *row_shmem = (threadgroup float4 *)shared;
    threadgroup float *pre_shmem = shared + n_embd;
    threadgroup float *sum_shmem = pre_shmem + 4;
    device const float *mix = (device const float *)(mixes + (ulong)row*args.nb_mix1);
    device float *out = (device float *)(split + (ulong)row*args.nb_split1);
    if (sgitg == 0) sum_shmem[tiisg] = 0.0f;
    if (tid == 0) {
        const float epsv = args.eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        const float4 pre_z = *((device const float4 *)mix)*pre_scale + *((device const float4 *)base);
        const float4 pre = 1.0f/(1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x; pre_shmem[1] = pre.y; pre_shmem[2] = pre.z; pre_shmem[3] = pre.w;
        const float4 post_z = *((device const float4 *)(mix + 4))*post_scale + *((device const float4 *)(base + 4));
        *((device float4 *)(out + 4)) = 2.0f/(1.0f + exp(-post_z));
        float4 r0 = *((device const float4 *)(mix + 8))*comb_scale + *((device const float4 *)(base + 8));
        float4 r1 = *((device const float4 *)(mix + 12))*comb_scale + *((device const float4 *)(base + 12));
        float4 r2 = *((device const float4 *)(mix + 16))*comb_scale + *((device const float4 *)(base + 16));
        float4 r3 = *((device const float4 *)(mix + 20))*comb_scale + *((device const float4 *)(base + 20));
        const float m0 = max(max(r0.x,r0.y),max(r0.z,r0.w));
        const float m1 = max(max(r1.x,r1.y),max(r1.z,r1.w));
        const float m2 = max(max(r2.x,r2.y),max(r2.z,r2.w));
        const float m3 = max(max(r3.x,r3.y),max(r3.z,r3.w));
        r0 = exp(r0-m0); r1 = exp(r1-m1); r2 = exp(r2-m2); r3 = exp(r3-m3);
        r0 = r0*(1.0f/(r0.x+r0.y+r0.z+r0.w)) + epsv;
        r1 = r1*(1.0f/(r1.x+r1.y+r1.z+r1.w)) + epsv;
        r2 = r2*(1.0f/(r2.x+r2.y+r2.z+r2.w)) + epsv;
        r3 = r3*(1.0f/(r3.x+r3.y+r3.z+r3.w)) + epsv;
        float4 col_inv = 1.0f/(r0+r1+r2+r3+epsv);
        r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f/(r0.x+r0.y+r0.z+r0.w+epsv);
            r1 *= 1.0f/(r1.x+r1.y+r1.z+r1.w+epsv);
            r2 *= 1.0f/(r2.x+r2.y+r2.z+r2.w+epsv);
            r3 *= 1.0f/(r3.x+r3.y+r3.z+r3.w+epsv);
            col_inv = 1.0f/(r0+r1+r2+r3+epsv);
            r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        }
        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0*args.nb_x1 + (ulong)row*args.nb_x2);
        device const float4 *x1 = (device const float4 *)(x + 1*args.nb_x1 + (ulong)row*args.nb_x2);
        device const float4 *x2 = (device const float4 *)(x + 2*args.nb_x1 + (ulong)row*args.nb_x2);
        device const float4 *x3 = (device const float4 *)(x + 3*args.nb_x1 + (ulong)row*args.nb_x2);
        float4 v = 0.0f;
        v += x0[i]*pre_shmem[0]; v += x1[i]*pre_shmem[1];
        v += x2[i]*pre_shmem[2]; v += x3[i]*pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v,v);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) sum_shmem[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = simd_sum(sum_shmem[tiisg]);
    const float norm_arg = sumf/float(n_embd) + args.norm_eps;
    const float norm_scale = args.n_rows > 1 ? 1.0f/sqrt(norm_arg) : rsqrt(norm_arg);
    device float4 *dst4 = (device float4 *)(dst + (ulong)row*args.nb1);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)(norm_dst + (ulong)row*args.nb_norm1);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v*norm_scale)*w4[i];
    }
}
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

// Match the oracle's descending bitonic permutation, including equal scores.
// Stable insertion order can select a different sixth expert on a score tie.
kernel void rust_star_router_topk_rows_batch(
        device const float *probs,
        device const float *bias,
        device int *selected,
        uint row [[threadgroup_position_in_grid]],
        uint col [[thread_position_in_threadgroup]]) {
    if (row >= rust_star_router_max_batch_rows) return;
    threadgroup int indices[256];
    threadgroup float scores[256];
    indices[col] = (int)col;
    scores[col] = probs[row*256u + col] + bias[col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2u; k <= 256u; k *= 2u) {
        for (uint j = k/2u; j > 0u; j /= 2u) {
            const uint other = col ^ j;
            if (other > col) {
                const bool swap = (col & k) == 0u
                    ? scores[indices[col]] < scores[indices[other]]
                    : scores[indices[col]] > scores[indices[other]];
                if (swap) {
                    const int prior = indices[col];
                    indices[col] = indices[other];
                    indices[other] = prior;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (col < 6u) selected[row*6u + col] = indices[col];
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
struct ds4_metal_args_dsv4_rope_tail {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_dims;
    int32_t  mode;
    int32_t  n_ctx_orig;
    int32_t  inverse;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    bool     src2;
};

struct ds4_metal_args_dsv4_rope_affine_pair {
    uint64_t row_bytes;
    uint64_t token_bytes;
    int32_t head_dim;
    int32_t n_dims;
    int32_t n_ctx_orig;
    int32_t inverse;
    uint32_t pos0;
    uint32_t pos_step;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
};

struct ds4_metal_args_dsv4_head_norm_rope {
    int32_t n_head;
    int32_t head_dim;
    int32_t head_dim4;
    int32_t n_dims;
    int32_t n_ctx_orig;
    int32_t pos0;
    int32_t inverse;
    float eps;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

// Applies DeepSeek V4's partial RoPE: the no-position prefix is copied and only
// the rotated tail is transformed. This is used for Q/K after their projections
// and before writing/reading the attention KV state.
kernel void kernel_dsv4_rope_tail_f32(
        constant ds4_metal_args_dsv4_rope_tail & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        uint  tid   [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];

    const int n_nope = args.ne00 - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    device const int32_t * pos = (device const int32_t *) src1;

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;
    const bool is_neox = args.mode == 2;

    for (int i0 = tid; i0 < args.ne00; i0 += ntg.x) {
        device const char * src_base = src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01;
        device       char * dst_base = dst  + i3*args.nb3  + i2*args.nb2  + i1*args.nb1;

        if (i0 < n_nope) {
            *((device float *) (dst_base + i0*args.nb0)) = *((device const float *) (src_base + i0*args.nb00));
            continue;
        }

        const int r = i0 - n_nope;
        if (is_neox) {
            const int n_half = args.n_dims/2;
            if (r >= n_half) {
                continue;
            }

            const int ic = r;
            const int rel_i0 = 2*ic;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
            // Equivalent to pow(freq_base, k) but expressed through IEEE-754
            // primitives that have tighter precision guarantees than Metal's pow().
            const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
            const float theta = theta_base * pow(args.freq_base, inv_ndims*rel_i0);
#endif
            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, rel_i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = *((device const float *) (src_base + j0*args.nb00));
            const float x1 = *((device const float *) (src_base + j1*args.nb00));

            *((device float *) (dst_base + j0*args.nb0)) = x0*cos_theta - x1*sin_theta;
            *((device float *) (dst_base + j1*args.nb0)) = x0*sin_theta + x1*cos_theta;
        } else {
            if ((r & 1) != 0) {
                continue;
            }

            const int ic = r/2;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
            const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
            const float theta = theta_base * pow(args.freq_base, inv_ndims*r);
#endif
            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, r, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = *((device const float *) (src_base + j0*args.nb00));
            const float x1 = *((device const float *) (src_base + j1*args.nb00));

            *((device float *) (dst_base + j0*args.nb0)) = x0*cos_theta - x1*sin_theta;
            *((device float *) (dst_base + j1*args.nb0)) = x0*sin_theta + x1*cos_theta;
        }
    }
}

// DS4 only calls the Metal RoPE helper in-place and uses the adjacent-pair
// layout (mode 0). The generic kernel above still copies the no-position
// prefix, even though source and destination alias, and dispatches enough
// lanes for the full head. This specialization maps lanes directly to the
// rotated pairs and deliberately never reads or writes the unchanged prefix.
// Keep the arithmetic below in the same order as the mode-0 branch above so
// the optimized and reference paths remain bit-identical.
kernel void kernel_dsv4_rope_tail_f32_inplace_pair(
        constant ds4_metal_args_dsv4_rope_tail & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        uint  tid   [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    if (args.mode != 0) {
        return;
    }

    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];
    const int n_nope = args.ne00 - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    device const int32_t * pos = (device const int32_t *) src1;

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;
    device const char * src_base = src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01;
    device       char * dst_base = dst  + i3*args.nb3  + i2*args.nb2  + i1*args.nb1;

    // Keep each pair on the same SIMD lane as the generic in-place dispatch.
    // n_nope is 32-aligned for the supported DS4 shapes, so logical tail index
    // r ran on lane r%32 in the reference kernel. Compacting pairs would move
    // them to lane (r/2)%32 and can perturb fast-math code generation.
    for (int r = tid; r < args.n_dims; r += ntg.x) {
        if ((r & 1) != 0) {
            continue;
        }
        const int ic = r/2;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims*r);
#endif
        const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, r, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
        if (args.inverse) {
            sin_theta = -sin_theta;
        }

        const int j0 = n_nope + r;
        const int j1 = j0 + 1;
        const float x0 = *((device const float *) (src_base + j0*args.nb00));
        const float x1 = *((device const float *) (src_base + j1*args.nb00));

        *((device float *) (dst_base + j0*args.nb0)) = x0*cos_theta - x1*sin_theta;
        *((device float *) (dst_base + j1*args.nb0)) = x0*sin_theta + x1*cos_theta;
    }
}

// The Q/K RoPE calls use the same position and scaling parameters for every
// head. Group four heads into one threadgroup so one 64-thread cohort computes
// the 32 adjacent-pair coefficients and the other cohorts reuse them. Keeping
// r on the same r%32 SIMD lane as the per-head specialization preserves the
// fast-math instruction mapping; only the redundant coefficient work changes.
kernel void kernel_dsv4_rope_tail_f32_inplace_pair_shared4(
        constant ds4_metal_args_dsv4_rope_tail & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        uint  tid   [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    if (args.mode != 0 || args.n_dims != 64) {
        return;
    }

    const int n_nope = args.ne00 - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    const uint cohort = tid >> 6;
    const int r = (int)(tid & 63u);
    threadgroup float cos_shared[32];
    threadgroup float sin_shared[32];

    device const int32_t * pos = (device const int32_t *) src1;
    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    const int i2 = tgpig[1];
    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    if (cohort == 0 && (r & 1) == 0) {
        const int ic = r/2;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims*r);
#endif
        const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, r, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
        if (args.inverse) {
            sin_theta = -sin_theta;
        }
        cos_shared[ic] = cos_theta;
        sin_shared[ic] = sin_theta;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int i1 = (int)tgpig[0]*4 + (int)cohort;
    if (i1 >= args.ne01 || (r & 1) != 0) {
        return;
    }

    const int i3 = tgpig[2];
    device const char * src_base = src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01;
    device       char * dst_base = dst  + i3*args.nb3  + i2*args.nb2  + i1*args.nb1;
    const int j0 = n_nope + r;
    const int j1 = j0 + 1;
    const float x0 = *((device const float *) (src_base + j0*args.nb00));
    const float x1 = *((device const float *) (src_base + j1*args.nb00));
    const float cos_theta = cos_shared[r/2];
    const float sin_theta = sin_shared[r/2];

    *((device float *) (dst_base + j0*args.nb0)) = x0*cos_theta - x1*sin_theta;
    *((device float *) (dst_base + j1*args.nb0)) = x0*sin_theta + x1*cos_theta;
}

// Fuses the per-head RMSNorm and partial Q RoPE while retaining the standalone
// norm reduction tree and the mode-0 RoPE lane mapping.
kernel void kernel_dsv4_head_rms_norm_rope_tail_f32(
        constant ds4_metal_args_dsv4_head_norm_rope & args,
        device char * xraw,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const uint head = tgpig.x;
    const uint tok = tgpig.y;
    device float4 * x4 = (device float4 *)xraw +
        ((uint64_t)tok * (uint64_t)args.n_head + head) *
        (uint64_t)args.head_dim4;

    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.head_dim4; i00 += ntg.x) {
        sumf += dot(x4[i00], x4[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = simd_sum(shmem_f32[tiisg]);
    const float scale = 1.0f / sqrt(sumf / args.head_dim + args.eps);
    const int n_nope = args.head_dim - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base,
                        args.beta_fast, args.beta_slow, corr_dims);
    const float theta_base = (float)(args.pos0 + (int)tok);
    const float inv_ndims = -1.0f / args.n_dims;
    device float * xs = (device float *)x4;

    for (int i0 = tpitg.x; i0 < args.head_dim; i0 += ntg.x) {
        if (i0 < n_nope) {
            xs[i0] = xs[i0] * scale;
            continue;
        }
        const int r = i0 - n_nope;
        if ((r & 1) != 0) {
            continue;
        }
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta =
            theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta =
            theta_base * pow(args.freq_base, inv_ndims * r);
#endif
        float cos_theta;
        float sin_theta;
        rope_yarn(theta, args.freq_scale, corr_dims, r,
                  args.ext_factor, args.attn_factor,
                  &cos_theta, &sin_theta);
        if (args.inverse) {
            sin_theta = -sin_theta;
        }

        const float x0 = xs[i0] * scale;
        const float x1 = xs[i0 + 1] * scale;
        xs[i0] = x0 * cos_theta - x1 * sin_theta;
        xs[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

// The 64 Q heads and the single in-place KV tail use the same position and
// YaRN parameters. Group 64 folds the standalone KV RoPE launch into the Q
// dispatch while groups 0..63 retain the exact Q reduction and lane mapping.
kernel void kernel_dsv4_head_rms_norm_q_kv_rope_tail_f32(
        constant ds4_metal_args_dsv4_head_norm_rope & args,
        device char * qraw,
        device char * kvraw,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint head = tgpig.x;
    const uint tok = tgpig.y;
    const int n_nope = args.head_dim - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    if (head == (uint)args.n_head) {
        float corr_dims[2];
        rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base,
                            args.beta_fast, args.beta_slow, corr_dims);
        const float theta_base = (float)(args.pos0 + (int)tok);
        const float inv_ndims = -1.0f / args.n_dims;
        device float * xs = (device float *)kvraw +
            (uint64_t)tok * (uint64_t)args.head_dim;

        for (int i0 = tpitg.x; i0 < args.head_dim; i0 += ntg.x) {
            if (i0 < n_nope) {
                continue;
            }
            const int r = i0 - n_nope;
            if ((r & 1) != 0) {
                continue;
            }
#ifdef DS4_METAL_ROPE_EXP2_LOG2
            const float theta =
                theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
            const float theta =
                theta_base * pow(args.freq_base, inv_ndims * r);
#endif
            float cos_theta;
            float sin_theta;
            rope_yarn(theta, args.freq_scale, corr_dims, r,
                      args.ext_factor, args.attn_factor,
                      &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }
            const float x0 = xs[i0];
            const float x1 = xs[i0 + 1];
            xs[i0] = x0 * cos_theta - x1 * sin_theta;
            xs[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
        }
        return;
    }
    if (head >= (uint)args.n_head) {
        return;
    }

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }
    device float4 * x4 = (device float4 *)qraw +
        ((uint64_t)tok * (uint64_t)args.n_head + head) *
        (uint64_t)args.head_dim4;
    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.head_dim4; i00 += ntg.x) {
        sumf += dot(x4[i00], x4[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = simd_sum(shmem_f32[tiisg]);
    const float scale = 1.0f / sqrt(sumf / args.head_dim + args.eps);
    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base,
                        args.beta_fast, args.beta_slow, corr_dims);
    const float theta_base = (float)(args.pos0 + (int)tok);
    const float inv_ndims = -1.0f / args.n_dims;
    device float * xs = (device float *)x4;

    for (int i0 = tpitg.x; i0 < args.head_dim; i0 += ntg.x) {
        if (i0 < n_nope) {
            xs[i0] = xs[i0] * scale;
            continue;
        }
        const int r = i0 - n_nope;
        if ((r & 1) != 0) {
            continue;
        }
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta =
            theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta =
            theta_base * pow(args.freq_base, inv_ndims * r);
#endif
        float cos_theta;
        float sin_theta;
        rope_yarn(theta, args.freq_scale, corr_dims, r,
                  args.ext_factor, args.attn_factor,
                  &cos_theta, &sin_theta);
        if (args.inverse) {
            sin_theta = -sin_theta;
        }
        const float x0 = xs[i0] * scale;
        const float x1 = xs[i0 + 1] * scale;
        xs[i0] = x0 * cos_theta - x1 * sin_theta;
        xs[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

// DS4 positions are always affine within one RoPE dispatch. This variant
// reconstructs the same wrapped int32 position in-kernel, avoiding the host
// position array and its buffer binding while preserving the pair lane mapping
// and all floating-point operations of the specialization above.
kernel void kernel_dsv4_rope_tail_f32_inplace_pair_affine(
        constant ds4_metal_args_dsv4_rope_affine_pair & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device       char * dst  [[buffer(4)]],
        uint  tid   [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int n_nope = args.head_dim - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    const uint raw_pos = args.pos0 + (uint)i2 * args.pos_step;
    const float theta_base = (float)as_type<int>(raw_pos);
    const float inv_ndims = -1.f/args.n_dims;
    device const char * src_base =
        src0 + (uint64_t)i2*args.token_bytes + (uint64_t)i1*args.row_bytes;
    device char * dst_base =
        dst + (uint64_t)i2*args.token_bytes + (uint64_t)i1*args.row_bytes;

    for (int r = tid; r < args.n_dims; r += ntg.x) {
        if ((r & 1) != 0) {
            continue;
        }
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims*r);
#endif
        const float freq_factor = 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, r, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
        if (args.inverse) {
            sin_theta = -sin_theta;
        }

        const int j0 = n_nope + r;
        const int j1 = j0 + 1;
        const float x0 = *((device const float *) (src_base + j0*sizeof(float)));
        const float x1 = *((device const float *) (src_base + j1*sizeof(float)));

        *((device float *) (dst_base + j0*sizeof(float))) = x0*cos_theta - x1*sin_theta;
        *((device float *) (dst_base + j1*sizeof(float))) = x0*sin_theta + x1*cos_theta;
    }
}
constant float dsv4_e4m3fn_exp_scale[16] = {
    0.0f, 0.015625f, 0.03125f, 0.0625f,
    0.125f, 0.25f, 0.5f, 1.0f,
    2.0f, 4.0f, 8.0f, 16.0f,
    32.0f, 64.0f, 128.0f, 256.0f,
};

constant float dsv4_e2m1fn_values[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

struct ds4_metal_args_dsv4_fp8_kv_quantize {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    ulong nb0;
    ulong nb1;
    ulong nb2;
    ulong nb3;
    int n_rot;
};

struct ds4_metal_args_dsv4_kv_fp8_store {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
};

struct ds4_metal_args_dsv4_indexer_qat {
    uint32_t n_rows;
    uint32_t head_dim;
    uint64_t row_stride;
};

struct ds4_metal_args_dsv4_ratio4_shift {
    uint32_t width;
};

struct ds4_metal_args_dsv4_compressor_pack_ratio4 {
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t n_threads;
};

struct ds4_metal_args_dsv4_compressor_store_one {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
};

static inline float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * dsv4_e4m3fn_exp_scale[exp];
}

static inline float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = abs(ax - dsv4_e4m3fn_value(best));
        const float next_diff = abs(ax - dsv4_e4m3fn_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best = best + 1;
        }
    }

    return sign * dsv4_e4m3fn_value(best);
}

static inline float dsv4_e2m1fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 6.0f);
    int best = 0;
    float best_diff = abs(ax - dsv4_e2m1fn_values[0]);
    for (int i = 1; i < 8; i++) {
        const float diff = abs(ax - dsv4_e2m1fn_values[i]);
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_values[best];
}

// Quantizes the non-RoPE part of a KV row through E4M3FN and writes the
// dequantized value back as float. DS4 uses this to match the FP8 KV-cache
// semantics while keeping the Metal graph's cache buffers float-addressable.
kernel void kernel_dsv4_fp8_kv_quantize_f32(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const char * src0,
        device        char * dst,
        threadgroup  float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if ((int64_t) row >= n_rows) {
        return;
    }

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    device const char * src_base = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device       char * dst_base = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;

    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64) {
            v = *((device const float *) (src_base + (off + tid)*args.nb00));
            scratch[tid] = abs(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float scale = exp2(ceil(log2(amax / 448.0f)));
        if (tid < 64) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / scale, -448.0f, 448.0f)) * scale;
            *((device float *) (dst_base + (off + tid)*args.nb0)) = q;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int64_t i = n_nope + tid; i < args.ne00; i += 64) {
        *((device float *) (dst_base + i*args.nb0)) = *((device const float *) (src_base + i*args.nb00));
    }
}

// The official DS4 indexer applies a 128-wide Hadamard rotation and then an
// inplace FP4 activation-simulation pass to both indexer Q and indexer KV.
kernel void kernel_dsv4_indexer_hadamard_fp4_f32(
        constant ds4_metal_args_dsv4_indexer_qat & args,
        device   char  * x,
        threadgroup float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.n_rows || args.head_dim != 128u || tid >= 128u) {
        return;
    }

    threadgroup float *vals = scratch;
    threadgroup float *absbuf = scratch + 128;
    device float *xr = (device float *)(x + (uint64_t)row * args.row_stride);

    vals[tid] = xr[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            const uint base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            const float a = vals[base];
            const float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float v = vals[tid] * 0.08838834764831845f;
    const uint block = tid >> 5u;
    const uint lane = tid & 31u;
    const uint block_base = block * 32u;
    absbuf[tid] = abs(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = max(absbuf[block_base + lane],
                                            absbuf[block_base + lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float amax = max(absbuf[block_base], 7.052966104933725e-38f);
    const float scale = exp2(ceil(log2(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant(clamp(v / scale, -6.0f, 6.0f)) * scale;
}

// Decode-side KV finalizer after RoPE. The normal RoPE kernel intentionally
// remains separate because tiny trigonometric codegen changes can flip later
// sampled tokens. This kernel only fuses the FP8 round-trip for the non-RoPE
// prefix with the F16-rounded raw-cache row used by FlashAttention.
kernel void kernel_dsv4_kv_fp8_store_f32(
        constant ds4_metal_args_dsv4_kv_fp8_store & args,
        device        float * kv,
        device        float * raw_cache,
        threadgroup   float * scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int head_dim = args.head_dim;
    const int n_rot = args.n_rot;
    const int n_nope = head_dim - n_rot;
    if (head_dim <= 0 || n_rot < 0 || n_nope < 0 || tid >= 64) {
        return;
    }

    device float * raw = raw_cache + (int64_t)args.raw_row * head_dim;

    for (int off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + (int)tid < n_nope) {
            v = kv[off + tid];
            scratch[tid] = abs(v);
        } else {
            scratch[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float fp8_scale = exp2(ceil(log2(amax / 448.0f)));
        if (off + (int)tid < n_nope) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / fp8_scale, -448.0f, 448.0f)) * fp8_scale;
            kv[off + tid] = q;
            // Diagnostic only: skip the FP16 round-trip that normally matches the
            // half-typed FlashAttention KV buffer's precision. With this enabled the
            // indexer will see higher-precision raw values than FlashAttention does,
            // which is informative but not a production-ready setting.
#ifdef DS4_METAL_KV_RAW_F32
            raw[off + tid] = q;
#else
            raw[off + tid] = (float)((half)q);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = n_nope + tid; i < head_dim; i += 64) {
#ifdef DS4_METAL_KV_RAW_F32
        raw[i] = kv[i];
#else
        raw[i] = (float)((half)kv[i]);
#endif
    }
}

// Builds the two ratio-4 softmax-pool packs together. Each output plane holds
// four previous-half rows followed by four current-half rows. Normal prefill
// seeds plane zero with 0/-inf; replay seeds it from the previous compressor
// state. Later planes take their previous half from the prior input group.
kernel void kernel_dsv4_compressor_pack_ratio4(
        constant ds4_metal_args_dsv4_compressor_pack_ratio4 & args,
        device const uint * kv,
        device const uint * score,
        device const uint * state_kv,
        device const uint * state_score,
        device       uint * packed_kv,
        device       uint * packed_score,
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (group.x >= args.n_comp || group.y >= 8u ||
        args.head_dim == 0u || args.n_threads == 0u) {
        return;
    }

    const uint plane = group.x;
    const uint row = group.y;
    const uint64_t input_row_stride = 2ull * args.head_dim;
    const uint64_t dst_row = ((uint64_t)plane * 8u + row) * args.head_dim;

    for (uint col = tid; col < args.head_dim; col += args.n_threads) {
        const uint64_t dst = dst_row + col;
        if (row >= 4u) {
            const uint token = plane * 4u + (row - 4u);
            const uint64_t src = (uint64_t)token * input_row_stride +
                                 args.head_dim + col;
            packed_kv[dst] = kv[src];
            packed_score[dst] = score[src];
        } else if (plane != 0u) {
            const uint token = (plane - 1u) * 4u + row;
            const uint64_t src = (uint64_t)token * input_row_stride + col;
            packed_kv[dst] = kv[src];
            packed_score[dst] = score[src];
        } else if (args.replay != 0u) {
            const uint64_t src = (uint64_t)row * input_row_stride + col;
            packed_kv[dst] = state_kv[src];
            packed_score[dst] = state_score[src];
        } else {
            packed_kv[dst] = 0u;
            packed_score[dst] = 0xff800000u;
        }
    }
}

// Ratio-4 compression keeps two 4-row halves of recurrent state. After an
// emitted compressed row, the second half becomes the next window's previous
// half. The old encoder expressed this as four generic copies; this DS4-specific
// kernel performs the KV and score copies together.
kernel void kernel_dsv4_ratio4_shift_f32(
        constant ds4_metal_args_dsv4_ratio4_shift & args,
        device float * state_kv,
        device float * state_score,
        uint gid [[thread_position_in_grid]]) {
    const uint n = 4u * args.width;
    if (gid >= n) return;

    const float next_kv = state_kv[n + gid];
    const float next_score = state_score[n + gid];
    state_kv[gid] = next_kv;
    state_score[gid] = next_score;
    state_kv[n + gid] = 0.0f;
    state_score[n + gid] = -INFINITY;
}

// One-token compressor frontier update. Decode appends exactly one projected KV
// row and one score row into a small recurrent state. The generic batch helper
// expresses this as APE copy, score add, and two set_rows operations; this
// kernel writes both state tensors directly while preserving the same
// score + APE arithmetic.
kernel void kernel_dsv4_compressor_store_one(
        constant ds4_metal_args_dsv4_compressor_store_one & args,
        device const float * kv,
        device const float * score,
        device const char  * ape,
        device       float * state_kv,
        device       float * state_score,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.width || args.width == 0 || args.ratio == 0) {
        return;
    }

    const uint pos_mod = args.pos % args.ratio;
    const uint dst_row = args.ratio == 4u ? args.ratio + pos_mod : pos_mod;
    const uint dst = dst_row * args.width + gid;
    const uint ape_i = pos_mod * args.width + gid;

    float ape_v;
    if (args.ape_type == 1u) {
        ape_v = (float)(((device const half *)ape)[ape_i]);
    } else {
        ape_v = ((device const float *)ape)[ape_i];
    }

    state_kv[dst] = kv[gid];
    state_score[dst] = score[gid] + ape_v;
}
struct ds4_metal_args_cpy {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Typed copy/conversion between graph tensors. DS4 uses this for layout
// materialization and F32/F16 conversions at graph boundaries such as KV/cache
// packing and compressor pooling.
template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ds4_metal_args_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;
// Host-visible copy/conversion variants used by the DS4 graph.
template [[host_name("kernel_cpy_f32_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<float, float>;
template [[host_name("kernel_cpy_f32_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<float, half>;
template [[host_name("kernel_cpy_f16_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<half, float>;
template [[host_name("kernel_cpy_f16_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<half, half>;

// Contiguous 1D conversions avoid the generic tensor-index reconstruction
// above. Packed vector types retain scalar alignment, so tensor views whose
// offsets are float/half aligned do not need additional 16/8-byte alignment.
// The final vector is converted element-by-element when n is not divisible by
// four, preserving the generic kernel's bounds and conversion semantics.
kernel void kernel_cpy_contig_f32_f16_4(
        constant uint & n,
        device const packed_float4 * src,
        device       packed_half4  * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        const float4 value = float4(src[gid]);
        dst[gid] = packed_half4(half4(value));
        return;
    }

    device const float * src_scalar = (device const float *)src;
    device       half  * dst_scalar = (device       half  *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = half(src_scalar[i + lane]);
    }
}

kernel void kernel_cpy_contig_f16_f32_4(
        constant uint & n,
        device const packed_half4  * src,
        device       packed_float4 * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        const half4 value = half4(src[gid]);
        dst[gid] = packed_float4(float4(value));
        return;
    }

    device const half  * src_scalar = (device const half  *)src;
    device       float * dst_scalar = (device       float *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = float(src_scalar[i + lane]);
    }
}

// Bitwise F16 transport for cache staging. Use ushort rather than half so NaN
// payloads and every other binary16 encoding pass through unchanged.
kernel void kernel_cpy_contig_f16_f16_bits_4(
        constant uint & n,
        device const packed_ushort4 * src,
        device       packed_ushort4 * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        dst[gid] = src[gid];
        return;
    }

    device const ushort * src_scalar = (device const ushort *)src;
    device       ushort * dst_scalar = (device       ushort *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = src_scalar[i + lane];
    }
}

struct ds4_metal_args_flash_kv_stage_f16 {
    uint raw_cap;
    uint raw_start;
    uint n_raw;
    uint n_comp;
    uint pad_rows;
    uint shared_pad;
};

// Decode-time gathered attention consumes a logical raw-cache ring followed
// by an already-F16 compressed cache. Pack both regions into the contiguous
// F16 FlashAttention scratch in one dispatch. The raw conversion expression
// and compressed ushort4 transport exactly match the standalone copy kernels.
kernel void kernel_dsv4_flash_kv_stage_f16(
        constant ds4_metal_args_flash_kv_stage_f16 & args,
        device const char * raw_src,
        device const char * comp_src,
        device       char * dst,
        device const char * mask_src,
        device       char * pad_dst,
        uint gid [[thread_position_in_grid]]) {
    constexpr uint row_vecs = 128;
    const uint raw_vecs = args.n_raw * row_vecs;
    const uint n_keys = args.n_raw + args.n_comp;
    const uint total_vecs = n_keys * row_vecs;

    if (gid < raw_vecs) {
        const uint logical_row = gid >> 7;
        const uint col = gid & 127u;
        uint physical_row = args.raw_start + logical_row;
        if (physical_row >= args.raw_cap) {
            physical_row -= args.raw_cap;
        }
        device const packed_float4 *raw =
            (device const packed_float4 *)raw_src;
        device packed_half4 *dst_half = (device packed_half4 *)dst;
        const float4 value =
            float4(raw[physical_row * row_vecs + col]);
        dst_half[gid] = packed_half4(half4(value));
        return;
    }

    if (gid < total_vecs) {
        device const packed_ushort4 *comp =
            (device const packed_ushort4 *)comp_src;
        device packed_ushort4 *dst_bits = (device packed_ushort4 *)dst;
        dst_bits[gid] = comp[gid - raw_vecs];
        return;
    }

    // The vector FlashAttention kernel redirects its final partial block to
    // a compact K/V/mask buffer. When requested, append those writes to this
    // dispatch so gathered decode does not need a standalone pad dispatch.
    const uint pad_rows = args.pad_rows;
    const uint pad_vecs = pad_rows * row_vecs;
    uint pad_gid = gid - total_vecs;
    if (pad_gid < pad_vecs) {
        const uint row = pad_gid >> 7;
        const uint col = pad_gid & 127u;
        const uint valid_rows = n_keys % pad_rows;
        device packed_half4 *pad_half = (device packed_half4 *)pad_dst;
        device packed_ushort4 *pad_bits = (device packed_ushort4 *)pad_dst;
        if (row >= valid_rows) {
            const packed_half4 zero = packed_half4(half4(0.0h));
            pad_half[pad_gid] = zero;
            if (!args.shared_pad) {
                pad_half[pad_vecs + pad_gid] = zero;
            }
            return;
        }

        const uint logical_row = n_keys - valid_rows + row;
        if (logical_row < args.n_raw) {
            uint physical_row = args.raw_start + logical_row;
            if (physical_row >= args.raw_cap) {
                physical_row -= args.raw_cap;
            }
            device const packed_float4 *raw =
                (device const packed_float4 *)raw_src;
            const float4 value =
                float4(raw[physical_row * row_vecs + col]);
            const packed_half4 value_half = packed_half4(half4(value));
            pad_half[pad_gid] = value_half;
            if (!args.shared_pad) {
                pad_half[pad_vecs + pad_gid] = value_half;
            }
        } else {
            device const packed_ushort4 *comp =
                (device const packed_ushort4 *)comp_src;
            const packed_ushort4 value_bits =
                comp[(logical_row - args.n_raw) * row_vecs + col];
            pad_bits[pad_gid] = value_bits;
            if (!args.shared_pad) {
                pad_bits[pad_vecs + pad_gid] = value_bits;
            }
        }
        return;
    }

    pad_gid -= pad_vecs;
    if (pad_gid < pad_rows) {
        const uint valid_rows = n_keys % pad_rows;
        device const ushort *mask_bits = (device const ushort *)mask_src;
        device ushort *pad_mask_bits =
            (device ushort *)pad_dst + 2u * pad_vecs * 4u;
        pad_mask_bits[pad_gid] = pad_gid < valid_rows
            ? mask_bits[n_keys - valid_rows + pad_gid]
            : 0xfbffu;
    }
}
#define FC_FLASH_ATTN_EXT_PAD 100
#define FC_FLASH_ATTN_EXT_BLK 200
#define FC_FLASH_ATTN_EXT 300
#define FC_FLASH_ATTN_EXT_VEC 400
#define FC_FLASH_ATTN_EXT_VEC_REDUCE 500
#define OP_FLASH_ATTN_EXT_NQPSG 8
#define OP_FLASH_ATTN_EXT_NCPSG 64
#define OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#define OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#ifndef PAD2
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#endif

template <typename type4>
void dequantize_f32_t4(device const float4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg);

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg);

struct ds4_metal_args_flash_attn_ext_pad {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
};

struct ds4_metal_args_flash_attn_ext_blk {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
};

struct ds4_metal_args_flash_attn_ext {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

struct ds4_metal_args_flash_attn_ext_vec {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

struct ds4_metal_args_flash_attn_ext_vec_reduce {
    int32_t nrows;
};

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];
constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// DS4 FlashAttention padding: pads the final partial K/V/mask cache block so the
// vector FlashAttention kernel can read full 32-row chunks.
kernel void kernel_flash_attn_ext_pad(
        constant ds4_metal_args_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// DS4 FlashAttention mask scan: marks blocks so the non-vector kernel can skip
// blocks that are entirely masked or entirely zero.
kernel void kernel_flash_attn_ext_blk(
        constant ds4_metal_args_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 ? 1 : 0;

    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;
        const int32_t q0 = i1*Q;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            if (q0 + j < args.ne31) {
                device const half * mask_src =
                    (device const half *) (mask + (q0 + j)*args.nb31 + i2*args.nb32 + i3*args.nb33) +
                    i0*C + tiisg;

                FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                    mmin = min(mmin, mask_src[ii*NW]);
                    mmax = max(mmax, mask_src[ii*NW]);
                }
            }
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}

constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];

constant bool FC_flash_attn_ext_bc_mask [[function_constant(FC_FLASH_ATTN_EXT + 10)]];

constant int32_t FC_flash_attn_ext_ns10 [[function_constant(FC_FLASH_ATTN_EXT + 20)]];
constant int32_t FC_flash_attn_ext_ns20 [[function_constant(FC_FLASH_ATTN_EXT + 21)]];
constant int32_t FC_flash_attn_ext_nsg  [[function_constant(FC_FLASH_ATTN_EXT + 22)]];

// DS4 non-vector FlashAttention. The only exported instance uses the model's
// 512-wide F16 K/V rows; keeping the template body generic preserves the same
// arithmetic for dense and compressed-attention prefill.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q,
    short C,
    short NSG>
void kernel_flash_attn_ext_impl(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C;

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV;

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T);
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T);
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK);
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T);
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T);

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            if (FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (FC_flash_attn_ext_bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }
            }

            if (is_same<kd4x4_t, k4x4_t>::value) {
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
                            simdgroup_load(mq, pq + 8*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            } else {
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                if (blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += s2_t(sm2[j*SH + tiisg]);
                    }
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs;
                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[2];

                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
}

// Batched FlashAttention for prompt/prefill rows. It computes QK, applies mask,
// sinks, ALiBi/softcap options when enabled, and multiplies by V without
// materializing the full attention matrix.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q  = OP_FLASH_ATTN_EXT_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_NCPSG>
kernel void kernel_flash_attn_ext(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q, C
#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
    switch (FC_flash_attn_ext_nsg) {
        case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4>(FWD_ARGS); break;
        case 8: kernel_flash_attn_ext_impl<FWD_TMPL, 8>(FWD_ARGS); break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

#define FA_NONVEC_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8

typedef decltype(kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>) flash_attn_ext_dk512_t;
typedef decltype(kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 256, 256>) flash_attn_ext_dk256_t;

// Host-visible prefill FlashAttention variant for DS4's 512-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]
kernel flash_attn_ext_dk512_t kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>;

// Host-visible prefill FlashAttention variant for GLM's 256-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_f16_dk256_dv256")]]
kernel flash_attn_ext_dk256_t kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 256, 256>;

#undef FA_NONVEC_TYPES

constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];
constant bool FC_flash_attn_ext_vec_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 4)]];
constant bool FC_flash_attn_ext_vec_shared_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 5)]];
constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];

// Decode FlashAttention for one query row. DS4 uses this in generation to scan
// raw and compressed KV cache chunks, optionally splitting long contexts across
// workgroups and writing partial softmax state for a later reduction.
template<
    typename q4_t,
    typename k4_t,
    typename v4_t,
    typename qk_t,
    typename s_t,
    typename s4_t,
    typename o4_t,
    typename kd4_t,
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t,
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,
    short DV,
    short NE = 4,
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>
kernel void kernel_flash_attn_ext_vec(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)
#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE;
    constexpr short SH  = 4*C;

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                      0*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + NSG*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + NSG*PK + NSG*SH);

    so4 += tiisg;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    device const float4 * q4 = (device const float4 *) ((device const char *) q);

    if (iq1 < args.ne01) {
        for (short i = tiisg; i < PK4; i += NW) {
            if (i < DK4) {
                sq4[i] = (q4_t) q4[i];
            } else {
                sq4[i] = (q4_t) 0.0f;
            }
        }
    }

    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (o4_t) 0.0f;
    }

    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        device const half * pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        float slope = 1.0f;

        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                if (FC_flash_attn_ext_vec_shared_kvpad) {
                    v = k;
                    mask = k + args.nb11*C*args.ne_12_2*args.ne_12_3 +
                        args.nb21*C*args.ne_12_2*args.ne_12_3;
                } else {
                    v = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                    mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;
                }

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    if (ic + tiisg >= args.ne11) {
                        sm[tiisg] = -MAXHALF;
                    }
                } else {
                    pm = (device const half *) (mask) +
                        iq1*C +
                        (iq2%args.ne32)*(C*args.ne31) +
                        (iq3%args.ne33)*(C*args.ne31*args.ne32);
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                sm[tiisg] = pm[ic + tiisg];
            }

            if (simd_max(sm[tiisg]) <= -MAXHALF) {
                continue;
            }

            {
                device      const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);
                threadgroup const q4_t * pq4 = sq4;

                pk4 += ty*NS10/4 + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 +  ii*NL], (float4) pq4[ii*NL]);
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);
                        }
                    }

                    if (NE == 1) {
                        mqk[cc] = simd_sum(mqk[cc]);
                    } else {
                        if (NE <= 1) {
                            mqk[cc] += simd_shuffle_down(mqk[cc], 16);
                        }
                        if (NE <= 2) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
                        }
                        if (NE <= 4) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
                        }
                        if (NE <= 8) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
                        }
                        if (NE <= 16) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
                        }

                        mqk[cc] = simd_shuffle(mqk[cc], NL*ty);
                    }
                }

                if (FC_flash_attn_ext_vec_has_mask &&
                   !FC_flash_attn_ext_vec_has_scap &&
                   !FC_flash_attn_ext_vec_has_bias) {
                    ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);
                } else {
                    mqk[tx] *= args.scale;

                    if (FC_flash_attn_ext_vec_has_scap) {
                        mqk[tx] = args.logit_softcap*precise::tanh(mqk[tx]);
                    }

                    if (FC_flash_attn_ext_vec_has_bias) {
                        mqk[tx] += (qk_t) sm[NE*tx + ty]*slope;
                    } else {
                        mqk[tx] += (qk_t) sm[NE*tx + ty];
                    }

                    ss[NE*tx + ty] = mqk[tx];
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                ss[tiisg] = vs;

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    const auto sst = ss + ty;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            lo[ii] += o4_t(float4(mv)*float4(ss[NE*cc + ty]));
                        }
                    }
                }

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    if (NE > 1) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);
                    }

                    if (NE > 2) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);
                    }

                    if (NE > 4) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  4);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  4);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  4);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  4);
                    }

                    if (NE > 8) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  2);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  2);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  2);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  2);
                    }

                    if (NE > 16) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  1);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  1);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  1);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  1);
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] += lo[ii];
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            const float m = M;
            const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

            M = simd_max(max(M, s));

            const float ms = exp(m - M);
            const float vs = exp(s - M);

            S = S*ms + simd_sum(vs);

            if ((DV4/NL % NW == 0) || ty == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    so4[ii*NL] *= ms;
                }
            }
        }

        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[           0];
            const float S1 = ss[r*(SH/2) + 0];

            const float M0 = ss[           1];
            const float M1 = ss[r*(SH/2) + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            for (short i = tiisg; i < DV4; i += NW) {
                so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid   = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

        const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;

        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i]*S;
        }

        if (NWG > 1) {
            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

// Host-visible decode FlashAttention variant for DS4's 512-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,  1, dequantize_f16_t4, half4,  1, dequantize_f16_t4, 512, 512, 1>;

#undef FA_TYPES
#undef FA_TYPES_F32

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

// Reduces split-K decode FlashAttention partials. It combines each workgroup's
// output vector and softmax (sum,max) pair into the final attention result.
kernel void kernel_flash_attn_ext_vec_reduce(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}

#ifdef RUST_STAR_ATTENTION_INGRESS
// Decode-only DS4 specialization that folds the immediately following inverse
// partial-RoPE pass into split-K reduction. The reduction statements above are
// intentionally kept unchanged. `dst` retains the pre-RoPE diagnostic boundary;
// `rope_dst` may alias it in production or point at the separately captured
// post-RoPE boundary in probes.
kernel void kernel_flash_attn_ext_vec_reduce_inverse_rope(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        device        char * rope_dst,
        constant ds4_metal_args_dsv4_rope_tail & rope_args,
        device  const char * position_bytes,
        uint   tgpig[[threadgroup_position_in_grid]],
        uint   tid[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;
    const short iwg = tiisg;
    device const float * ss = (device const float *) htmp + (uint64_t)args.nrows*DV*NWG;
    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];
    const float m = simd_max(M);
    const float ms = exp(M - m);
    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;
    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device float4 * dst4 = (device float4 *) dst + rid*DV4;
    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);
        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    if (tid >= 256 || rope_args.mode != 0 || rope_args.ne00 != DV) {
        return;
    }
    const int n_nope = rope_args.ne00 - rope_args.n_dims;
    device const float * src = (device const float *) dst + rid*DV;
    device float * out = (device float *) rope_dst + rid*DV;
    device const int32_t * pos = (device const int32_t *)position_bytes;
    float corr_dims[2];
    rope_yarn_corr_dims(rope_args.n_dims, rope_args.n_ctx_orig, rope_args.freq_base,
                        rope_args.beta_fast, rope_args.beta_slow, corr_dims);
    const float theta_base = (float)pos[0];
    const float inv_ndims = -1.f/rope_args.n_dims;
    for (int i0 = tid; i0 < rope_args.ne00; i0 += 256) {
        if (i0 < n_nope) {
            out[i0] = src[i0];
            continue;
        }
        const int r = i0 - n_nope;
        if ((r & 1) != 0) {
            continue;
        }
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(rope_args.freq_base));
#else
        const float theta = theta_base * pow(rope_args.freq_base, inv_ndims*r);
#endif
        float cos_theta;
        float sin_theta;
        rope_yarn(theta, rope_args.freq_scale, corr_dims, r, rope_args.ext_factor,
                  rope_args.attn_factor, &cos_theta, &sin_theta);
        if (rope_args.inverse) {
            sin_theta = -sin_theta;
        }
        const int j0 = n_nope + r;
        const int j1 = j0 + 1;
        const float x0 = src[j0];
        const float x1 = src[j1];
        out[j0] = x0*cos_theta - x1*sin_theta;
        out[j1] = x0*sin_theta + x1*cos_theta;
    }

#undef NWG
#undef DV
}
#endif
// DS4 Metal softmax kernel used by the compressor pooling compatibility path.
// The single-compressed-row path is intentionally left as soft_max -> mul ->
// sum_rows instead of using the fused dsv4_softmax_pool kernel.

struct ds4_metal_args_soft_max {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
};

// Row softmax for score matrices. DS4 uses it in the literal one-compressor-row
// path where preserving the original graph operation boundary avoids drift.
template<typename T>
kernel void kernel_soft_max(
        constant ds4_metal_args_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

// Vectorized float4 row softmax for contiguous score rows whose length is a
// multiple of four; used by the same DS4 compressor/indexer graph path.
template<typename T>
kernel void kernel_soft_max_4(
        constant ds4_metal_args_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

// Host-visible F32 softmax variants used by compressor pooling.
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;
// DS4 Metal row-sum kernel.

#define FC_SUM_ROWS 1400

#define OP_SUM_ROWS_NUM_SUM_ROWS 10
#define OP_SUM_ROWS_NUM_MEAN     11

struct ds4_metal_args_sum_rows {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

static inline float sum(float x) {
    return x;
}

static inline float sum(float4 x) {
    return x[0] + x[1] + x[2] + x[3];
}

constant short FC_sum_rows_op [[function_constant(FC_SUM_ROWS + 0)]];

// Reduces each row to a sum or mean. DS4 mainly uses the sum form to preserve
// the compressor-pooling graph boundary in the single-compressor-row case.
template <typename T0, typename T>
kernel void kernel_sum_rows_impl(
        constant ds4_metal_args_sum_rows & args,
        device const char * src0,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP  FC_sum_rows_op

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    threadgroup T0 * shmem_t = (threadgroup T0 *) shmem;

    if (sgitg == 0) {
        shmem_t[tiisg] = 0.0f;
    }

    device const T0 * src_row = (device const T0 *) (src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device       T  * dst_row = (device       T  *) (dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);

    T0 sumf = T0(0.0f);

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        sumf += src_row[i0];
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_t[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_t[tiisg];
    sumf = simd_sum(sumf);

    if (tpitg.x == 0) {
        if (FC_OP == OP_SUM_ROWS_NUM_MEAN) {
            if (is_same<float4, T0>::value) {
                dst_row[0] = sum(sumf) / (4*args.ne00);
            } else {
                dst_row[0] = sum(sumf) / args.ne00;
            }
        } else {
            dst_row[0] = sum(sumf);
        }
    }

#undef FC_OP
}

typedef decltype(kernel_sum_rows_impl<float, float>) kernel_sum_rows_t;

// Host-visible F32 row reduction used by compressor pooling.
template [[host_name("kernel_sum_rows_f32_f32")]] kernel kernel_sum_rows_t kernel_sum_rows_impl<float, float>;
// DS4 non-vector FlashAttention. The only exported instance uses the model's
// 512-wide F16 K/V rows; keeping the template body generic preserves the same
// arithmetic for dense and compressed-attention prefill.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q,
    short C,
    short NSG>
void rust_star_compressed_prefill_impl(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C;

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV;

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T);
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T);
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK);
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T);
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T);

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = 0; ; ++ic0) {
            // Skip only complete zero-marked blocks. Leave the partial padded
            // tail on the original path, and preserve every active-block step.
            if (FC_flash_attn_ext_has_mask) {
                const int nfull = args.ne11/C;
                const int first = ic0;
                while (ic0 + 4 <= nfull &&
                       (blk[ic0] | blk[ic0 + 1] | blk[ic0 + 2] | blk[ic0 + 3]) == 0) {
                    ic0 += 4;
                }
                while (ic0 < nfull && blk[ic0] == 0) {
                    ++ic0;
                }
                if (ic0 != first) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += (ic0 - first)*NW;
                    }
                }
            }
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            if (FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (FC_flash_attn_ext_bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }
            }

            if (is_same<kd4x4_t, k4x4_t>::value) {
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
                            simdgroup_load(mq, pq + 8*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            } else {
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                if (blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += s2_t(sm2[j*SH + tiisg]);
                    }
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs;
                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[2];

                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
}

// Batched FlashAttention for prompt/prefill rows. It computes QK, applies mask,
// sinks, ALiBi/softcap options when enabled, and multiplies by V without
// materializing the full attention matrix.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q  = OP_FLASH_ATTN_EXT_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_NCPSG>
kernel void rust_star_compressed_prefill_ext(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q, C
#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
    switch (FC_flash_attn_ext_nsg) {
        case 4: rust_star_compressed_prefill_impl<FWD_TMPL, 4>(FWD_ARGS); break;
        case 8: rust_star_compressed_prefill_impl<FWD_TMPL, 8>(FWD_ARGS); break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

#define FA_NONVEC_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8

typedef decltype(rust_star_compressed_prefill_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>) rust_star_compressed_prefill_512_t;
typedef decltype(rust_star_compressed_prefill_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 256, 256>) rust_star_compressed_prefill_256_t;

// Host-visible prefill FlashAttention variant for DS4's 512-wide F16 K/V rows.
template [[host_name("rust_star_compressed_prefill")]]
kernel rust_star_compressed_prefill_512_t rust_star_compressed_prefill_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>;

// Host-visible prefill FlashAttention variant for GLM's 256-wide F16 K/V rows.
template [[host_name("rust_star_compressed_prefill_256")]]
kernel rust_star_compressed_prefill_256_t rust_star_compressed_prefill_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 256, 256>;

#undef FA_NONVEC_TYPES
