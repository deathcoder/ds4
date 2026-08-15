#include <metal_stdlib>
using namespace metal;

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
    sumf = simd_sum(shmem_f32[tiisg]);
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
};

// Reproduces the two strided concat copies that precede the legacy compressor
// softmax graph. Output is [head_dim, 8] with the eight values contiguous.
kernel void rust_star_compressor_pack_ratio4_one(
        constant rust_star_compressor_pack_args & args,
        device const float * state_kv,
        device const float * state_score,
        device float * packed_kv,
        device float * packed_score,
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.head_dim*8u;
    if (gid >= total) return;
    const uint col = gid/8u;
    const uint row = gid - col*8u;
    const uint src = row < 4u
        ? row*args.width + col
        : row*args.width + args.head_dim + col;
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
