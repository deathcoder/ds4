#include <metal_stdlib>
using namespace metal;

#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

struct ds4_metal_args_mul_mv {
    int ne00; int ne01; int ne02;
    ulong nb00; ulong nb01; ulong nb02; ulong nb03;
    int ne10; int ne11; int ne12;
    ulong nb10; ulong nb11; ulong nb12; ulong nb13;
    int ne0; int ne1; int nr0; short r2; short r3;
};

struct ds4_metal_args_mul_mv_id {
    int nei0; int nei1; ulong nbi1;
    int ne00; int ne01; int ne02;
    ulong nb00; ulong nb01; ulong nb02;
    int ne10; int ne11; int ne12; int ne13;
    ulong nb10; ulong nb11; ulong nb12;
    int ne0; int ne1; ulong nb1; int nr0;
    int tp_rank; int tp_world; int tp_addend; int tp_expert_base;
};

struct ds4_metal_args_dsv4_hc_expand {
    long n_embd; long n_hc; long n_tokens;
    ulong nb_block0; ulong nb_block1;
    ulong nb_add0; ulong nb_add1;
    ulong nb_res0; ulong nb_res1; ulong nb_res2;
    ulong nb_post0; ulong nb_post1;
    ulong nb_comb0; ulong nb_comb1; ulong nb_comb2;
    ulong nb0; ulong nb1; ulong nb2;
    int has_add;
};

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;
    threadgroup float * shmem_f32[NR0];
    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW*row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) dst_f32[r0 + row] = tot;
    }
}

template<short NR0, typename args_t>
static inline void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device char * dst,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    const int nb = args.ne00/QK8_0;
    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const float * y = (device const float *)(src1 + offset1);
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        ax[row] = (device const block_q8_0 *)(src0 + offset0);
    }
    float sumf[NR0] = { 0.f };
    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int ib0 = sgitg*NQ + ix;
    float yl[NQ];
    device const float * yb = y + ib0*QK8_0 + il*NQ;
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            float sumq = 0.f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) sumq += qs[i] * yl[i];
            sumf[row] += sumq*ax[row][ib].d;
        }
        yb += NSG*NQ*QK8_0;
    }
    device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// Imported from DwarfStar metal/moe.metal. The fixed group id is encoded in
// tgpig.z, preserving the release path's grouped block-diagonal projection.
kernel void kernel_dsv4_attn_out_low_q8_0_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device char * dst,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;
    const int iid1 = tgpig.z/args.nei0;
    const int idx = tgpig.z%args.nei0;
    tgpig.z = 0;
    const int64_t i11 = idx%args.ne11;
    const int64_t i12 = iid1;
    device const char * src0_cur = src0s + idx*args.nb02;
    device const char * src1_cur = src1 + i11*args.nb11 + i12*args.nb12;
    device char * dst_cur = dst + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);
    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &>(
        args0, src0_cur, src1_cur, dst_cur, shmem, tgpig, tiisg, sgitg);
}

// Imported from DwarfStar metal/dsv4_hc.metal. It keeps the output Q8 dot
// reduction and HC post-update in the same release kernel.
kernel void kernel_dsv4_q8_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device const char * weight,
        device const char * input,
        device char * block_out,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device char * dst,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) return;
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;
    const int nb = mv.ne00/QK8_0;
    const int row0 = tgpig.x*NR0;
    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int ib0 = sgitg*NQ + ix;
    device const float * y = (device const float *)input;
    device const float * yb = y + ib0*QK8_0 + il*NQ;
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row)*mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }
    float sumf[NR0] = { 0.0f };
    float yl[NQ];
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) yl[i] = yb[i];
        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) sumq += qs[i]*yl[i];
            sumf[row] += sumq*ax[row][ib].d;
        }
        yb += NSG*NQ*QK8_0;
    }
    threadgroup float * shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW*row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) continue;
        const float block_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(block_out + (uint64_t)d*sizeof(float))) = block_v;
            const float r0 = *((device const float *)(residual + (uint64_t)d*hc.nb_res0 + 0*hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d*hc.nb_res0 + 1*hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d*hc.nb_res0 + 2*hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d*hc.nb_res0 + 3*hc.nb_res1));
            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc*hc.nb_post0));
                acc += *((device const float *)(comb + dst_hc*hc.nb_comb0 + 0*hc.nb_comb1))*r0;
                acc += *((device const float *)(comb + dst_hc*hc.nb_comb0 + 1*hc.nb_comb1))*r1;
                acc += *((device const float *)(comb + dst_hc*hc.nb_comb0 + 2*hc.nb_comb1))*r2;
                acc += *((device const float *)(comb + dst_hc*hc.nb_comb0 + 3*hc.nb_comb1))*r3;
                *((device float *)(dst + (uint64_t)d*hc.nb0 + dst_hc*hc.nb1)) = acc;
            }
        }
    }
}
