#include <metal_stdlib>
#ifdef DS4_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;
#ifdef DS4_METAL_HAS_TENSOR
using namespace mpp::tensor_ops;
#endif

// Exact standalone prefix used by DwarfStar before concatenating its Metal
// source files. Only the diagnostic page-touch kernel is omitted.
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};

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

// DS4 Metal matvec kernels used by generation.

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

struct ds4_metal_args_mul_mv {
    int ne00;
    int ne01;
    int ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int ne10;
    int ne11;
    int ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int ne0;
    int ne1;
    int nr0;
    short r2;
    short r3;
};

struct ds4_metal_args_compressor_pair_store {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
};

struct ds4_metal_args_mul_mm {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

struct ds4_metal_args_mul_mv_ext {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
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
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
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

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// Decode-time Q8_0 matrix-vector multiply. DS4 uses this for Q8_0 dense
// projections such as shared experts and output-side small matvecs.
[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_r4")]]
kernel void kernel_mul_mv_q8_0_f32_r4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<4, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// Output projection alias used by the optimized host dispatch.
[[host_name("kernel_mul_mv_q8_0_f32_nr4")]]
kernel void kernel_mul_mv_q8_0_f32_nr4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<4, constant ds4_metal_args_mul_mv &>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// Decode Q-A/KV pair. Both projections consume the same activation row but
// have independent weight ranges and output extents. Keep the standalone Q8_0
// lane/block traversal and two-stage reduction verbatim for each bank; only
// the activation load and threadgroup scheduling are shared.
[[host_name("kernel_mul_mv_q8_0_f32_pair")]]
kernel void kernel_mul_mv_q8_0_f32_pair(
        constant ds4_metal_args_mul_mv & args0,
        constant ds4_metal_args_mul_mv & args1,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = args0.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const bool active_a = r0 < args0.ne01;
    const bool active_b = r0 < args1.ne01;

    device const float *y = (device const float *)src1;
    device const block_q8_0 *ax_a[NR0];
    device const block_q8_0 *ax_b[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int out_row = r0 + row;
        ax_a[row] = active_a && out_row < args0.ne01
            ? (device const block_q8_0 *)(src0_a + (uint64_t)out_row * args0.nb01)
            : (device const block_q8_0 *)src0_a;
        ax_b[row] = active_b && out_row < args1.ne01
            ? (device const block_q8_0 *)(src0_b + (uint64_t)out_row * args1.nb01)
            : (device const block_q8_0 *)src0_b;
    }

    float suma[NR0] = { 0.f };
    float sumb[NR0] = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const int out_row = r0 + row;
            if (active_a && out_row < args0.ne01) {
                device const int8_t *qs = ax_a[row][ib].qs + il * NQ;
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[i];
                }
                suma[row] += sumq * ax_a[row][ib].d;
            }
            if (active_b && out_row < args1.ne01) {
                device const int8_t *qs = ax_b[row][ib].qs + il * NQ;
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[i];
                }
                sumb[row] += sumq * ax_b[row][ib].d;
            }
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shared = (threadgroup float *)shmem;
    threadgroup float *sha[NR0];
    threadgroup float *shb[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sha[row] = shared + NW * row;
        shb[row] = shared + NW * (NR0 + row);
        if (sgitg == 0) {
            sha[row][tiisg] = 0.0f;
            if (active_b) shb[row][tiisg] = 0.0f;
        }
        suma[row] = simd_sum(suma[row]);
        if (active_b) sumb[row] = simd_sum(sumb[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sha[row][sgitg] = suma[row];
            if (active_b) shb[row][sgitg] = sumb[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out_a = (device float *)dst_a;
    device float *out_b = (device float *)dst_b;
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const float total_a = simd_sum(sha[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const int out_row = r0 + row;
            if (active_a && out_row < args0.ne01) out_a[out_row] = total_a;
        }
        if (active_b) {
            const float total_b = simd_sum(shb[row][tiisg]);
            if (tiisg == 0 && sgitg == 0) {
                const int out_row = r0 + row;
                if (out_row < args1.ne01) out_b[out_row] = total_b;
            }
        }
    }
}

// Decode shared-expert gate/up projections followed by SwiGLU:
//
//     mid = silu(min(gate, limit)) * clamp(up, -limit, limit)
//
// DS4's shared expert uses two Q8_0 matrices with the same input row.  This
// kernel preserves the exact Q8_0 dot-product reduction shape for both
// projections, still writes gate/up for diagnostics, and derives `mid` in the
// same lane that owns the reduced output row.  The point is not to fuse two
// independent weight streams into one matmul; it is to remove the separate
// activation pass and its reread of the two 2048-wide rows.
template<short NR0, bool STORE_GATE_UP>
void kernel_dsv4_shared_gate_up_swiglu_q8_0_impl(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q8_0 *ag[NR0];
    device const block_q8_0 *au[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
                                 (i12 / args.r2) * args.nb02 +
                                 (i13 / args.r3) * args.nb03;
        ag[row] = (device const block_q8_0 *)((device const char *)src0_gate + offset0);
        au[row] = (device const block_q8_0 *)((device const char *)src0_up   + offset0);
    }

    float sumg[NR0] = { 0.f };
    float sumu[NR0] = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const int8_t *qg = ag[row][ib].qs + il * NQ;
            device const int8_t *qu = au[row][ib].qs + il * NQ;

            float sg = 0.f;
            float su = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sg += qg[i] * yl[i];
                su += qu[i] * yl[i];
            }

            sumg[row] += sg * ag[row][ib].d;
            sumu[row] += su * au[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32 = (threadgroup float *)shmem;
    threadgroup float *sh_gate[NR0];
    threadgroup float *sh_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sh_gate[row] = shmem_f32 + NW * row;
        sh_up[row]   = shmem_f32 + NW * (NR0 + row);
        if (sgitg == 0) {
            sh_gate[row][tiisg] = 0.0f;
            sh_up[row][tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sh_gate[row][sgitg] = sumg[row];
            sh_up[row][sgitg] = sumu[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *gate_f32 = (device float *)dst_gate +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *up_f32 = (device float *)dst_up +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *mid_f32 = (device float *)dst_mid +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    FOR_UNROLL (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        const float gate = simd_sum(sh_gate[row][tiisg]);
        const float up = simd_sum(sh_up[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const uint out_row = r0 + row;
            if (STORE_GATE_UP) {
                gate_f32[out_row] = gate;
                up_f32[out_row] = up;
            }
            float g = gate;
            float u = up;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u;
        }
    }
}

[[host_name("kernel_dsv4_shared_gate_up_swiglu_q8_0")]]
kernel void kernel_dsv4_shared_gate_up_swiglu_q8_0(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_dsv4_shared_gate_up_swiglu_q8_0_impl<N_R0_Q8_0, true>(
            args, src0_gate, src0_up, src1, dst_gate, dst_up, dst_mid,
            clamp_value, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_dsv4_shared_gate_up_swiglu_q8_0_r4")]]
kernel void kernel_dsv4_shared_gate_up_swiglu_q8_0_r4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_dsv4_shared_gate_up_swiglu_q8_0_impl<4, true>(
            args, src0_gate, src0_up, src1, dst_gate, dst_up, dst_mid,
            clamp_value, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_dsv4_shared_mid_swiglu_q8_0")]]
kernel void kernel_dsv4_shared_mid_swiglu_q8_0(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_dsv4_shared_gate_up_swiglu_q8_0_impl<N_R0_Q8_0, false>(
            args, src0_gate, src0_up, src1, dst_gate, dst_up, dst_mid,
            clamp_value, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_dsv4_shared_mid_swiglu_q8_0_r4")]]
kernel void kernel_dsv4_shared_mid_swiglu_q8_0_r4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_dsv4_shared_gate_up_swiglu_q8_0_impl<4, false>(
            args, src0_gate, src0_up, src1, dst_gate, dst_up, dst_mid,
            clamp_value, shmem, tgpig, tiisg, sgitg);
}

template<typename T0, typename T1, short NR0, typename args_t>
void kernel_mul_mv_t_t_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 8;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    device const T0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const T0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T1 yl[NF];

    device const T1 * yb = y + (ib0*NB + il*NF);

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T0 * xb = ax[row] + (ib*NB + il*NF);

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF; ++i) {
                sumq += xb[i] * yl[i];
            }

            sumf[row] += sumq;
        }

        yb += NSG*NF*NW;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_t_t_impl<T0, T1, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_t_t_impl<T0, T1, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    }
}

// Decode-time dense F32/F16 matrix-vector multiply. The instantiated kernels
// handle unquantized DS4 weights and activations that are already float rows.
template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_disp<T0, T1, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t<half, half>) mul_mv_t_t;

// Host-visible dense matvec variants used by the graph for F32 and F16 weights.
template [[host_name("kernel_mul_mv_f32_f32")]] kernel mul_mv_t_t kernel_mul_mv_t_t<float, float>;
template [[host_name("kernel_mul_mv_f16_f32")]] kernel mul_mv_t_t kernel_mul_mv_t_t<half,  float>;

template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1  * y  = (device const T1  *) (src1 + offset1);
    device const T14 * y4 = (device const T14 *) (src1 + offset1);

    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax [row] = (device const T0  *) ((device char *) src0 + offset0);
        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_t_t_4_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    };
}

// Vectorized dense matvec using float4/half4 loads. DS4 uses this where the
// inner dimension and alignment make vector loads cheaper than scalar lanes.
template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv_t_t_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4>) mul_mv_t_t_4;

// Host-visible vectorized dense matvec variants for F32 and F16 weights.
template [[host_name("kernel_mul_mv_f32_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4>;
template [[host_name("kernel_mul_mv_f16_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4>;

// DS4 compressor projections always compute two same-shaped F16 matvecs from
// the same normalized activation: one for projected KV and one for pooling
// scores.  This paired variant keeps the exact dense F16 row-reduction shape
// for each matrix, but shares one dispatch and one activation stream.
template<short NR0, typename args_t>
void kernel_mul_mv_f16_f32_pair_4_impl(
        args_t args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float  * y  = (device const float  *) (src1 + offset1);
    device const float4 * y4 = (device const float4 *) (src1 + offset1);

    device const half  * ax_a [NR0];
    device const half4 * ax4_a[NR0];
    device const half  * ax_b [NR0];
    device const half4 * ax4_b[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax_a [row] = (device const half  *) ((device char *) src0_a + offset0);
        ax4_a[row] = (device const half4 *) ((device char *) src0_a + offset0);
        ax_b [row] = (device const half  *) ((device char *) src0_b + offset0);
        ax4_b[row] = (device const half4 *) ((device char *) src0_b + offset0);
    }

    float sum_a[NR0] = { 0.f };
    float sum_b[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    float4 yl4[NF4];

    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

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
            sum_a[row] += ax_a[row][i] * yi;
            sum_b[row] += ax_b[row][i] * yi;
        }
    }

    device float * dst_a_f32 = (device float *) dst_a + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_b_f32 = (device float *) dst_b + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_a_f32, sum_a, r0, args.ne01, tiisg, sgitg, shmem);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    helper_mv_reduce_and_write<NR0>(dst_b_f32, sum_b, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename args_t>
void kernel_mul_mv_f16_f32_pair_4_disp(
        args_t args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_f16_f32_pair_4_impl<2>(args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_f16_f32_pair_4_impl<4>(args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg); break;
    }
}

kernel void kernel_mul_mv_f16_f32_pair_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_f16_f32_pair_4_disp<constant ds4_metal_args_mul_mv &>(
            args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg);
}

// Decode compressor projection plus recurrent-state append. The paired
// matvec remains unchanged and still materializes both F32 outputs. After a
// device-memory barrier, the first NR0 threads reload those exact stored bits
// and perform the same state write and score+APE addition as
// kernel_dsv4_compressor_store_one.
kernel void kernel_mul_mv_f16_f32_pair_compressor_store_4(
        constant ds4_metal_args_mul_mv & args,
        constant ds4_metal_args_compressor_pair_store & store,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        device const char * ape,
        device       float * state_kv,
        device       float * state_score,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_f16_f32_pair_4_disp<constant ds4_metal_args_mul_mv &>(
            args, src0_a, src0_b, src1, dst_a, dst_b,
            shmem, tgpig, tiisg, sgitg);

    threadgroup_barrier(mem_flags::mem_device);

    if (tiitg >= args.nr0 || store.width == 0u || store.ratio == 0u) {
        return;
    }
    const uint col = tgpig.x * (uint)args.nr0 + tiitg;
    if (col >= store.width) return;

    const uint pos_mod = store.pos % store.ratio;
    const uint dst_row = store.ratio == 4u ? store.ratio + pos_mod : pos_mod;
    const uint dst = dst_row * store.width + col;
    const uint ape_i = pos_mod * store.width + col;

    device volatile const float * projected_kv =
            (device volatile const float *)dst_a;
    device volatile const float * projected_score =
            (device volatile const float *)dst_b;
    float ape_v;
    if (store.ape_type == 1u) {
        ape_v = (float)(((device const half *)ape)[ape_i]);
    } else {
        ape_v = ((device const float *)ape)[ape_i];
    }

    state_kv[dst] = projected_kv[col];
    state_score[dst] = projected_score[col] + ape_v;
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_short_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x*32 + tiisg;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    if (r0 >= args.ne01) {
        return;
    }

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    float res = 0.0f;

    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * (float) y[i];
    }

    dst_f32[(uint64_t)r1*args.ne0 + r0] = res;
}

// Scalar fallback for short rows. It trades parallelism for lower dispatch and
// reduction overhead when DS4 asks for tiny dense matvecs.
template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t_short(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_t_t_short_impl<T0, T1, constant ds4_metal_args_mul_mv &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv_t_t_short<half, half>) mul_mv_t_t_short_t;

// Host-visible short-row dense matvec variants.
template [[host_name("kernel_mul_mv_f32_f32_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<float, float>;
template [[host_name("kernel_mul_mv_f16_f32_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  float>;

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

struct ds4_dense_block_q4_0 {
    half d;
    uchar qs[16];
};

struct ds4_dense_block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

static inline uchar2 ds4_dense_q4_K_scale_min(int j, int k, device const uchar *q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0x0f) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_dense_q4_0(device const ds4_dense_block_q4_0 *xb, short il, thread type4x4 &reg) {
    float4x4 reg_f;
    const float d = (float)xb->d;
    const int base = 16 * (int)il;
    for (int i = 0; i < 16; i++) {
        const int k = base + i;
        /* ggml Q4_0: elems 0..15 = low nibbles of qs[0..15], 16..31 = high. */
        const uchar packed = xb->qs[(uint)(k & 15)];
        const uchar q = (k < 16) ? (packed & 0x0f) : (packed >> 4);
        reg_f[i / 4][i % 4] = d * ((float)q - 8.0f);
    }
    reg = (type4x4)reg_f;
}

template <typename type4x4>
void dequantize_dense_q4_K(device const ds4_dense_block_q4_K *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = ds4_dense_q4_K_scale_min(is, il / 2, xb->scales);
    const float d = il < 2 ? (float)xb->d : (float)xb->d * (1.0f / 16.0f);
    const float min = (float)xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;

    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        reg_f[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
    reg = (type4x4)reg_f;
}

/*
 * Bit-identical twin of dequantize_q8_0 for the MPP staging loop: same
 * half(float(qs[i]) * d) per element, but the 16 consecutive int8 lanes are
 * fetched as eight aligned 16-bit loads instead of sixteen byte loads.
 * xb->qs is always 2-byte aligned (block_q8_0.d is a half).
 */
void dequantize_q8_0_pairs(device const block_q8_0 *xb, short il, thread half4x4 & reg) {
    device const ushort *qs16 = (device const ushort *)(xb->qs + 16*il);
    const float d = xb->d;

    float4x4 reg_f;

    FOR_UNROLL (short i = 0; i < 8; i++) {
        const ushort u = qs16[i];
        reg_f[i/2][(i%2)*2 + 0] = ((float)(int8_t)(u & 0xFF)) * d;
        reg_f[i/2][(i%2)*2 + 1] = ((float)(int8_t)(u >> 8)) * d;
    }

    reg = (half4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    for (int i = 0; i < 4; i++) {
        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
    }
}

template <typename type4>
void dequantize_dense_q4_0_t4(device const ds4_dense_block_q4_0 *xb, short il, thread type4 &reg) {
    const float d = (float)xb->d;
    const int base = 4 * (int)il;
    for (int i = 0; i < 4; i++) {
        const int k = base + i;
        /* ggml Q4_0: elems 0..15 = low nibbles of qs[0..15], 16..31 = high. */
        const uchar packed = xb->qs[(uint)(k & 15)];
        const uchar q = (k < 16) ? (packed & 0x0f) : (packed >> 4);
        reg[i] = d * ((float)q - 8.0f);
    }
}

template <typename type4>
void dequantize_dense_q4_K_t4(device const ds4_dense_block_q4_K *xb, short il, thread type4 &reg) {
    float4x4 tmp;
    dequantize_dense_q4_K(xb, il / 4, tmp);
    const short row = il & 3;
    for (int i = 0; i < 4; i++) {
        reg[i] = tmp[row][i];
    }
}

// DS4 small-batch mat-vec kernel used for 2..8 prompt tokens.
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread

    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// Small-batch prompt matvec for 2..5 tokens. It bridges decode-style matvec and
// full matmul when DS4 prefill chunks are too small to amortize matrix tiles.
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;

// Host-visible small-batch variants for r1=2..5 during tiny prompt/support
// paths.
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_2")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, float4,     4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_3")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, float4,     4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_4")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, float4,     4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_5")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, float4,     4,  dequantize_f32_t4>;

template<short r1ptg>
void kernel_mul_mv_ext_q8_0_pair_swiglu_f32_impl(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4;
    const short chpb = 8;
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q8_0 * xq_gate =
        (i01 < args.ne01) ? (device const block_q8_0 *)(src0_gate + offset0) + tx/chpb
                          : (device const block_q8_0 *)src0_gate;
    device const block_q8_0 * xq_up =
        (i01 < args.ne01) ? (device const block_q8_0 *)(src0_up + offset0) + tx/chpb
                          : (device const block_q8_0 *)src0_up;

    device const float4 * y4[r1ptg];
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11)
            ? (device const float4 *)(src1 + offset1 + ir1*args.nb11) + tx
            : (device const float4 *)src1;
    }

    float sum_gate[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };
    float sum_up[r1ptg]   = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb;
    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lg[chpt];
        float4 lu[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            dequantize_q8_0_t4(xq_gate, cch, lg[ch]);
            dequantize_q8_0_t4(xq_up,   cch, lu[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq_gate += cch/chpb;
                xq_up   += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                const float4 y = y4[ir1][ch*nxpsg];
                sum_gate[ir1] += dot(lg[ch], y);
                sum_up[ir1] += dot(lu[ch], y);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1], 16);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],   16);
        }
        if (nxpsg >= 16) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1],  8);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],    8);
        }
        if (nxpsg >= 8) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1],  4);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],    4);
        }
        if (nxpsg >= 4) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1],  2);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],    2);
        }
        if (nxpsg >= 2) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1],  1);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],    1);
        }
    }

    if (tx == 0 && i01 < args.ne01) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            const uint64_t dst_base =
                (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;
            device float * gate_f32 = (device float *)dst_gate + dst_base;
            device float * up_f32   = (device float *)dst_up   + dst_base;
            device float * mid_f32  = (device float *)dst_mid  + dst_base;

            const float gate = sum_gate[ir1];
            const float up = sum_up[ir1];
            gate_f32[i01] = gate;
            up_f32[i01] = up;

            float g = gate;
            float u = up;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[i01] = silu * u;
        }
    }
}

template<short r1ptg>
kernel void kernel_mul_mv_ext_q8_0_pair_swiglu_f32_disp(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q8_0_pair_swiglu_f32_impl<r1ptg>(
            args, src0_gate, src0_up, src1, dst_gate, dst_up, dst_mid, clamp_value,
            tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q8_0_pair_swiglu_f32_disp<2>) mul_mv_ext_q8_0_pair_swiglu_f32_t;

// Host-visible small-batch variants. DS4 currently needs F16 and Q8_0 weights
// for r1=2..5 during the prompt path.
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_2")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_3")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_4")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_5")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, half4,      4,  dequantize_f16_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0, 32, dequantize_q8_0_t4>;

template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_1")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<1, ds4_dense_block_q4_0, 32,  dequantize_dense_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, ds4_dense_block_q4_0, 32,  dequantize_dense_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, ds4_dense_block_q4_0, 32,  dequantize_dense_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, ds4_dense_block_q4_0, 32,  dequantize_dense_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, ds4_dense_block_q4_0, 32,  dequantize_dense_q4_0_t4>;

template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_1")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<1, ds4_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, ds4_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, ds4_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, ds4_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, ds4_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_2")]] kernel mul_mv_ext_q8_0_pair_swiglu_f32_t kernel_mul_mv_ext_q8_0_pair_swiglu_f32_disp<2>;
template [[host_name("kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_3")]] kernel mul_mv_ext_q8_0_pair_swiglu_f32_t kernel_mul_mv_ext_q8_0_pair_swiglu_f32_disp<3>;
template [[host_name("kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_4")]] kernel mul_mv_ext_q8_0_pair_swiglu_f32_t kernel_mul_mv_ext_q8_0_pair_swiglu_f32_disp<4>;
template [[host_name("kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_5")]] kernel mul_mv_ext_q8_0_pair_swiglu_f32_t kernel_mul_mv_ext_q8_0_pair_swiglu_f32_disp<5>;

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];

#ifdef DS4_METAL_HAS_TENSOR
template<
    short NR0, short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    threadgroup SA *sb = sa + NR0*NK;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NK, NR1));

    device const T1 *ptrB = (device const T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (!FC_mul_mm_bc_out || r0 + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 *row_ptr = (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? (SA)row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos/(16*nl);
                    const short il = (k_pos/16)%nl;
                    device const block_q *row_ptr = (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (SA)0;
                }
            }
        }
        for (int work = tiitg; work < NK*NR1; work += NUM_THREADS) {
            const int col = work/NK;
            const int k = work%NK;
            if ((!FC_mul_mm_bc_out && !FC_mul_mm_bc_inp) ||
                (r1 + col < N && loop_k + k < K)) {
                sb[col*NK + k] = (SA)ptrB[(uint64_t)(r1 + col)*strideB + loop_k + k];
            } else {
                sb[col*NK + k] = (SA)0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(0, 0);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    if (!FC_mul_mm_bc_out) {
        device float *dst_tile = dst_batch + r0 + (uint64_t)r1*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

typedef decltype(kernel_mul_mm_mpp<64, 32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp")]]  kernel mul_mm_mpp_t kernel_mul_mm_mpp<64, 32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;

// Retained Metal4/TensorOps dense prefill kernel.  The legacy MPP prototype
// staged both operands in threadgroup memory; this version stages only the
// model weight tile and lets MPP read the dense RHS activation matrix directly
// from device memory.  That direct-RHS shape was the clear win for DS4's large
// aligned F16/Q8_0 prompt matmuls.  The host selects the widest token tile that
// evenly divides the batch, with 128-token tiles retained after the 64-token
// retest was neutral or slower.
//
// The host dispatch guarantees M % NR0 == 0 and K % NK == 0, so the dequant
// stage does no bounds work.  The weight tile is double-buffered: the next
// k-step's dequant overlaps the current cooperative matmul, so the k-loop
// needs one threadgroup barrier per step instead of two.
template<
    short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp_direct_rhs(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    auto tA0 = tensor(sa,          dextents<int32_t, 2>(NK, NR0));
    auto tA1 = tensor(sa + NR0*NK, dextents<int32_t, 2>(NK, NR0));

    device T1 *ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA0), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    // NR0*NL/NUM_THREADS 16-value weight chunks per thread (1 at NR0=64).
    auto stage_tile = [&](const int loop_k, threadgroup SA *buf) {
        FOR_UNROLL (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work / NL;
            const int k_chunk = work % NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;
            if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                device const T0 *row_ptr_f =
                    (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    buf[row*NK + k_base + i] = (SA)row_ptr_f[k_pos + i];
                }
            } else {
                device const block_q *row_ptr =
                    (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);
                SA_4x4 temp_a;
                dequantize_func(row_ptr + k_pos/(16*nl), (k_pos/16)%nl, temp_a);
                typedef vec<SA, 4> SA4;
                threadgroup SA4 *dst4 = (threadgroup SA4 *)(buf + row*NK + k_base);
                dst4[0] = temp_a[0];
                dst4[1] = temp_a[1];
                dst4[2] = temp_a[2];
                dst4[3] = temp_a[3];
            }
        }
    };

    stage_tile(0, sa);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_sel = 0;
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        auto mA = (buf_sel ? tA1 : tA0).slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        const int next_k = loop_k + NK;
        if (next_k < K) {
            buf_sel ^= 1u;
            stage_tile(next_k, buf_sel ? sa + NR0*NK : sa);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

typedef decltype(kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_direct_rhs_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n64")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n128")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_q4_0_f32_nax_direct_rhs")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, ds4_dense_block_q4_0, 2, dequantize_dense_q4_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q4_0_f32_nax_direct_rhs_n64")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, ds4_dense_block_q4_0, 2, dequantize_dense_q4_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q4_0_f32_nax_direct_rhs_n128")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, ds4_dense_block_q4_0, 2, dequantize_dense_q4_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q4_K_f32_nax_direct_rhs")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, ds4_dense_block_q4_K, 16, dequantize_dense_q4_K, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q4_K_f32_nax_direct_rhs_n64")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, ds4_dense_block_q4_K, 16, dequantize_dense_q4_K, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q4_K_f32_nax_direct_rhs_n128")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, ds4_dense_block_q4_K, 16, dequantize_dense_q4_K, float, float4x4, float>;

template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, block_q8_0, 2, dequantize_q8_0_pairs, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, block_q8_0, 2, dequantize_q8_0_pairs, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, block_q8_0, 2, dequantize_q8_0_pairs, float, float4x4, float>;
#endif

// Tiled matrix-matrix kernel used for prompt batches larger than 8. DS4 uses
// this to turn prefill into large simdgroup matrix operations; each block_q
// contains 16*nl weights.
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // Pointer-form store avoids a slower address-lowering path in
                // current Apple Metal compilers for this dequantized tile write.
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

// Legacy F16-weight/F32-RHS prefill matmul with a per-row RMS scale applied at
// the existing F32-to-F16 RHS staging boundary.  The tile layout, half inputs,
// float accumulators, and output path intentionally mirror kernel_mul_mm so the
// only arithmetic change versus materializing RMSNorm first is where x*scale is
// rounded from F32 to F16.
kernel void kernel_mul_mm_f16_f32_scaled(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const float * scales,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0;

    device const half4x4 * x = (device const half4x4 *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);
    const float row_scale = scales[r1 + lr1];

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device half *) x + i) : 0;
            }
        } else {
            half4x4 temp_a;
            dequantize_f16(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // Pointer-form store matches the legacy F16 tile layout.
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;
                const float scaled = loop_k + iy + i < args.ne00 ? y[i] * row_scale : 0.0f;

                *(sb + 64*ib + 8*ly + lx) = (half)scaled;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            const float2x4 raw = *((device const float2x4 *) y);
            float2x4 scaled;
            scaled[0] = raw[0] * row_scale;
            scaled[1] = raw[1] * row_scale;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)scaled;
        }

        il = (il + 2 < 1) ? il + 2 : il % 2;
        x  = (il < 2) ? x + 2 : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

kernel void kernel_mul_mm_f16_f32_pair(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_a = (threadgroup half *)(shmem);
    threadgroup half * sa_b = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0;

    device const half4x4 * xa = (device const half4x4 *)(src0_a + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const half4x4 * xb = (device const half4x4 *)(src0_b + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc_a[8];
    simdgroup_float8x8 mc_b[8];

    for (short i = 0; i < 8; i++) {
        mc_a[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_b[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_a;
        half4x4 temp_b;
        dequantize_f16(xa, il, temp_a);
        dequantize_f16(xb, il, temp_b);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;

            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;

            const short ib = 8*sx + sy;

            *(sa_a + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            *(sa_b + 64*ib + 8*ly + lx) = temp_b[i/4][i%4];
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (half) *((device float *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)(*((device float2x4 *) y));
        }

        il = (il + 2 < 1) ? il + 2 : il % 2;
        xa = (il < 2) ? xa + 2 : xa;
        xb = (il < 2) ? xb + 2 : xb;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_a = (sa_a + 4*64*(sgitg%2));
        threadgroup const half * lsma_b = (sa_b + 4*64*(sgitg%2));
        threadgroup const half * lsmb   = (sb   + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_a + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_a[i], mb[i/4], ma[i%4], mc_a[i]);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_b + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_b[i], mb[i/4], ma[i%4], mc_b[i]);
            }

            lsma_a += 8*64;
            lsma_b += 8*64;
            lsmb   += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        device float * C_a = (device float *) dst_a +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;
        device float * C_b = (device float *) dst_b +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_a[i], C_a + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
            simdgroup_store(mc_b[i], C_b + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = (threadgroup float *) shmem;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_a[i],
                            temp_str + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0 + 8*(i%4) + 8*NR0*(i/4),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *) dst_a + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_b[i],
                            temp_str + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0 + 8*(i%4) + 8*NR0*(i/4),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *) dst_b + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

// Host-visible prefill matmul variants for F16 and Q8_0 weights.
template [[host_name("kernel_mul_mm_f16_f32")]]  kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, half4x4, 1, dequantize_f16,  half,  half4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, ds4_dense_block_q4_0, 2, dequantize_dense_q4_0, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, ds4_dense_block_q4_K, 16, dequantize_dense_q4_K, float, float4x4, float, float2x4>;

struct ds4_metal_args_glu {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
};

// SwiGLU activation for the FFN inner state. DS4 clamps the shared expert with
// the same swiglu_limit used by routed experts.
kernel void kernel_swiglu_f32(
        constant ds4_metal_args_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        float x0 = src0_row[i0];
        float x1 = src1_row[i0];
        if (args.limit > 1.0e-6f) {
            x0 = min(x0, args.limit);
            x1 = clamp(x1, -args.limit, args.limit);
        }

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = silu*x1*args.alpha;
    }
}

kernel void kernel_swiglu_flat_f32(
        constant ds4_metal_args_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint i [[thread_position_in_grid]]) {
    if (i >= (uint)args.ne0) return;

    device const float * src0_f32 = (device const float *) src0 + args.i00;
    device const float * src1_f32 = (device const float *) src1 + args.i10;
    device       float * dst_f32  = (device       float *) dst;

    float x0 = src0_f32[i];
    float x1 = src1_f32[i];
    if (args.limit > 1.0e-6f) {
        x0 = min(x0, args.limit);
        x1 = clamp(x1, -args.limit, args.limit);
    }

    const float silu = x0 / (1.0f + exp(-x0));
    dst_f32[i] = silu*x1*args.alpha;
}

// DS4 Metal routed-MoE matvec kernels.

#ifndef QK_K
#define QK_K 256
#endif
#define QK_MXFP4 32
#define N_R0_Q2_K 4
#define N_R0_GLM_Q2_PAIR2_K 1
#define N_R0_Q4_K 2
#define N_R0_Q8_K 2
#define N_R0_GLM_Q4_PAIR2_K 1
#define N_R0_GLM_Q4_PAIR_K 4
#define N_R0_Q5_PAIR_K 4
#define N_R0_Q5_K 4
#define N_R0_Q6_K 2
#define N_R0_IQ2_XXS 4
#define N_R0_MXFP4 2

static constant float ds4_metal_mxfp4_values[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static inline float ds4_metal_e8m0_to_f32(uchar e) {
    const uint bits = e == 0 ? 0x00400000u : (uint)e << 23;
    return as_type<float>(bits);
}

static constant uchar ds4_metal_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static constant uchar ds4_metal_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant ulong ds4_metal_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

#define kmask_iq2xs ds4_metal_kmask_iq2xs
#define ksigns_iq2xs ds4_metal_ksigns_iq2xs
#define iq2xxs_grid ds4_metal_iq2xxs_grid

struct block_q2_K {
    uchar scales[QK_K/16];
    uchar qs[QK_K/4];
    half d;
    half dmin;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K/2];
};

struct block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[QK_K/8];
    uchar qs[QK_K/2];
};

struct block_q6_K {
    uchar ql[QK_K/2];
    uchar qh[QK_K/4];
    char scales[QK_K/16];
    half d;
};

struct block_iq2_xxs {
    half d;
    ushort qs[QK_K/8];
};

struct block_mxfp4 {
    uchar e;
    uchar qs[QK_MXFP4/2];
};

struct ds4_metal_glm_routed_moe_args {
    uint32_t in_dim;
    uint32_t mid_dim;
    uint32_t out_dim;
    uint32_t n_total_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
    uint32_t mid_token_stride;
    uint32_t down_type;
    /* Expert ownership under tensor parallelism: tp_world 0/1 = full
     * compute; otherwise each rank owns a contiguous expert range. */
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_expert_base;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_expert_bytes;
    uint64_t up_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
};


static inline bool ds4_tp_owns_expert(int expert, int n_total,
                                      int tp_rank, int tp_world) {
    if (tp_world <= 1) return true;
    const int first = tp_rank * (n_total / tp_world);
    const int last = tp_rank + 1 == tp_world ? n_total : first + n_total / tp_world;
    return expert >= first && expert < last;
}

struct ds4_metal_dsv4_moe_swiglu_weight_args {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

struct ds4_metal_dsv4_moe_sum6_args {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
};

// Routed-MoE activation for the selected experts:
// clamp(gate), clamp(up), silu(gate) * up * route_weight.  Normal inference
// does not consume gate/up after this point, so the fast path avoids writing the
// clamped intermediates back.  A diagnostic env switch can restore those writes
// when comparing the old multi-kernel intermediate tensors.
kernel void kernel_dsv4_moe_swiglu_weight(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device float *mid_row  = (device float *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = silu * u * route_weight;
    }
}

// Same routed-MoE activation as above, but stores the down-projection input in
// half precision. The grouped matmul path converts F32 activations to half
// before MMA anyway, so this cuts the large mid write/read traffic without
// changing the effective matmul input precision.
kernel void kernel_dsv4_moe_swiglu_weight_f16(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device half  *mid_row  = (device half  *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = (half)(silu * u * route_weight);
    }
}

kernel void kernel_dsv4_moe_sum6_f32(
        constant ds4_metal_dsv4_moe_sum6_args &args,
        device const char *src,
        device       char *dst,
        uint token[[threadgroup_position_in_grid]],
        uint tid[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (token >= args.tokens) return;

    device const float *s =
        (device const float *)(src + (uint64_t)token * args.src_token_stride);
    device float *d =
        (device float *)(dst + (uint64_t)token * args.dst_token_stride);

    for (uint col = tid; col < args.width; col += ntg) {
        float v = s[col];
        v += s[args.width + col];
        v += s[2u * args.width + col];
        v += s[3u * args.width + col];
        v += s[4u * args.width + col];
        v += s[5u * args.width + col];
        d[col] = v;
    }
}

kernel void kernel_dsv4_moe_sum8_f32(
        constant ds4_metal_dsv4_moe_sum6_args &args,
        device const char *src,
        device       char *dst,
        uint token[[threadgroup_position_in_grid]],
        uint tid[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (token >= args.tokens) return;

    device const float *s =
        (device const float *)(src + (uint64_t)token * args.src_token_stride);
    device float *d =
        (device float *)(dst + (uint64_t)token * args.dst_token_stride);

    for (uint col = tid; col < args.width; col += ntg) {
        float v = s[col];
        v += s[args.width + col];
        v += s[2u * args.width + col];
        v += s[3u * args.width + col];
        v += s[4u * args.width + col];
        v += s[5u * args.width + col];
        v += s[6u * args.width + col];
        v += s[7u * args.width + col];
        d[col] = v;
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

static inline float ds4_glm_q2_K_value(device const block_q2_K *blocks, uint k) {
    const uint block = k / QK_K;
    const uint idx = k - block * QK_K;
    device const block_q2_K *xb = blocks + block;
    const uint group = idx / 16u;
    const uint l = idx - group * 16u;
    const uint q_base = 32u * (group / 8u) + 16u * (group & 1u);
    const uint shift = ((group / 2u) & 3u) * 2u;
    const uint q = ((uint)xb->qs[q_base + l] >> shift) & 0x03u;
    const uint sc = (uint)xb->scales[group];
    return (float)xb->d * (float)(sc & 0x0fu) * (float)q -
           (float)xb->dmin * (float)(sc >> 4u);
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

static inline float ds4_glm_q4_K_value(device const block_q4_K *blocks, uint k) {
    const uint block = k / QK_K;
    const uint idx = k - block * QK_K;
    device const block_q4_K *xb = blocks + block;
    const uint group = idx / 32u;
    const uint l = idx - group * 32u;
    const uchar2 sm = get_scale_min_k4_just2((int)group, 0, xb->scales);
    const uint byte_off = (group >> 1u) * 32u + l;
    const uint shift = (group & 1u) * 4u;
    const uint q = (xb->qs[byte_off] >> shift) & 0x0Fu;
    return (float)xb->d * (float)sm.x * (float)q -
           (float)xb->dmin * (float)sm.y;
}

static inline float ds4_glm_q5_K_value(device const block_q5_K *blocks, uint k) {
    const uint block = k / QK_K;
    const uint idx = k - block * QK_K;
    device const block_q5_K *xb = blocks + block;
    const uint group = idx / 32u;
    const uint l = idx - group * 32u;
    const uchar2 sm = get_scale_min_k4_just2((int)group, 0, xb->scales);
    const uint ql_base = (group >> 1u) * 32u + l;
    const uint shift = (group & 1u) * 4u;
    uint q = (xb->qs[ql_base] >> shift) & 0x0Fu;
    q += (xb->qh[l] & (uchar)(1u << group)) ? 16u : 0u;
    return (float)xb->d * (float)sm.x * (float)q -
           (float)xb->dmin * (float)sm.y;
}

static inline float ds4_glm_q6_K_value(device const block_q6_K *blocks, uint k) {
    const uint block = k / QK_K;
    const uint idx = k - block * QK_K;
    device const block_q6_K *xb = blocks + block;
    const uint n128 = idx >> 7u;
    const uint r = idx & 127u;
    const uint l = r & 31u;
    const uint quarter = r >> 5u;
    const uint ql_base = n128 * 64u;
    const uint qh_base = n128 * 32u;
    const uint sc_base = n128 * 8u;
    uint q;
    int sc;

    if (quarter == 0u) {
        q = (xb->ql[ql_base + l] & 0x0Fu) | (((xb->qh[qh_base + l] >> 0u) & 3u) << 4u);
        sc = (int)xb->scales[sc_base + l / 16u + 0u];
    } else if (quarter == 1u) {
        q = (xb->ql[ql_base + 32u + l] & 0x0Fu) | (((xb->qh[qh_base + l] >> 2u) & 3u) << 4u);
        sc = (int)xb->scales[sc_base + l / 16u + 2u];
    } else if (quarter == 2u) {
        q = (xb->ql[ql_base + l] >> 4u) | (((xb->qh[qh_base + l] >> 4u) & 3u) << 4u);
        sc = (int)xb->scales[sc_base + l / 16u + 4u];
    } else {
        q = (xb->ql[ql_base + 32u + l] >> 4u) | (((xb->qh[qh_base + l] >> 6u) & 3u) << 4u);
        sc = (int)xb->scales[sc_base + l / 16u + 6u];
    }

    return (float)xb->d * (float)sc * (float)((int)q - 32);
}

kernel void kernel_glm_q4_K_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint ntg = 256u;
    const uint row = tgpig.x;
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (row >= args.mid_dim || slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const uint64_t mid_off = (uint64_t)token * args.mid_token_stride +
                             (uint64_t)slot * args.mid_dim + row;
    const int expert = selected[selected_off];
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tid == 0u) mid[mid_off] = 0.0f;
        return;
    }

    device const block_q4_K *gate_row =
        (device const block_q4_K *)(gate +
            (uint64_t)(uint)expert * args.gate_expert_bytes +
            (uint64_t)row * args.gate_row_bytes);
    device const block_q4_K *up_row =
        (device const block_q4_K *)(up +
            (uint64_t)(uint)expert * args.up_expert_bytes +
            (uint64_t)row * args.up_row_bytes);

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    device const float *token_x = x + (uint64_t)token * args.in_dim;
    for (uint k = tid; k < args.in_dim; k += ntg) {
        const float xv = token_x[k];
        acc_gate += ds4_glm_q4_K_value(gate_row, k) * xv;
        acc_up += ds4_glm_q4_K_value(up_row, k) * xv;
    }

    scratch[tid] = acc_gate;
    scratch[ntg + tid] = acc_up;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[ntg + tid] += scratch[ntg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float g = scratch[0];
        const float u = scratch[ntg];
        const float sw = g / (1.0f + exp(-g));
        mid[mid_off] = sw * u * weights[selected_off];
    }
}

template <short N_R0>
static inline void glm_q2_K_pair_swiglu_simd_f32_impl(
        ds4_metal_glm_routed_moe_args args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch,
        uint3 tgpig,
        uint slot,
        uint token,
        uint64_t selected_off,
        int expert,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0;
    if (row0 >= args.mid_dim || slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0 && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    const int nb = args.in_dim / QK_K;
    const uint64_t expert_gate = (uint64_t)(uint)expert * args.gate_expert_bytes;
    const uint64_t expert_up = (uint64_t)(uint)expert * args.up_expert_bytes;
    device const block_q2_K *xg =
        (device const block_q2_K *)(gate + expert_gate +
            (uint64_t)row0 * args.gate_row_bytes);
    device const block_q2_K *xu =
        (device const block_q2_K *)(up + expert_up +
            (uint64_t)row0 * args.up_row_bytes);
    device const float *y = x + (uint64_t)token * args.in_dim;
    device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    float yl[32];
    float sumg[N_R0] = {0.f};
    float sumu[N_R0] = {0.f};

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
            yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
        }

        device const uint8_t *scg = (device const uint8_t *)xg[ib].scales + 8 * iq + is;
        device const uint16_t *qsg = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint8_t *scu = (device const uint8_t *)xu[ib].scales + 8 * iq + is;
        device const uint16_t *qsu = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0;
             row < N_R0 && row0 + (uint)row < args.mid_dim;
             row++) {
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            for (int i = 0; i < 8; i += 2) {
                acc1g[0] += yl[i +  0] * (qsg[i / 2] & 0x0003);
                acc2g[0] += yl[i +  1] * (qsg[i / 2] & 0x0300);
                acc1g[1] += yl[i +  8] * (qsg[i / 2] & 0x000c);
                acc2g[1] += yl[i +  9] * (qsg[i / 2] & 0x0c00);
                acc1g[2] += yl[i + 16] * (qsg[i / 2] & 0x0030);
                acc2g[2] += yl[i + 17] * (qsg[i / 2] & 0x3000);
                acc1g[3] += yl[i + 24] * (qsg[i / 2] & 0x00c0);
                acc2g[3] += yl[i + 25] * (qsg[i / 2] & 0xc000);

                acc1u[0] += yl[i +  0] * (qsu[i / 2] & 0x0003);
                acc2u[0] += yl[i +  1] * (qsu[i / 2] & 0x0300);
                acc1u[1] += yl[i +  8] * (qsu[i / 2] & 0x000c);
                acc2u[1] += yl[i +  9] * (qsu[i / 2] & 0x0c00);
                acc1u[2] += yl[i + 16] * (qsu[i / 2] & 0x0030);
                acc2u[2] += yl[i + 17] * (qsu[i / 2] & 0x3000);
                acc1u[3] += yl[i + 24] * (qsu[i / 2] & 0x00c0);
                acc2u[3] += yl[i + 25] * (qsu[i / 2] & 0xc000);
            }

            const float dg = dhg[0];
            const float mg = dhg[1] * 1.f / 16.f;
            sumg[row] += dg * ((acc1g[0] + 1.f / 256.f * acc2g[0]) * (scg[0] & 0xF) * 1.f /  1.f +
                               (acc1g[1] + 1.f / 256.f * acc2g[1]) * (scg[2] & 0xF) * 1.f /  4.f +
                               (acc1g[2] + 1.f / 256.f * acc2g[2]) * (scg[4] & 0xF) * 1.f / 16.f +
                               (acc1g[3] + 1.f / 256.f * acc2g[3]) * (scg[6] & 0xF) * 1.f / 64.f) -
                         mg * (sumy[0] * (scg[0] & 0xF0) + sumy[1] * (scg[2] & 0xF0) +
                               sumy[2] * (scg[4] & 0xF0) + sumy[3] * (scg[6] & 0xF0));

            const float du = dhu[0];
            const float mu = dhu[1] * 1.f / 16.f;
            sumu[row] += du * ((acc1u[0] + 1.f / 256.f * acc2u[0]) * (scu[0] & 0xF) * 1.f /  1.f +
                               (acc1u[1] + 1.f / 256.f * acc2u[1]) * (scu[2] & 0xF) * 1.f /  4.f +
                               (acc1u[2] + 1.f / 256.f * acc2u[2]) * (scu[4] & 0xF) * 1.f / 16.f +
                               (acc1u[3] + 1.f / 256.f * acc2u[3]) * (scu[6] & 0xF) * 1.f / 64.f) -
                         mu * (sumy[0] * (scu[0] & 0xF0) + sumy[1] * (scu[2] & 0xF0) +
                               sumy[2] * (scu[4] & 0xF0) + sumy[3] * (scu[6] & 0xF0));

            qsg += args.gate_row_bytes / 2;
            scg += args.gate_row_bytes;
            dhg += args.gate_row_bytes / 2;
            qsu += args.up_row_bytes / 2;
            scu += args.up_row_bytes;
            dhu += args.up_row_bytes / 2;
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0;
         row < N_R0 && row0 + (uint)row < args.mid_dim;
         row++) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);
        if (tiisg == 0u) {
            const float sw = g / (1.0f + exp(-g));
            mid[mid_base + row0 + (uint)row] = sw * u * weights[selected_off];
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q2_K_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    if (!ds4_tp_owns_expert(expert, args.n_total_expert,
                            args.tp_rank, args.tp_world)) return;
    glm_q2_K_pair_swiglu_simd_f32_impl<N_R0_Q2_K>(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, expert - args.tp_expert_base,
        tiisg, sgitg);
}

kernel void kernel_glm_q2_K_addr_pair_swiglu2_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *gate_addrs,
        device const uint64_t *up_addrs,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_GLM_Q2_PAIR2_K;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (row0 >= args.mid_dim) return;

    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_GLM_Q2_PAIR2_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)expert];
    const uint64_t up_addr = up_addrs[(uint)expert];
    if (gate_addr == 0 || up_addr == 0) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_GLM_Q2_PAIR2_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    ds4_metal_glm_routed_moe_args local = args;
    local.n_total_expert = 1;
    local.gate_expert_bytes = 0;
    local.up_expert_bytes = 0;
    glm_q2_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q2_PAIR2_K>(
        local,
        reinterpret_cast<device const char *>(gate_addr),
        reinterpret_cast<device const char *>(up_addr),
        x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q2_K_addr_pair_swiglu2_f32_masked(
        constant ds4_metal_glm_routed_moe_args &args,
        constant uint32_t &active_mask,
        device const uint64_t *gate_addrs,
        device const uint64_t *up_addrs,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    if ((active_mask & (1u << slot)) == 0u) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_GLM_Q2_PAIR2_K;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (row0 >= args.mid_dim) return;

    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_GLM_Q2_PAIR2_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)expert];
    const uint64_t up_addr = up_addrs[(uint)expert];
    if (gate_addr == 0 || up_addr == 0) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_GLM_Q2_PAIR2_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    ds4_metal_glm_routed_moe_args local = args;
    local.n_total_expert = 1;
    local.gate_expert_bytes = 0;
    local.up_expert_bytes = 0;
    glm_q2_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q2_PAIR2_K>(
        local,
        reinterpret_cast<device const char *>(gate_addr),
        reinterpret_cast<device const char *>(up_addr),
        x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

template <short N_R0>
static inline void glm_q4_K_pair_swiglu_simd_f32_impl(
        ds4_metal_glm_routed_moe_args args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch,
        uint3 tgpig,
        uint slot,
        uint token,
        uint64_t selected_off,
        int expert,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0;
    if (row0 >= args.mid_dim || slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0 && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = args.in_dim / QK_K;
    const uint64_t expert_gate = (uint64_t)(uint)expert * args.gate_expert_bytes;
    const uint64_t expert_up = (uint64_t)(uint)expert * args.up_expert_bytes;
    device const block_q4_K *xg =
        (device const block_q4_K *)(gate + expert_gate +
            (uint64_t)row0 * args.gate_row_bytes);
    device const block_q4_K *xu =
        (device const block_q4_K *)(up + expert_up +
            (uint64_t)row0 * args.up_row_bytes);
    device const float *y = x + (uint64_t)token * args.in_dim;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0] = {0.f};
    float sumu[N_R0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)xg[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint16_t *scu = (device const uint16_t *)xu[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0;
             row < N_R0 && row0 + (uint)row < args.mid_dim;
             row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1g[0] += yl[2 * i + 0] * (qg1[i] & 0x000F);
                acc1g[1] += yl[2 * i + 1] * (qg1[i] & 0x0F00);
                acc1g[2] += yl[2 * i + 8] * (qg1[i] & 0x00F0);
                acc1g[3] += yl[2 * i + 9] * (qg1[i] & 0xF000);
                acc2g[0] += yh[2 * i + 0] * (qg2[i] & 0x000F);
                acc2g[1] += yh[2 * i + 1] * (qg2[i] & 0x0F00);
                acc2g[2] += yh[2 * i + 8] * (qg2[i] & 0x00F0);
                acc2g[3] += yh[2 * i + 9] * (qg2[i] & 0xF000);
            }

            sumg[row] += dhg[0] * ((acc1g[0] + 1.f / 256.f * acc1g[1]) * sc8[0] +
                                   (acc1g[2] + 1.f / 256.f * acc1g[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2g[0] + 1.f / 256.f * acc2g[1]) * sc8[4] +
                                   (acc2g[2] + 1.f / 256.f * acc2g[3]) * sc8[5] * 1.f / 16.f) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1u[0] += yl[2 * i + 0] * (qu1[i] & 0x000F);
                acc1u[1] += yl[2 * i + 1] * (qu1[i] & 0x0F00);
                acc1u[2] += yl[2 * i + 8] * (qu1[i] & 0x00F0);
                acc1u[3] += yl[2 * i + 9] * (qu1[i] & 0xF000);
                acc2u[0] += yh[2 * i + 0] * (qu2[i] & 0x000F);
                acc2u[1] += yh[2 * i + 1] * (qu2[i] & 0x0F00);
                acc2u[2] += yh[2 * i + 8] * (qu2[i] & 0x00F0);
                acc2u[3] += yh[2 * i + 9] * (qu2[i] & 0xF000);
            }

            sumu[row] += dhu[0] * ((acc1u[0] + 1.f / 256.f * acc1u[1]) * sc8[0] +
                                   (acc1u[2] + 1.f / 256.f * acc1u[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2u[0] + 1.f / 256.f * acc2u[1]) * sc8[4] +
                                   (acc2u[2] + 1.f / 256.f * acc2u[3]) * sc8[5] * 1.f / 16.f) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.gate_row_bytes / 2;
            scg += args.gate_row_bytes / 2;
            dhg += args.gate_row_bytes / 2;
            qu1 += args.up_row_bytes / 2;
            scu += args.up_row_bytes / 2;
            dhu += args.up_row_bytes / 2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0;
         row < N_R0 && row0 + (uint)row < args.mid_dim;
         ++row) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);
        if (tiisg == 0u) {
            const float sw = g / (1.0f + exp(-g));
            mid[mid_base + row0 + (uint)row] = sw * u * weights[selected_off];
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q4_K_pair_swiglu2_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q4_PAIR2_K>(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, expert, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_addr_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *gate_addrs,
        device const uint64_t *up_addrs,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q4_K;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (row0 >= args.mid_dim) return;

    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_Q4_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)expert];
    const uint64_t up_addr = up_addrs[(uint)expert];
    if (gate_addr == 0 || up_addr == 0) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_Q4_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    ds4_metal_glm_routed_moe_args local = args;
    local.n_total_expert = 1;
    local.gate_expert_bytes = 0;
    local.up_expert_bytes = 0;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        local,
        reinterpret_cast<device const char *>(gate_addr),
        reinterpret_cast<device const char *>(up_addr),
        x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_addr_pair_swiglu_f32_masked(
        constant ds4_metal_glm_routed_moe_args &args,
        constant uint32_t &active_mask,
        device const uint64_t *gate_addrs,
        device const uint64_t *up_addrs,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    if ((active_mask & (1u << slot)) == 0u) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q4_K;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (row0 >= args.mid_dim) return;

    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_Q4_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)expert];
    const uint64_t up_addr = up_addrs[(uint)expert];
    if (gate_addr == 0 || up_addr == 0) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_Q4_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    ds4_metal_glm_routed_moe_args local = args;
    local.n_total_expert = 1;
    local.gate_expert_bytes = 0;
    local.up_expert_bytes = 0;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        local,
        reinterpret_cast<device const char *>(gate_addr),
        reinterpret_cast<device const char *>(up_addr),
        x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_slots6_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_slots8_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *gate6,
        device const char *gate7,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const char *up6,
        device const char *up7,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    case 6: gate_cur = gate6; up_cur = up6; break;
    case 7: gate_cur = gate7; up_cur = up7; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_slots6_pair_swiglu4_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q4_PAIR_K>(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_slots8_pair_swiglu4_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *gate6,
        device const char *gate7,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const char *up6,
        device const char *up7,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    case 6: gate_cur = gate6; up_cur = up6; break;
    case 7: gate_cur = gate7; up_cur = up7; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q4_PAIR_K>(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_pair_swiglu4_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q4_PAIR_K>(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, expert, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_pair_swiglu2_mapped_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const uint32_t *htpe,
        device const int32_t *hids,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint expert = tgpig.z;
    if (expert >= args.n_total_expert) return;
    const uint count = htpe[expert];
    const uint map_base = tgpig.y * 32u;
    for (uint i = 0; i < 32u; i++) {
        const uint map_row = map_base + i;
        if (map_row >= count) break;
        const int id = hids[(uint64_t)expert * args.n_tokens + map_row];
        if (id < 0) continue;
        const uint token = (uint)id / args.n_expert_used;
        const uint slot = (uint)id - token * args.n_expert_used;
        if (slot >= args.n_expert_used || token >= args.n_tokens) continue;
        const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
        glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
            args, gate, up, x, weights, mid, scratch,
            tgpig, slot, token, selected_off, (int)expert, tiisg, sgitg);
    }
}

kernel void kernel_glm_q4_K_pair_swiglu2_mapped_row_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const uint32_t *htpe,
        device const int32_t *hids,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint expert = tgpig.z;
    const uint map_row = tgpig.y;
    if (expert >= args.n_total_expert || map_row >= htpe[expert]) return;
    const int id = hids[(uint64_t)expert * args.n_tokens + map_row];
    if (id < 0) return;
    const uint token = (uint)id / args.n_expert_used;
    const uint slot = (uint)id - token * args.n_expert_used;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, (int)expert, tiisg, sgitg);
}

static inline void glm_q5_K_pair_swiglu_f32_impl(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch,
        uint3 tgpig,
        uint slot,
        uint token,
        uint64_t selected_off,
        int expert,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q5_PAIR_K;
    if (row0 >= args.mid_dim || slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0; row < N_R0_Q5_PAIR_K && row0 + (uint)row < args.mid_dim; row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const uint bit0 = 2u * (uint)iq;
    const uint bit1 = bit0 + 1u;
    const uint bit2 = bit0 + 4u;
    const uint bit3 = bit0 + 5u;
    const int nb = args.in_dim / QK_K;

    const uint64_t expert_gate = (uint64_t)(uint)expert * args.gate_expert_bytes;
    const uint64_t expert_up = (uint64_t)(uint)expert * args.up_expert_bytes;
    device const block_q5_K *gate_rows =
        (device const block_q5_K *)(gate +
            expert_gate + (uint64_t)row0 * args.gate_row_bytes);
    device const block_q5_K *up_rows =
        (device const block_q5_K *)(up +
            expert_up + (uint64_t)row0 * args.up_row_bytes);
    device const float *y = x + (uint64_t)token * args.in_dim;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0_Q5_PAIR_K] = {0.f};
    float sumu[N_R0_Q5_PAIR_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)gate_rows[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)gate_rows[ib].qs + 16 * iq + 4 * ir;
        device const uint16_t *qhg = (device const uint16_t *)gate_rows[ib].qh + 4 * ir;
        device const half *dhg = &gate_rows[ib].d;
        device const uint16_t *scu = (device const uint16_t *)up_rows[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)up_rows[ib].qs + 16 * iq + 4 * ir;
        device const uint16_t *qhu = (device const uint16_t *)up_rows[ib].qh + 4 * ir;
        device const half *dhu = &up_rows[ib].d;

        for (short row = 0; row < N_R0_Q5_PAIR_K && row0 + (uint)row < args.mid_dim; row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 accg = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                const uint ql1 = (uint)qg1[i];
                const uint ql2 = (uint)qg2[i];
                const uint hb = (uint)qhg[i];
                accg[0] += yl[2 * i + 0] *
                               (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                           yl[2 * i + 1] *
                               (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                accg[1] += yl[2 * i + 8] *
                               (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                           yl[2 * i + 9] *
                               (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                accg[2] += yh[2 * i + 0] *
                               (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                           yh[2 * i + 1] *
                               (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                accg[3] += yh[2 * i + 8] *
                               (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                           yh[2 * i + 9] *
                               (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
            }

            sumg[row] += dhg[0] * (accg[0] * sc8[0] + accg[1] * sc8[1] +
                                   accg[2] * sc8[4] + accg[3] * sc8[5]) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 accu = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                const uint ql1 = (uint)qu1[i];
                const uint ql2 = (uint)qu2[i];
                const uint hb = (uint)qhu[i];
                accu[0] += yl[2 * i + 0] *
                               (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                           yl[2 * i + 1] *
                               (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                accu[1] += yl[2 * i + 8] *
                               (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                           yl[2 * i + 9] *
                               (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                accu[2] += yh[2 * i + 0] *
                               (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                           yh[2 * i + 1] *
                               (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                accu[3] += yh[2 * i + 8] *
                               (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                           yh[2 * i + 9] *
                               (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
            }

            sumu[row] += dhu[0] * (accu[0] * sc8[0] + accu[1] * sc8[1] +
                                   accu[2] * sc8[4] + accu[3] * sc8[5]) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.gate_row_bytes / 2;
            qhg += args.gate_row_bytes / 2;
            scg += args.gate_row_bytes / 2;
            dhg += args.gate_row_bytes / 2;
            qu1 += args.up_row_bytes / 2;
            qhu += args.up_row_bytes / 2;
            scu += args.up_row_bytes / 2;
            dhu += args.up_row_bytes / 2;
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < N_R0_Q5_PAIR_K && row0 + (uint)row < args.mid_dim; row++) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);
        if (tiisg == 0u) {
            const float sw = g / (1.0f + exp(-g));
            mid[mid_base + row0 + (uint)row] = sw * u * weights[selected_off];
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q5_K_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    glm_q5_K_pair_swiglu_f32_impl(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, expert, tiisg, sgitg);
}

kernel void kernel_glm_q5_K_slots6_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q5_K_pair_swiglu_f32_impl(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q5_K_slots8_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *gate6,
        device const char *gate7,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const char *up6,
        device const char *up7,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    case 6: gate_cur = gate6; up_cur = up6; break;
    case 7: gate_cur = gate7; up_cur = up7; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q5_K_pair_swiglu_f32_impl(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q5_K_pair_swiglu_mapped_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const uint32_t *htpe,
        device const int32_t *hids,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint expert = tgpig.z;
    if (expert >= args.n_total_expert) return;
    const uint count = htpe[expert];
    const uint map_base = tgpig.y * 32u;
    for (uint i = 0; i < 32u; i++) {
        const uint map_row = map_base + i;
        if (map_row >= count) break;
        const int id = hids[(uint64_t)expert * args.n_tokens + map_row];
        if (id < 0) continue;
        const uint token = (uint)id / args.n_expert_used;
        const uint slot = (uint)id - token * args.n_expert_used;
        if (slot >= args.n_expert_used || token >= args.n_tokens) continue;
        const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
        glm_q5_K_pair_swiglu_f32_impl(
            args, gate, up, x, weights, mid, scratch,
            tgpig, slot, token, selected_off, (int)expert, tiisg, sgitg);
    }
}

kernel void kernel_glm_q5_K_pair_swiglu_mapped_row_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const uint32_t *htpe,
        device const int32_t *hids,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint expert = tgpig.z;
    const uint map_row = tgpig.y;
    if (expert >= args.n_total_expert || map_row >= htpe[expert]) return;
    const int id = hids[(uint64_t)expert * args.n_tokens + map_row];
    if (id < 0) return;
    const uint token = (uint)id / args.n_expert_used;
    const uint slot = (uint)id - token * args.n_expert_used;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q5_K_pair_swiglu_f32_impl(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, (int)expert, tiisg, sgitg);
}

kernel void kernel_glm_q5_K_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q5_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const uint bit0 = 2u * (uint)iq;
    const uint bit1 = bit0 + 1u;
    const uint bit2 = bit0 + 4u;
    const uint bit3 = bit0 + 5u;
    const int nb = args.mid_dim / QK_K;

    float sumf[N_R0_Q5_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        device const block_q5_K *x =
            (device const block_q5_K *)(down +
                (uint64_t)(uint)expert * args.down_expert_bytes +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const uint16_t *qh = (device const uint16_t *)x[ib].qh + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    const uint ql1 = (uint)q1[i];
                    const uint ql2 = (uint)q2[i];
                    const uint hb = (uint)qh[i];
                    acc[0] += yl[2 * i + 0] *
                                  (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                              yl[2 * i + 1] *
                                  (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                    acc[1] += yl[2 * i + 8] *
                                  (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                              yl[2 * i + 9] *
                                  (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                    acc[2] += yh[2 * i + 0] *
                                  (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                              yh[2 * i + 1] *
                                  (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                    acc[3] += yh[2 * i + 8] *
                                  (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                              yh[2 * i + 9] *
                                  (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
                }

                sumf[row] += dh[0] * (acc[0] * sc8[0] + acc[1] * sc8[1] +
                                      acc[2] * sc8[4] + acc[3] * sc8[5]) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                qh += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q2_K_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q2_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q2_K] = {0.f};
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        if (!ds4_tp_owns_expert(expert, args.n_total_expert,
                                args.tp_rank, args.tp_world)) continue;
        device const block_q2_K *x =
            (device const block_q2_K *)(down +
                (uint64_t)(uint)(expert - args.tp_expert_base) * args.down_expert_bytes +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;
            for (short row = 0; row < N_R0_Q2_K && row0 + (uint)row < args.out_dim; row++) {
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                    acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                    acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                    acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                    acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                    acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                }

                const float d = dh[0];
                const float m = dh[1] * 1.f / 16.f;
                sumf[row] += d * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                  (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                  (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                  (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                             m * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                  sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                qs += args.down_row_bytes / 2;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q2_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q2_K_addr_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *down_addrs,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q2_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q2_K] = {0.f};
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        const uint64_t down_addr = down_addrs[(uint)expert];
        if (down_addr == 0) continue;
        device const block_q2_K *x =
            (device const block_q2_K *)(reinterpret_cast<device const char *>(down_addr) +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;
            for (short row = 0; row < N_R0_Q2_K && row0 + (uint)row < args.out_dim; row++) {
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                    acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                    acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                    acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                    acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                    acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                }

                const float d = dh[0];
                const float m = dh[1] * 1.f / 16.f;
                sumf[row] += d * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                  (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                  (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                  (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                             m * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                  sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                qs += args.down_row_bytes / 2;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q2_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q4_K_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint ntg = 256u;
    const uint row = tgpig.x;
    const uint token = tgpig.y;
    if (row >= args.out_dim || token >= args.n_tokens) return;

    float acc = 0.0f;
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        device const block_q4_K *down_row =
            (device const block_q4_K *)(down +
                (uint64_t)(uint)expert * args.down_expert_bytes +
                (uint64_t)row * args.down_row_bytes);
        device const float *slot_mid = mid + mid_base + (uint64_t)slot * args.mid_dim;
        for (uint k = tid; k < args.mid_dim; k += ntg) {
            acc += ds4_glm_q4_K_value(down_row, k) * slot_mid[k];
        }
    }

    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        out[(uint64_t)token * args.out_dim + row] = scratch[0];
    }
}

kernel void kernel_glm_q4_K_addr_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *down_addrs,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint ntg = 256u;
    const uint row = tgpig.x;
    const uint token = tgpig.y;
    if (row >= args.out_dim || token >= args.n_tokens) return;

    float acc = 0.0f;
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        const uint64_t down_addr = down_addrs[(uint)expert];
        if (down_addr == 0) continue;
        device const block_q4_K *down_row =
            (device const block_q4_K *)(reinterpret_cast<device const char *>(down_addr) +
                (uint64_t)row * args.down_row_bytes);
        device const float *slot_mid = mid + mid_base + (uint64_t)slot * args.mid_dim;
        for (uint k = tid; k < args.mid_dim; k += ntg) {
            acc += ds4_glm_q4_K_value(down_row, k) * slot_mid[k];
        }
    }

    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        out[(uint64_t)token * args.out_dim + row] = scratch[0];
    }
}

kernel void kernel_glm_q4_K_down_simd_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.mid_dim / QK_K;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * (uint)nr0;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;

        device const block_q4_K *x =
            (device const block_q4_K *)(down +
                (uint64_t)(uint)expert * args.down_expert_bytes +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0 && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                }

                sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                      (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                      (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                      (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < nr0 && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q4_K_addr_down_simd_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *down_addrs,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.mid_dim / QK_K;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * (uint)nr0;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        const uint64_t down_addr = down_addrs[(uint)expert];
        if (down_addr == 0) continue;

        device const block_q4_K *x =
            (device const block_q4_K *)(reinterpret_cast<device const char *>(down_addr) +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0 && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                }

                sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                      (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                      (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                      (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < nr0 && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q6_K_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q6_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q6_K] = {0.f};
    float yl[16];
    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        device const block_q6_K *x =
            (device const block_q6_K *)(down +
                (uint64_t)(uint)expert * args.down_expert_bytes +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *yy = mid + mid_base + (uint64_t)slot * args.mid_dim;

        for (int ib = ix; ib < nb; ib += 2) {
            device const uchar *q1 = x[ib].ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = x[ib].qh + q_offset_h;
            device const char *sc = x[ib].scales + is;
            device const half *dh = &x[ib].d;
            device const float *y = yy + (uint64_t)ib * QK_K + y_offset;

            for (short l = 0; l < 4; ++l) {
                yl[4 * l + 0] = y[l + 0];
                yl[4 * l + 1] = y[l + 32];
                yl[4 * l + 2] = y[l + 64];
                yl[4 * l + 3] = y[l + 96];
            }

            for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint h = (uint)qh[l];
                    sums[0] += yl[4 * l + 0] *
                        (float)((int)((q1[l] & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                    sums[1] += yl[4 * l + 1] *
                        (float)((int)((q2[l] & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                    sums[2] += yl[4 * l + 2] *
                        (float)((int)((q1[l] >> 4u) | (h & kmask3)) - 32);
                    sums[3] += yl[4 * l + 3] *
                        (float)((int)((q2[l] >> 4u) | ((h & kmask4) >> 2u)) - 32);
                }

                sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                      sums[2] * sc[4] + sums[3] * sc[6]);

                q1 += args.down_row_bytes;
                q2 += args.down_row_bytes;
                qh += args.down_row_bytes;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }
        }
    }

    for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q5_K_slots6_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down0,
        device const char *down1,
        device const char *down2,
        device const char *down3,
        device const char *down4,
        device const char *down5,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q5_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const uint bit0 = 2u * (uint)iq;
    const uint bit1 = bit0 + 1u;
    const uint bit2 = bit0 + 4u;
    const uint bit3 = bit0 + 5u;
    const int nb = args.mid_dim / QK_K;

    float sumf[N_R0_Q5_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        device const char *down_cur = down0;
        switch (slot) {
        case 1: down_cur = down1; break;
        case 2: down_cur = down2; break;
        case 3: down_cur = down3; break;
        case 4: down_cur = down4; break;
        case 5: down_cur = down5; break;
        default: break;
        }

        device const block_q5_K *x =
            (device const block_q5_K *)(down_cur +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const uint16_t *qh = (device const uint16_t *)x[ib].qh + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    const uint ql1 = (uint)q1[i];
                    const uint ql2 = (uint)q2[i];
                    const uint hb = (uint)qh[i];
                    acc[0] += yl[2 * i + 0] *
                                  (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                              yl[2 * i + 1] *
                                  (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                    acc[1] += yl[2 * i + 8] *
                                  (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                              yl[2 * i + 9] *
                                  (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                    acc[2] += yh[2 * i + 0] *
                                  (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                              yh[2 * i + 1] *
                                  (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                    acc[3] += yh[2 * i + 8] *
                                  (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                              yh[2 * i + 9] *
                                  (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
                }

                sumf[row] += dh[0] * (acc[0] * sc8[0] + acc[1] * sc8[1] +
                                      acc[2] * sc8[4] + acc[3] * sc8[5]) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                qh += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q5_K_slots8_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down0,
        device const char *down1,
        device const char *down2,
        device const char *down3,
        device const char *down4,
        device const char *down5,
        device const char *down6,
        device const char *down7,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q5_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const uint bit0 = 2u * (uint)iq;
    const uint bit1 = bit0 + 1u;
    const uint bit2 = bit0 + 4u;
    const uint bit3 = bit0 + 5u;
    const int nb = args.mid_dim / QK_K;

    float sumf[N_R0_Q5_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        device const char *down_cur = down0;
        switch (slot) {
        case 1: down_cur = down1; break;
        case 2: down_cur = down2; break;
        case 3: down_cur = down3; break;
        case 4: down_cur = down4; break;
        case 5: down_cur = down5; break;
        case 6: down_cur = down6; break;
        case 7: down_cur = down7; break;
        default: break;
        }

        device const block_q5_K *x =
            (device const block_q5_K *)(down_cur +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const uint16_t *qh = (device const uint16_t *)x[ib].qh + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    const uint ql1 = (uint)q1[i];
                    const uint ql2 = (uint)q2[i];
                    const uint hb = (uint)qh[i];
                    acc[0] += yl[2 * i + 0] *
                                  (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                              yl[2 * i + 1] *
                                  (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                    acc[1] += yl[2 * i + 8] *
                                  (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                              yl[2 * i + 9] *
                                  (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                    acc[2] += yh[2 * i + 0] *
                                  (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                              yh[2 * i + 1] *
                                  (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                    acc[3] += yh[2 * i + 8] *
                                  (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                              yh[2 * i + 9] *
                                  (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
                }

                sumf[row] += dh[0] * (acc[0] * sc8[0] + acc[1] * sc8[1] +
                                      acc[2] * sc8[4] + acc[3] * sc8[5]) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                qh += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q6_K_slots6_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down0,
        device const char *down1,
        device const char *down2,
        device const char *down3,
        device const char *down4,
        device const char *down5,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q6_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q6_K] = {0.f};
    float yl[16];
    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        device const char *down_cur = down0;
        switch (slot) {
        case 1: down_cur = down1; break;
        case 2: down_cur = down2; break;
        case 3: down_cur = down3; break;
        case 4: down_cur = down4; break;
        case 5: down_cur = down5; break;
        default: break;
        }

        device const block_q6_K *x =
            (device const block_q6_K *)(down_cur +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *yy = mid + mid_base + (uint64_t)slot * args.mid_dim;

        for (int ib = ix; ib < nb; ib += 2) {
            device const uchar *q1 = x[ib].ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = x[ib].qh + q_offset_h;
            device const char *sc = x[ib].scales + is;
            device const half *dh = &x[ib].d;
            device const float *y = yy + (uint64_t)ib * QK_K + y_offset;

            for (short l = 0; l < 4; ++l) {
                yl[4 * l + 0] = y[l + 0];
                yl[4 * l + 1] = y[l + 32];
                yl[4 * l + 2] = y[l + 64];
                yl[4 * l + 3] = y[l + 96];
            }

            for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint h = (uint)qh[l];
                    sums[0] += yl[4 * l + 0] *
                        (float)((int)((q1[l] & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                    sums[1] += yl[4 * l + 1] *
                        (float)((int)((q2[l] & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                    sums[2] += yl[4 * l + 2] *
                        (float)((int)((q1[l] >> 4u) | (h & kmask3)) - 32);
                    sums[3] += yl[4 * l + 3] *
                        (float)((int)((q2[l] >> 4u) | ((h & kmask4) >> 2u)) - 32);
                }

                sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                      sums[2] * sc[4] + sums[3] * sc[6]);

                q1 += args.down_row_bytes;
                q2 += args.down_row_bytes;
                qh += args.down_row_bytes;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }
        }
    }

    for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q6_K_slots8_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down0,
        device const char *down1,
        device const char *down2,
        device const char *down3,
        device const char *down4,
        device const char *down5,
        device const char *down6,
        device const char *down7,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q6_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q6_K] = {0.f};
    float yl[16];
    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        device const char *down_cur = down0;
        switch (slot) {
        case 1: down_cur = down1; break;
        case 2: down_cur = down2; break;
        case 3: down_cur = down3; break;
        case 4: down_cur = down4; break;
        case 5: down_cur = down5; break;
        case 6: down_cur = down6; break;
        case 7: down_cur = down7; break;
        default: break;
        }

        device const block_q6_K *x =
            (device const block_q6_K *)(down_cur +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *yy = mid + mid_base + (uint64_t)slot * args.mid_dim;

        for (int ib = ix; ib < nb; ib += 2) {
            device const uchar *q1 = x[ib].ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = x[ib].qh + q_offset_h;
            device const char *sc = x[ib].scales + is;
            device const half *dh = &x[ib].d;
            device const float *y = yy + (uint64_t)ib * QK_K + y_offset;

            for (short l = 0; l < 4; ++l) {
                yl[4 * l + 0] = y[l + 0];
                yl[4 * l + 1] = y[l + 32];
                yl[4 * l + 2] = y[l + 64];
                yl[4 * l + 3] = y[l + 96];
            }

            for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint h = (uint)qh[l];
                    sums[0] += yl[4 * l + 0] *
                        (float)((int)((q1[l] & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                    sums[1] += yl[4 * l + 1] *
                        (float)((int)((q2[l] & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                    sums[2] += yl[4 * l + 2] *
                        (float)((int)((q1[l] >> 4u) | (h & kmask3)) - 32);
                    sums[3] += yl[4 * l + 3] *
                        (float)((int)((q2[l] >> 4u) | ((h & kmask4) >> 2u)) - 32);
                }

                sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                      sums[2] * sc[4] + sums[3] * sc[6]);

                q1 += args.down_row_bytes;
                q2 += args.down_row_bytes;
                qh += args.down_row_bytes;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }
        }
    }

    for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ?
        (float)xb->d :
        (float)xb->d * (1.0f / 16.0f);
    const float min = (float)xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_mxfp4(device const block_mxfp4 *xb, short il, thread type4x4 &reg) {
    const float d = ds4_metal_e8m0_to_f32(xb->e);
    const uint shift = il == 0 ? 0u : 4u;
    FOR_UNROLL (short i = 0; i < QK_MXFP4/2; i++) {
        const uint q = ((uint)xb->qs[i] >> shift) & 0x0fu;
        reg[i/4][i%4] = d * ds4_metal_mxfp4_values[q];
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 &reg) {
    const short group = il / 2;
    const short l0 = (il & 1) * 16;
    const uchar2 sm = get_scale_min_k4_just2(group, 0, xb->scales);
    const float dl = (float)xb->d * (float)sm.x;
    const float ml = (float)xb->dmin * (float)sm.y;
    device const uchar *q = xb->qs + (group / 2) * 32 + l0;
    device const uchar *qh = xb->qh + l0;
    const uint shift = (uint)(group & 1) * 4u;
    const uchar hmask = (uchar)(1u << (uint)group);

    for (int i = 0; i < 16; ++i) {
        uint v = ((uint)q[i] >> shift) & 0x0Fu;
        v += (qh[i] & hmask) ? 16u : 0u;
        reg[i / 4][i % 4] = dl * (float)v - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 &reg) {
    const short n128 = il / 8;
    const short il128 = il & 7;
    const short quarter = il128 / 2;
    const short l0 = (il128 & 1) * 16;
    const uint ql_base = (uint)n128 * 64u;
    const uint qh_base = (uint)n128 * 32u;
    const uint sc_base = (uint)n128 * 8u + (uint)quarter * 2u + (uint)(il128 & 1);
    const float d = (float)xb->d * (float)((int)xb->scales[sc_base]);

    for (int i = 0; i < 16; ++i) {
        const uint l = (uint)l0 + (uint)i;
        uint v;
        if (quarter == 0) {
            v = ((uint)xb->ql[ql_base + l] & 0x0Fu) |
                ((((uint)xb->qh[qh_base + l] >> 0u) & 3u) << 4u);
        } else if (quarter == 1) {
            v = ((uint)xb->ql[ql_base + 32u + l] & 0x0Fu) |
                ((((uint)xb->qh[qh_base + l] >> 2u) & 3u) << 4u);
        } else if (quarter == 2) {
            v = ((uint)xb->ql[ql_base + l] >> 4u) |
                ((((uint)xb->qh[qh_base + l] >> 4u) & 3u) << 4u);
        } else {
            v = ((uint)xb->ql[ql_base + 32u + l] >> 4u) |
                ((((uint)xb->qh[qh_base + l] >> 6u) & 3u) << 4u);
        }
        reg[i / 4][i % 4] = d * (float)((int)v - 32);
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

struct ds4_metal_args_mul_mv_id {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
    /* Tensor-parallel expert ownership: when tp_world > 1, routed-MoE
     * kernels skip experts outside this rank's contiguous id range so this
     * rank never touches their weights. Zero-initialized (tp_world 0) means
     * no split; only the routed id kernels read these. */
    int32_t  tp_rank;
    int32_t  tp_world;
    /* When set, the sum6 kernels add this per-row vector (the shared-expert
     * partial) into their accumulation, folding the TP local-add. */
    int32_t  tp_addend;
    /* First expert id present at the bound blob base (TP mappings bind only
     * the owned range; kernels rebase before pointer math). */
    int32_t  tp_expert_base;
};

struct ds4_metal_moe_expert_group_args {
    uint32_t expert_base;
    uint32_t expert_count;
    uint32_t accumulate;
    uint32_t pad0;
};

struct ds4_metal_q4_gather_slots6_args {
    uint64_t expert_bytes;
    uint32_t group_size;
    uint32_t n_slots;
};

struct ds4_metal_q4_expert_table {
    array<device const char *, 384> experts [[id(0)]];
};

struct ds4_metal_expert_address_table {
    device const uint64_t *addrs;
};

struct ds4_metal_stream_expert_validate_args {
    uint32_t n_total_expert;
    uint32_t n_expert;
};

struct ds4_metal_stream_expert_split_args {
    uint32_t active_mask;
    uint32_t accumulate;
};

struct ds4_metal_args_mul_mm_id_map0 {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
};

struct ds4_metal_args_mul_mm_id {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    /* Tensor-parallel expert ownership; tp_world<=1 disables the split.
     * tp_expert_base rebases expert ids when the bind covers only the
     * owned contiguous range of the expert blob. */
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_expert_base;
};

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3
    const short is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_K *x = (device const block_q4_K *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    float yl[16];
    float yh[16];
    float sumf[nr0] = {0.f};

    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t *q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                  (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01 / 2;
            sc += args.nb01 / 2;
            dh += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }

    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_mxfp4_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00 / QK_MXFP4;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint i12 = tgpig.z % args.ne12;
    const uint i13 = tgpig.z / args.ne12;
    const uint64_t offset0 = (uint64_t)first_row * args.nb01 +
        (uint64_t)(i12 / args.r2) * args.nb02 + (uint64_t)(i13 / args.r3) * args.nb03;
    const uint64_t offset1 = (uint64_t)tgpig.y * args.nb11 +
        (uint64_t)i12 * args.nb12 + (uint64_t)i13 * args.nb13;

    device const block_mxfp4 *x = (device const block_mxfp4 *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);
    const int row_blocks = (int)(args.nb01 / sizeof(block_mxfp4));
    const short ix = tiisg / 2;
    const short it = tiisg & 1;

    threadgroup float *lut = (threadgroup float *)shmem;
    if (sgitg == 0) lut[tiisg] = ds4_metal_mxfp4_values[tiisg & 15];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf[nr0] = {0.f};
    device const float *yb = y + ix * QK_MXFP4 + it * 8;
    for (int ib = ix; ib < nb; ib += 16) {
        device const float4 *y4 = (device const float4 *)yb;
        const float4 yl0 = y4[0];
        const float4 yl1 = y4[4];
        const float4 yl2 = y4[1];
        const float4 yl3 = y4[5];

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            device const block_mxfp4 &xb = x[row * row_blocks + ib];
            device const uchar *q = xb.qs + 8 * it;
            float4 acc = yl0 * float4(lut[q[0] & 15], lut[q[1] & 15],
                                      lut[q[2] & 15], lut[q[3] & 15]);
            acc += yl1 * float4(lut[q[0] >> 4], lut[q[1] >> 4],
                                lut[q[2] >> 4], lut[q[3] >> 4]);
            acc += yl2 * float4(lut[q[4] & 15], lut[q[5] & 15],
                                lut[q[6] & 15], lut[q[7] & 15]);
            acc += yl3 * float4(lut[q[4] >> 4], lut[q[5] >> 4],
                                lut[q[6] >> 4], lut[q[7] >> 4]);
            sumf[row] += ds4_metal_e8m0_to_f32(xb.e) *
                         ((acc.x + acc.y) + (acc.z + acc.w));
        }
        yb += 16 * QK_MXFP4;
    }

    device float *dst_f32 = (device float *)dst +
        (uint64_t)tgpig.z * args.ne0 * args.ne1 + (uint64_t)tgpig.y * args.ne0;
    FOR_UNROLL (short row = 0; row < nr0; row++) {
        if (first_row + row < args.ne0) {
            const float sum = simd_sum(sumf[row]);
            if (tiisg == 0) dst_f32[first_row + row] = sum;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_q8_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q8_K *x = (device const block_q8_K *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short il = tiisg % 8;
    const int ib0 = ix;

    device const float *yb = y + ib0 * QK_K + il * 32;

    for (int ib = ib0; ib < nb; ib += 4) {
        float yl[32];
        for (short i = 0; i < 32; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < nr0; row++) {
            device const block_q8_K *xr =
                (device const block_q8_K *)((device const char *)x + (uint64_t)row * args.nb01);
            device const int8_t *qs = xr[ib].qs + il * 32;

            float sumq = 0.f;
            for (short i = 0; i < 32; ++i) {
                sumq += (float)qs[i] * yl[i];
            }

            sumf[row] += sumq * xr[ib].d;
        }

        yb += 4 * QK_K;
    }

    device float *dst_f32 =
        (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }

    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

template<int nr0>
void kernel_mul_mv_iq2_xxs_pair_f32_impl(
        ds4_metal_args_mul_mv args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * xg = (device const block_iq2_xxs *) (src0_gate + offset0);
    device const block_iq2_xxs * xu = (device const block_iq2_xxs *) (src0_up   + offset0);
    device const float         * y  = (device const float         *) (src1      + offset1);

    float yl[32];
    float sumg[nr0]={0.f};
    float sumu[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xgr = xg + ibl;
        device const block_iq2_xxs * xur = xu + ibl;
        device const uint16_t * qg = xgr->qs + 4 * ib;
        device const uint16_t * qu = xur->qs + 4 * ib;
        device const half * dhg = &xgr->d;
        device const half * dhu = &xur->d;

        for (short row = 0; row < nr0; row++) {
            device const uint8_t * aux8g = (device const uint8_t *)qg;
            device const uint8_t * aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t * gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7*l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8*l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01/2;
            dhu += args.nb01/2;
            qg  += args.nb01/2;
            qu  += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_gate_f32 = (device float *) dst_gate + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_up_f32   = (device float *) dst_up   + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            dst_gate_f32[first_row + row] = sum_gate * 0.25f;
            dst_up_f32[first_row + row]   = sum_up   * 0.25f;
        }
    }
}

typedef void (kernel_mul_mv2_disp_t)(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv2_disp_t disp_fn>
void mmv_fn(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>) mul_mv_id_disp_fn_t;

// Decode-time expert matvec. The ids tensor selects the routed expert for each
// slot, then this wrapper invokes the quantized row kernel for Q8_0, Q2_K, or
// IQ2_XXS weights without materializing per-expert dispatches on the CPU.
template<mul_mv_id_disp_fn_t disp_fn>
kernel void kernel_mul_mv_id(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;

    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    if (!ds4_tp_owns_expert(i02, args.ne02, args.tp_rank, args.tp_world)) {
        /* Unowned expert under the TP split: zero this threadgroup's output
         * rows so the downstream expert-sum stages stay unchanged. */
        const short NSG = FC_mul_mv_nsg;
        const int row0 = (tgpig.x * NSG + sgitg) * args.nr0;
        device float *dst_f32 = (device float *)dst_cur;
        for (int r = 0; r < args.nr0 && row0 + r < args.ne0; r++) {
            if (tiisg == 0) dst_f32[row0 + r] = 0.0f;
        }
        return;
    }

    device const char * src0_cur =
        src0s + (int64_t)(i02 - args.tp_expert_base)*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    disp_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>) kernel_mul_mv_id_q_t;
typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>) kernel_mul_mv_id_q8_0_t;

// Host-visible decode MoE matvec variants for the DS4 quant formats.
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_q8_0_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>>>;
template [[host_name("kernel_mul_mv_id_q8_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_K_f32_impl<N_R0_Q8_K>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;
template [[host_name("kernel_mul_mv_id_mxfp4_f32")]]   kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>>>;

// Plain dense Q4_K matvec on the classic impl. The mul_mv_ext family used
// by the generic dense path tops out around 220 GB/s on M5 for the GLM
// DenseQ4 decode shapes; this impl streams the same rows at 530-650 GB/s.
kernel void kernel_mul_mv_q4_K_dense_f32(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// DS4 attention output low projection, specialized for the fixed block
// diagonal mapping used by the model:
//
//     low[token, group, rank] = heads[token, group, :] * Woa[group, rank, :]
//
// The generic GGML-style id matvec supports arbitrary routed expert ids.  Here
// the id is always equal to the group number, so this wrapper keeps the exact
// Q8_0 dot kernel but removes the id-buffer load and the CPU-side id table.
kernel void kernel_dsv4_attn_out_low_q8_0_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_cur = src0s + idx*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;
    device       char * dst_cur  = dst   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

kernel void kernel_dsv4_attn_out_low_q4_K_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;

    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_cur = src0s + idx * args.nb02;
    device const char * src1_cur = src1  + i11 * args.nb11 + i12 * args.nb12;
    device       char * dst_cur  = dst   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

kernel void kernel_mul_mv_id_iq2_xxs_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device char * dst_gate_cur = dst_gate + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);
    device char * dst_up_cur   = dst_up   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    if (!ds4_tp_owns_expert(i02, args.ne02, args.tp_rank, args.tp_world)) {
        /* Unowned expert: zero this threadgroup's gate/up rows so the
         * separate swiglu-weight stage yields zero mid rows. */
        const short NSG_z = FC_mul_mv_nsg;
        const int row0_z = (tgpig.x * NSG_z + sgitg) * args.nr0;
        device float *zg = (device float *)dst_gate_cur;
        device float *zu = (device float *)dst_up_cur;
        for (int r = 0; r < args.nr0 && row0_z + r < args.ne0; r++) {
            if (tiisg == 0) {
                zg[row0_z + r] = 0.0f;
                zu[row0_z + r] = 0.0f;
            }
        }
        return;
    }

    device const char * src0_gate_cur = src0_gate + (int64_t)(i02 - args.tp_expert_base)*args.nb02;
    device const char * src0_up_cur   = src0_up   + (int64_t)(i02 - args.tp_expert_base)*args.nb02;
    device const char * src1_cur      = src1      + i11*args.nb11 + i12*args.nb12;

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Decode-only routed expert gate/up projection fused with the DS4 activation:
//
//     mid = silu(clamp(gate)) * clamp(up) * route_weight
//
// The quantized dot products are intentionally the same IQ2_XXS paired path as
// `kernel_mul_mv_id_iq2_xxs_pair_f32`.  The only extra work is done by lane 0
// after each exact reduced row has been produced.  This removes the separate
// routed activation dispatch and avoids rereading the gate/up rows before the
// down projection.  The host uses this only for the normal release path where
// diagnostics do not request clamped gate/up intermediates.
kernel void kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1 * args.nbi1))[idx];
    if (!ds4_tp_owns_expert(i02, args.ne02, args.tp_rank, args.tp_world)) return;
    const int i02b = i02 - args.tp_expert_base;
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    const int nb32 = nb * (QK_K / 32);

    device const block_iq2_xxs *xg =
        (device const block_iq2_xxs *)(src0_gate + (int64_t)i02b * args.nb02 + (uint64_t)first_row * args.nb01);
    device const block_iq2_xxs *xu =
        (device const block_iq2_xxs *)(src0_up + (int64_t)i02b * args.nb02 + (uint64_t)first_row * args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * args.nb11 + i12 * args.nb12);

    float yl[32];
    float sumg[N_R0_IQ2_XXS] = {0.f};
    float sumu[N_R0_IQ2_XXS] = {0.f};

    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        // The lookup tables contain 256 grid values and 128 sign values.  Split
        // those loads across every lane in the specialized threadgroup so the
        // decode kernel remains correct when FC_mul_mv_nsg is tuned away from
        // DwarfStar's default of two SIMD groups.
        int nval = 8 / NSG;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 4 / NSG;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs *xgr = xg + ibl;
        device const block_iq2_xxs *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;

        for (short row = 0; row < N_R0_IQ2_XXS; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7 * l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7 * l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01 / 2;
            dhu += args.nb01 / 2;
            qg  += args.nb01 / 2;
            qu  += args.nb01 / 2;
        }

        y4 += 32 * 32;
    }

    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + pair_row * act.weight_stride);

    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            const float gate = sum_gate * 0.25f;
            const float up = sum_up * 0.25f;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            dst_gate_f32[out_row] = gate;
            dst_up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            dst_mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (idx) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const uint64_t * gate_addrs,
        device const uint64_t * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)i02];
    const uint64_t up_addr = up_addrs[(uint)i02];
    if (gate_addr == 0 || up_addr == 0) {
        return;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addr);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addr);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_addr_iq2_xxs_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const uint64_t * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;

    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const uint64_t addr = addrs[(uint)i02];
    if (addr == 0) {
        return;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_cur = reinterpret_cast<device const char *>(addr);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_cur = dst + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        constant ds4_metal_stream_expert_split_args & split,
        device const uint64_t * gate_addrs,
        device const uint64_t * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;
    if ((split.active_mask & (1u << (uint)idx)) == 0) {
        return;
    }

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)i02];
    const uint64_t up_addr = up_addrs[(uint)i02];
    if (gate_addr == 0 || up_addr == 0) {
        return;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addr);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addr);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_stream_expert_cache_validate(
        constant ds4_metal_stream_expert_validate_args & args,
        device const char * ids,
        device const uint64_t * gate_addrs,
        device const uint64_t * up_addrs,
        device const uint64_t * down_addrs,
        device uint32_t * status,
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    uint32_t miss_mask = 0;
    uint32_t invalid_mask = 0;
    const uint32_t n_expert = min(args.n_expert, (uint32_t)6);
    device const int32_t *selected = (device const int32_t *)ids;

    status[3] = n_expert;
    for (uint32_t i = 0; i < 6; i++) {
        const int32_t expert = i < n_expert ? selected[i] : -1;
        status[4 + i] = as_type<uint32_t>(expert);
        if (i >= n_expert) continue;
        if (expert < 0 ||
            (uint32_t)expert >= args.n_total_expert ||
            (uint32_t)expert >= 384) {
            invalid_mask |= (1u << i);
            continue;
        }
        const uint32_t e = (uint32_t)expert;
        if (gate_addrs[e] == 0 || up_addrs[e] == 0 || down_addrs[e] == 0) {
            miss_mask |= (1u << i);
        }
    }

    status[0] = (miss_mask == 0 && invalid_mask == 0) ? 1u : 0u;
    status[1] = miss_mask;
    status[2] = invalid_mask;
}

kernel void kernel_mul_mv_id_q4_K_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    if (!ds4_tp_owns_expert(i02, args.ne02, args.tp_rank, args.tp_world)) {
        const short NSG_z = FC_mul_mv_nsg;
        const int row0_z = (tgpig.x * NSG_z + sgitg) * args.nr0;
        device float *zg = (device float *)dst_gate_cur;
        device float *zu = (device float *)dst_up_cur;
        for (int r = 0; r < args.nr0 && row0_z + r < args.ne0; r++) {
            if (tiisg == 0) {
                zg[row0_z + r] = 0.0f;
                zu[row0_z + r] = 0.0f;
            }
        }
        return;
    }

    device const char *src0_gate_cur = src0_gate + (int64_t)(i02 - args.tp_expert_base) * args.nb02;
    device const char *src0_up_cur   = src0_up   + (int64_t)(i02 - args.tp_expert_base) * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Same release-path fusion as the IQ2_XXS kernel above for the Q4_K expert
// variant.  The Q4 pair path reuses the existing exact matvec implementation
// for gate and up, then the same lane that wrote each row derives the routed
// SwiGLU input.  This keeps Q4 behavior aligned with the Q2 optimization while
// preserving the old pair projection arithmetic.
kernel void kernel_mul_mv_id_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (!ds4_tp_owns_expert(i02, args.ne02, args.tp_rank, args.tp_world)) return;
    const int i02b = i02 - args.tp_expert_base;
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + (int64_t)i02b * args.nb02;
    device const char *src0_up_cur   = src0_up   + (int64_t)i02b * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    const short NSG = FC_mul_mv_nsg;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    device const block_q4_K *xg =
        (device const block_q4_K *)(src0_gate_cur + (uint64_t)first_row * args.nb01);
    device const block_q4_K *xu =
        (device const block_q4_K *)(src0_up_cur + (uint64_t)first_row * args.nb01);
    device const float *y = (device const float *)src1_cur;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0_Q4_K] = {0.f};
    float sumu[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)xg[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint16_t *scu = (device const uint16_t *)xu[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0; row < N_R0_Q4_K; row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1g[0] += yl[2 * i + 0] * (qg1[i] & 0x000F);
                acc1g[1] += yl[2 * i + 1] * (qg1[i] & 0x0F00);
                acc1g[2] += yl[2 * i + 8] * (qg1[i] & 0x00F0);
                acc1g[3] += yl[2 * i + 9] * (qg1[i] & 0xF000);
                acc2g[0] += yh[2 * i + 0] * (qg2[i] & 0x000F);
                acc2g[1] += yh[2 * i + 1] * (qg2[i] & 0x0F00);
                acc2g[2] += yh[2 * i + 8] * (qg2[i] & 0x00F0);
                acc2g[3] += yh[2 * i + 9] * (qg2[i] & 0xF000);
            }

            sumg[row] += dhg[0] * ((acc1g[0] + 1.f / 256.f * acc1g[1]) * sc8[0] +
                                   (acc1g[2] + 1.f / 256.f * acc1g[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2g[0] + 1.f / 256.f * acc2g[1]) * sc8[4] +
                                   (acc2g[2] + 1.f / 256.f * acc2g[3]) * sc8[5] * 1.f / 16.f) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1u[0] += yl[2 * i + 0] * (qu1[i] & 0x000F);
                acc1u[1] += yl[2 * i + 1] * (qu1[i] & 0x0F00);
                acc1u[2] += yl[2 * i + 8] * (qu1[i] & 0x00F0);
                acc1u[3] += yl[2 * i + 9] * (qu1[i] & 0xF000);
                acc2u[0] += yh[2 * i + 0] * (qu2[i] & 0x000F);
                acc2u[1] += yh[2 * i + 1] * (qu2[i] & 0x0F00);
                acc2u[2] += yh[2 * i + 8] * (qu2[i] & 0x00F0);
                acc2u[3] += yh[2 * i + 9] * (qu2[i] & 0xF000);
            }

            sumu[row] += dhu[0] * ((acc1u[0] + 1.f / 256.f * acc1u[1]) * sc8[0] +
                                   (acc1u[2] + 1.f / 256.f * acc1u[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2u[0] + 1.f / 256.f * acc2u[1]) * sc8[4] +
                                   (acc2u[2] + 1.f / 256.f * acc2u[3]) * sc8[5] * 1.f / 16.f) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.nb01 / 2;
            scg += args.nb01 / 2;
            dhg += args.nb01 / 2;
            qu1 += args.nb01 / 2;
            scu += args.nb01 / 2;
            dhu += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
        const float gate = simd_sum(sumg[row]);
        const float up = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

template<typename args_t>
void kernel_mul_mv_mxfp4_pair_swiglu_impl(
        args_t args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args &act,
        device const char *src0_gate,
        device const char *src0_up,
        device const char *src1,
        device char *dst_gate,
        device char *dst_up,
        device char *dst_mid,
        float route_weight,
        threadgroup char *shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_MXFP4;
    const int nb = args.ne00 / QK_MXFP4;
    const int row_blocks = (int)(args.nb01 / sizeof(block_mxfp4));
    const short ix = tiisg / 2;
    const short it = tiisg & 1;

    threadgroup float *lut = (threadgroup float *)shmem;
    if (sgitg == 0) lut[tiisg] = ds4_metal_mxfp4_values[tiisg & 15];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const block_mxfp4 *xg =
        (device const block_mxfp4 *)(src0_gate + (uint64_t)first_row * args.nb01);
    device const block_mxfp4 *xu =
        (device const block_mxfp4 *)(src0_up + (uint64_t)first_row * args.nb01);
    device const float *yb = (device const float *)src1 + ix * QK_MXFP4 + it * 8;
    float sumg[N_R0_MXFP4] = {0.f};
    float sumu[N_R0_MXFP4] = {0.f};

    for (int ib = ix; ib < nb; ib += 16) {
        device const float4 *y4 = (device const float4 *)yb;
        const float4 yl0 = y4[0];
        const float4 yl1 = y4[4];
        const float4 yl2 = y4[1];
        const float4 yl3 = y4[5];

        FOR_UNROLL (short row = 0; row < N_R0_MXFP4; row++) {
            device const block_mxfp4 &bg = xg[row * row_blocks + ib];
            device const block_mxfp4 &bu = xu[row * row_blocks + ib];
            device const uchar *qg = bg.qs + 8 * it;
            device const uchar *qu = bu.qs + 8 * it;

            float4 ag = yl0 * float4(lut[qg[0] & 15], lut[qg[1] & 15],
                                      lut[qg[2] & 15], lut[qg[3] & 15]);
            ag += yl1 * float4(lut[qg[0] >> 4], lut[qg[1] >> 4],
                               lut[qg[2] >> 4], lut[qg[3] >> 4]);
            ag += yl2 * float4(lut[qg[4] & 15], lut[qg[5] & 15],
                               lut[qg[6] & 15], lut[qg[7] & 15]);
            ag += yl3 * float4(lut[qg[4] >> 4], lut[qg[5] >> 4],
                               lut[qg[6] >> 4], lut[qg[7] >> 4]);

            float4 au = yl0 * float4(lut[qu[0] & 15], lut[qu[1] & 15],
                                      lut[qu[2] & 15], lut[qu[3] & 15]);
            au += yl1 * float4(lut[qu[0] >> 4], lut[qu[1] >> 4],
                               lut[qu[2] >> 4], lut[qu[3] >> 4]);
            au += yl2 * float4(lut[qu[4] & 15], lut[qu[5] & 15],
                               lut[qu[6] & 15], lut[qu[7] & 15]);
            au += yl3 * float4(lut[qu[4] >> 4], lut[qu[5] >> 4],
                               lut[qu[6] >> 4], lut[qu[7] >> 4]);

            sumg[row] += ds4_metal_e8m0_to_f32(bg.e) *
                         ((ag.x + ag.y) + (ag.z + ag.w));
            sumu[row] += ds4_metal_e8m0_to_f32(bu.e) *
                         ((au.x + au.y) + (au.z + au.w));
        }
        yb += 16 * QK_MXFP4;
    }

    device float *gate_f32 = (device float *)dst_gate;
    device float *up_f32 = (device float *)dst_up;
    device float *mid_f32 = (device float *)dst_mid;
    FOR_UNROLL (short row = 0; row < N_R0_MXFP4; row++) {
        if (first_row + row < args.ne0) {
            const float gate = simd_sum(sumg[row]);
            const float up = simd_sum(sumu[row]);
            if (tiisg == 0) {
                const uint out_row = first_row + row;
                float g = gate;
                float u = up;
                if (act.clamp_value > 1.0e-6f) {
                    g = min(g, act.clamp_value);
                    u = clamp(u, -act.clamp_value, act.clamp_value);
                }
                gate_f32[out_row] = gate;
                up_f32[out_row] = up;
                mid_f32[out_row] = (g / (1.0f + exp(-g))) * u * route_weight;
            }
        }
    }
}

kernel void kernel_mul_mv_id_mxfp4_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id &args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args &act,
        device const char *src0_gate,
        device const char *src0_up,
        device const char *src1,
        device char *dst_gate,
        device char *dst_up,
        device char *dst_mid,
        device const char *ids,
        device const char *weights,
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx = tgpig.z % args.nei0;
    const int32_t expert = ((device const int32_t *)(ids + (uint64_t)iid1 * args.nbi1))[idx];
    if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) return;

    const uint64_t pair_row = (uint64_t)iid1 * args.nei0 + (uint64_t)idx;
    device const float *route =
        (device const float *)(weights + pair_row * act.weight_stride);
    device char *gate_cur = dst_gate + pair_row * args.ne0 * sizeof(float);
    device char *up_cur = dst_up + pair_row * args.ne0 * sizeof(float);
    device char *mid_cur = dst_mid + pair_row * act.mid_row_stride;
    device const char *x_cur = src1 + (uint64_t)(idx % args.ne11) * args.nb11 +
        (uint64_t)iid1 * args.nb12;
    device const char *gate_expert = src0_gate +
        (int64_t)(expert - args.tp_expert_base) * args.nb02;
    device const char *up_expert = src0_up +
        (int64_t)(expert - args.tp_expert_base) * args.nb02;
    tgpig.z = 0;
    kernel_mul_mv_mxfp4_pair_swiglu_impl(args, act, gate_expert, up_expert,
                                         x_cur, gate_cur, up_cur, mid_cur,
                                         route[0], shmem, tgpig, tiisg, sgitg);
    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_mxfp4_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id &args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args &act,
        device const char *gate0, device const char *gate1,
        device const char *gate2, device const char *gate3,
        device const char *gate4, device const char *gate5,
        device const char *up0, device const char *up1,
        device const char *up2, device const char *up3,
        device const char *up4, device const char *up5,
        device const char *src1,
        device char *dst_gate,
        device char *dst_up,
        device char *dst_mid,
        device const char *weights,
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx = tgpig.z % args.nei0;
    device const char *gate_expert = gate0;
    device const char *up_expert = up0;
    switch (idx) {
    case 1: gate_expert = gate1; up_expert = up1; break;
    case 2: gate_expert = gate2; up_expert = up2; break;
    case 3: gate_expert = gate3; up_expert = up3; break;
    case 4: gate_expert = gate4; up_expert = up4; break;
    case 5: gate_expert = gate5; up_expert = up5; break;
    default: break;
    }

    const uint64_t pair_row = (uint64_t)iid1 * args.nei0 + (uint64_t)idx;
    device const float *route =
        (device const float *)(weights + pair_row * act.weight_stride);
    device char *gate_cur = dst_gate + pair_row * args.ne0 * sizeof(float);
    device char *up_cur = dst_up + pair_row * args.ne0 * sizeof(float);
    device char *mid_cur = dst_mid + pair_row * act.mid_row_stride;
    device const char *x_cur = src1 + (uint64_t)(idx % args.ne11) * args.nb11 +
        (uint64_t)iid1 * args.nb12;
    tgpig.z = 0;
    kernel_mul_mv_mxfp4_pair_swiglu_impl(args, act, gate_expert, up_expert,
                                         x_cur, gate_cur, up_cur, mid_cur,
                                         route[0], shmem, tgpig, tiisg, sgitg);
    (void)tiitg;
}

kernel void kernel_mul_mv_table_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ds4_metal_q4_expert_table & gate_table,
        device const ds4_metal_q4_expert_table & up_table,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = gate_table.experts[(uint)i02];
    device const char *src0_up_cur   = up_table.experts[(uint)i02];
    device const char *src1_cur      = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    const short NSG = FC_mul_mv_nsg;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    device const block_q4_K *xg =
        (device const block_q4_K *)(src0_gate_cur + (uint64_t)first_row * args.nb01);
    device const block_q4_K *xu =
        (device const block_q4_K *)(src0_up_cur + (uint64_t)first_row * args.nb01);
    device const float *y = (device const float *)src1_cur;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0_Q4_K] = {0.f};
    float sumu[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)xg[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint16_t *scu = (device const uint16_t *)xu[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0; row < N_R0_Q4_K; row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1g[0] += yl[2 * i + 0] * (qg1[i] & 0x000F);
                acc1g[1] += yl[2 * i + 1] * (qg1[i] & 0x0F00);
                acc1g[2] += yl[2 * i + 8] * (qg1[i] & 0x00F0);
                acc1g[3] += yl[2 * i + 9] * (qg1[i] & 0xF000);
                acc2g[0] += yh[2 * i + 0] * (qg2[i] & 0x000F);
                acc2g[1] += yh[2 * i + 1] * (qg2[i] & 0x0F00);
                acc2g[2] += yh[2 * i + 8] * (qg2[i] & 0x00F0);
                acc2g[3] += yh[2 * i + 9] * (qg2[i] & 0xF000);
            }

            sumg[row] += dhg[0] * ((acc1g[0] + 1.f / 256.f * acc1g[1]) * sc8[0] +
                                   (acc1g[2] + 1.f / 256.f * acc1g[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2g[0] + 1.f / 256.f * acc2g[1]) * sc8[4] +
                                   (acc2g[2] + 1.f / 256.f * acc2g[3]) * sc8[5] * 1.f / 16.f) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1u[0] += yl[2 * i + 0] * (qu1[i] & 0x000F);
                acc1u[1] += yl[2 * i + 1] * (qu1[i] & 0x0F00);
                acc1u[2] += yl[2 * i + 8] * (qu1[i] & 0x00F0);
                acc1u[3] += yl[2 * i + 9] * (qu1[i] & 0xF000);
                acc2u[0] += yh[2 * i + 0] * (qu2[i] & 0x000F);
                acc2u[1] += yh[2 * i + 1] * (qu2[i] & 0x0F00);
                acc2u[2] += yh[2 * i + 8] * (qu2[i] & 0x00F0);
                acc2u[3] += yh[2 * i + 9] * (qu2[i] & 0xF000);
            }

            sumu[row] += dhu[0] * ((acc1u[0] + 1.f / 256.f * acc1u[1]) * sc8[0] +
                                   (acc1u[2] + 1.f / 256.f * acc1u[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2u[0] + 1.f / 256.f * acc2u[1]) * sc8[4] +
                                   (acc2u[2] + 1.f / 256.f * acc2u[3]) * sc8[5] * 1.f / 16.f) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.nb01 / 2;
            scg += args.nb01 / 2;
            dhg += args.nb01 / 2;
            qu1 += args.nb01 / 2;
            scu += args.nb01 / 2;
            dhu += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
        const float gate = simd_sum(sumg[row]);
        const float up = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_addr_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addrs[(uint)i02]);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addrs[(uint)i02]);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_q4_gather_slots6(
        constant ds4_metal_q4_gather_slots6_args &args,
        device const char *src_group0,
        device const char *src_group1,
        device const char *src_group2,
        device const char *src_group3,
        device const char *src_group4,
        device const char *src_group5,
        device const int32_t *ids,
        device char *dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.n_slots || args.group_size == 0) return;

    const int32_t expert = ids[slot];
    if (expert < 0) return;

    const uint expert_u = (uint)expert;
    const uint group = expert_u / args.group_size;
    if (group >= 6) return;

    const uint local_expert = expert_u - group * args.group_size;
    device const char *src_group = src_group0;
    switch (group) {
    case 1: src_group = src_group1; break;
    case 2: src_group = src_group2; break;
    case 3: src_group = src_group3; break;
    case 4: src_group = src_group4; break;
    case 5: src_group = src_group5; break;
    default: break;
    }

    const uint64_t chunk = (uint64_t)tgpig.x * 256ul + (uint64_t)tiitg;
    const uint64_t n_chunks = args.expert_bytes >> 4;
    if (chunk >= n_chunks) return;

    device const uint4 *src =
        (device const uint4 *)(src_group + (uint64_t)local_expert * args.expert_bytes);
    device uint4 *out =
        (device uint4 *)(dst + (uint64_t)slot * args.expert_bytes);
    out[chunk] = src[chunk];
}

kernel void kernel_mul_mv_slots6_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (idx) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

static inline device const char *ds4_q4_group24_select(
        uint32_t group_id,
        device const char *src00,
        device const char *src01,
        device const char *src02,
        device const char *src03,
        device const char *src04,
        device const char *src05,
        device const char *src06,
        device const char *src07,
        device const char *src08,
        device const char *src09,
        device const char *src10,
        device const char *src11,
        device const char *src12,
        device const char *src13,
        device const char *src14,
        device const char *src15,
        device const char *src16,
        device const char *src17,
        device const char *src18,
        device const char *src19,
        device const char *src20,
        device const char *src21,
        device const char *src22,
        device const char *src23) {
    switch (group_id) {
    case 1:  return src01;
    case 2:  return src02;
    case 3:  return src03;
    case 4:  return src04;
    case 5:  return src05;
    case 6:  return src06;
    case 7:  return src07;
    case 8:  return src08;
    case 9:  return src09;
    case 10: return src10;
    case 11: return src11;
    case 12: return src12;
    case 13: return src13;
    case 14: return src14;
    case 15: return src15;
    case 16: return src16;
    case 17: return src17;
    case 18: return src18;
    case 19: return src19;
    case 20: return src20;
    case 21: return src21;
    case 22: return src22;
    case 23: return src23;
    default: return src00;
    }
}

kernel void kernel_mul_mv_group6_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 64;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (group_id >= 6) {
        return;
    }
    const uint32_t expert_local = expert_u - group_id * expert_group_size;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (group_id) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    src0_gate_cur += (uint64_t)expert_local * args.nb02;
    src0_up_cur   += (uint64_t)expert_local * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_group8_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_gate6,
        device const char * src0_gate7,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src0_up6,
        device const char * src0_up7,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 48;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (group_id >= 8) {
        return;
    }
    const uint32_t expert_local = expert_u - group_id * expert_group_size;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (group_id) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    case 6: src0_gate_cur = src0_gate6; src0_up_cur = src0_up6; break;
    case 7: src0_gate_cur = src0_gate7; src0_up_cur = src0_up7; break;
    default: break;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    src0_gate_cur += (uint64_t)expert_local * args.nb02;
    src0_up_cur   += (uint64_t)expert_local * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_group24_q4_K_id_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src06,
        device const char * src07,
        device const char * src08,
        device const char * src09,
        device const char * src10,
        device const char * src11,
        device const char * src12,
        device const char * src13,
        device const char * src14,
        device const char * src15,
        device const char * src16,
        device const char * src17,
        device const char * src18,
        device const char * src19,
        device const char * src20,
        device const char * src21,
        device const char * src22,
        device const char * src23,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 16;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (group_id >= 24) {
        return;
    }
    const uint32_t expert_local = expert_u - group_id * expert_group_size;

    device const char *src0_cur = ds4_q4_group24_select(group_id,
                                                        src00, src01, src02, src03,
                                                        src04, src05, src06, src07,
                                                        src08, src09, src10, src11,
                                                        src12, src13, src14, src15,
                                                        src16, src17, src18, src19,
                                                        src20, src21, src22, src23);
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    src0_cur += (uint64_t)expert_local * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_cur = dst + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    (void)tiitg;
}

kernel void kernel_mul_mv_group_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        constant ds4_metal_moe_expert_group_args & group,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert_global = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert_global < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert_global;
    if (expert_u < group.expert_base ||
        expert_u >= group.expert_base + group.expert_count) {
        return;
    }
    const uint32_t expert_local = expert_u - group.expert_base;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + (uint64_t)expert_local * args.nb02;
    device const char *src0_up_cur   = src0_up   + (uint64_t)expert_local * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}


/* IQ2_XXS down projection summed over the token's selected experts (the
 * GLM 5.2 routed blobs are IQ2_XXS end to end, unlike DS4 Flash's Q2_K
 * down).  Same contract as the q2_K sum kernel: mid rows already carry
 * silu(gate)*up*route_weight, expert ownership honors the TP split, and
 * add_in folds the shared-expert partial when tp_addend is set. */
kernel void kernel_mul_mv_id_iq2_xxs_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_IQ2_XXS;
    const int nb = args.ne00/QK_K;
    const int nb32 = nb * (QK_K / 32);
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};
    float yl[32];

    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) continue;
        device const block_iq2_xxs *x =
            (device const block_iq2_xxs *)(src0s + (int64_t)(expert - args.tp_expert_base)*args.nb02 + first_row*args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float *y4 = y + 32 * ix;

        for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
            for (short i = 0; i < 32; ++i) {
                yl[i] = y4[i];
            }

            const int ibl = ib32 / (QK_K / 32);
            const int ib  = ib32 % (QK_K / 32);

            device const block_iq2_xxs *xr = x + ibl;
            device const uint16_t *q2 = xr->qs + 4 * ib;
            device const half *dh = &xr->d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    device const uint8_t *aux8 = (device const uint8_t *)q2;
                    const uint32_t aux32 = q2[2] | (q2[3] << 16);
                    const float d = (float)dh[0] * (0.5f + (aux32 >> 28));

                    float s = 0;
                    for (short l = 0; l < 4; ++l) {
                        const threadgroup uint8_t *grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                        const uint8_t sign = ssigns[(aux32 >> 7 * l) & 127];
                        for (short j = 0; j < 8; ++j) {
                            s += yl[8 * l + j] * grid[j] *
                                 (sign & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                        }
                    }
                    sumf[row] += d * s;
                }
                dh += args.nb01 / 2;
                q2 += args.nb01 / 2;
            }

            y4 += 32 * 32;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]) * 0.25f;
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_id_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) continue;
        device const block_q2_K * x = (device const block_q2_K *)(src0s + (int64_t)(expert - args.tp_expert_base)*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        device const char *src0_cur = src00;
        switch (expert_slot) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const uint64_t * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;
    device const int32_t *token_ids =
        (device const int32_t *)(ids + (uint64_t)token * args.nbi1);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            continue;
        }
        const uint64_t addr = addrs[(uint)expert];
        if (addr == 0) {
            continue;
        }
        device const char *src0_cur =
            reinterpret_cast<device const char *>(addr);
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }
                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }
            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_q2_K_sum6_masked_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_stream_expert_split_args & split,
        device const uint64_t * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;
    device const int32_t *token_ids =
        (device const int32_t *)(ids + (uint64_t)token * args.nbi1);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        if ((split.active_mask & (1u << (uint)expert_slot)) == 0) {
            continue;
        }
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            continue;
        }
        const uint64_t addr = addrs[(uint)expert];
        if (addr == 0) {
            continue;
        }
        device const char *src0_cur =
            reinterpret_cast<device const char *>(addr);
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }
                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }
            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            if (split.accumulate) {
                dst_f32[first_row + row] += sum_all;
            } else {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

static inline float2 ds4_mxfp4_accumulate_rows(
        device const char *src0,
        uint64_t row_bytes,
        device const float *y,
        uint32_t n_cols,
        uint32_t first_row,
        uint32_t n_rows,
        threadgroup const float *lut,
        ushort tiisg) {
    const int nb = (int)(n_cols / QK_MXFP4);
    const int row_blocks = (int)(row_bytes / sizeof(block_mxfp4));
    const short ix = tiisg / 2;
    const short it = tiisg & 1;
    device const block_mxfp4 *x =
        (device const block_mxfp4 *)(src0 + (uint64_t)first_row * row_bytes);
    device const float *yb = y + ix * QK_MXFP4 + it * 8;
    float2 sums = 0.0f;

    for (int ib = ix; ib < nb; ib += 16) {
        device const float4 *y4 = (device const float4 *)yb;
        const float4 yl0 = y4[0];
        const float4 yl1 = y4[4];
        const float4 yl2 = y4[1];
        const float4 yl3 = y4[5];
        FOR_UNROLL (short row = 0; row < N_R0_MXFP4; row++) {
            if (first_row + row < n_rows) {
                device const block_mxfp4 &b = x[row * row_blocks + ib];
                device const uchar *q = b.qs + 8 * it;
                float4 acc = yl0 * float4(lut[q[0] & 15], lut[q[1] & 15],
                                          lut[q[2] & 15], lut[q[3] & 15]);
                acc += yl1 * float4(lut[q[0] >> 4], lut[q[1] >> 4],
                                    lut[q[2] >> 4], lut[q[3] >> 4]);
                acc += yl2 * float4(lut[q[4] & 15], lut[q[5] & 15],
                                    lut[q[6] & 15], lut[q[7] & 15]);
                acc += yl3 * float4(lut[q[4] >> 4], lut[q[5] >> 4],
                                    lut[q[6] >> 4], lut[q[7] >> 4]);
                sums[row] += ds4_metal_e8m0_to_f32(b.e) *
                             ((acc.x + acc.y) + (acc.z + acc.w));
            }
        }
        yb += 16 * QK_MXFP4;
    }
    return sums;
}

kernel void kernel_mul_mv_id_mxfp4_sum6_f32(
        constant ds4_metal_args_mul_mv_id &args,
        device const char *src0s,
        device const char *src1,
        device char *dst,
        device const char *ids,
        device const char *add_in,
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const uint32_t first_row = (uint32_t)((tgpig.x * NSG + sgitg) * N_R0_MXFP4);
    const uint32_t token = tgpig.y;
    device const int32_t *token_ids =
        (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;
    threadgroup float *lut = (threadgroup float *)shmem;
    if (sgitg == 0) lut[tiisg] = ds4_metal_mxfp4_values[tiisg & 15];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 sumf = 0.0f;
    for (int slot = 0; slot < args.nei0; slot++) {
        const int32_t expert = token_ids[slot];
        if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) continue;
        device const char *expert_base = src0s +
            (int64_t)(expert - args.tp_expert_base) * args.nb02;
        device const float *y =
            (device const float *)(token_src1 + (uint64_t)slot * args.nb11);
        sumf += ds4_mxfp4_accumulate_rows(expert_base, args.nb01, y,
                                           args.ne00, first_row, args.ne0,
                                           lut, tiisg);
    }

    device float *out = (device float *)(dst + (uint64_t)token * args.nb1);
    FOR_UNROLL (short row = 0; row < N_R0_MXFP4; row++) {
        if (first_row + row < (uint32_t)args.ne0) {
            const float value = simd_sum(sumf[row]);
            if (tiisg == 0) {
                out[first_row + row] = value +
                    (args.tp_addend ? ((device const float *)add_in)[first_row + row] : 0.0f);
            }
        }
    }
    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_mxfp4_sum6_f32(
        constant ds4_metal_args_mul_mv_id &args,
        device const char *src00, device const char *src01,
        device const char *src02, device const char *src03,
        device const char *src04, device const char *src05,
        device const char *src1,
        device char *dst,
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const uint32_t first_row = (uint32_t)((tgpig.x * NSG + sgitg) * N_R0_MXFP4);
    const uint32_t token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;
    threadgroup float *lut = (threadgroup float *)shmem;
    if (sgitg == 0) lut[tiisg] = ds4_metal_mxfp4_values[tiisg & 15];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 sumf = 0.0f;
    for (int slot = 0; slot < 6; slot++) {
        device const char *expert_base = src00;
        switch (slot) {
        case 1: expert_base = src01; break;
        case 2: expert_base = src02; break;
        case 3: expert_base = src03; break;
        case 4: expert_base = src04; break;
        case 5: expert_base = src05; break;
        default: break;
        }
        device const float *y =
            (device const float *)(token_src1 + (uint64_t)slot * args.nb11);
        sumf += ds4_mxfp4_accumulate_rows(expert_base, args.nb01, y,
                                           args.ne00, first_row, args.ne0,
                                           lut, tiisg);
    }

    device float *out = (device float *)(dst + (uint64_t)token * args.nb1);
    FOR_UNROLL (short row = 0; row < N_R0_MXFP4; row++) {
        if (first_row + row < (uint32_t)args.ne0) {
            const float value = simd_sum(sumf[row]);
            if (tiisg == 0) out[first_row + row] = value;
        }
    }
    (void)tiitg;
}

kernel void kernel_mul_mv_id_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) continue;
        device const block_q4_K *x =
            (device const block_q4_K *)(src0s + (int64_t)(expert - args.tp_expert_base) * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_moe_expert_group_args & group,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        if (expert_u < group.expert_base ||
            expert_u >= group.expert_base + group.expert_count) {
            continue;
        }
        const uint32_t expert_local = expert_u - group.expert_base;

        device const block_q4_K *x =
            (device const block_q4_K *)(src0s + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            if (group.accumulate) {
                dst_f32[first_row + row] += sum_all;
            } else {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_table_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const ds4_metal_q4_expert_table & table,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            return;
        }
        device const block_q4_K *x =
            (device const block_q4_K *)(table.experts[(uint)expert] + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const ulong * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            return;
        }
        device const char *expert_base =
            reinterpret_cast<device const char *>(addrs[(uint)expert]);
        device const block_q4_K *x =
            (device const block_q4_K *)(expert_base + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        device const char *src0_cur = src00;
        switch (expert_slot) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }
        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group6_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 64;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 6) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = src00;
        switch (group_id) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }

        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group8_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src06,
        device const char * src07,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 48;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 8) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = src00;
        switch (group_id) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        case 6: src0_cur = src06; break;
        case 7: src0_cur = src07; break;
        default: break;
        }

        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group24_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src06,
        device const char * src07,
        device const char * src08,
        device const char * src09,
        device const char * src10,
        device const char * src11,
        device const char * src12,
        device const char * src13,
        device const char * src14,
        device const char * src15,
        device const char * src16,
        device const char * src17,
        device const char * src18,
        device const char * src19,
        device const char * src20,
        device const char * src21,
        device const char * src22,
        device const char * src23,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 16;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 24) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = ds4_q4_group24_select(group_id,
                                                            src00, src01, src02, src03,
                                                            src04, src05, src06, src07,
                                                            src08, src09, src10, src11,
                                                            src12, src13, src14, src15,
                                                            src16, src17, src18, src19,
                                                            src20, src21, src22, src23);
        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

#define QK_NL 16

// Builds the compact per-expert work map used by batched MoE matmul. DS4 routes
// each token to a small fixed top-k list, so this turns token-major ids into
// expert-major slices that the tiled matmul can consume.
template<short ne20>
kernel void kernel_mul_mm_id_map0(
        constant ds4_metal_args_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        device        char * work,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg;

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;

    // Reuse the route-id staging memory after the map is complete to build a
    // compact list of non-empty 32-row matmul tiles. The old dispatch covered
    // every possible token tile for every expert, even though most experts
    // receive only a small fraction of the prompt rows.
    threadgroup uint16_t * tile_counts = (threadgroup uint16_t *) shmem;
    const uint16_t n_tiles = (uint16_t)((n_all + 31u) / 32u);
    tile_counts[ide] = n_tiles;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint32_t tile_base = 0;
    for (ushort i = 0; i < ide; i++) {
        tile_base += tile_counts[i];
    }

    device uint32_t * work_count = (device uint32_t *) work;
    device uint2 * work_items = (device uint2 *)(work + 8);
    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        work_items[tile_base + tile] = uint2((uint32_t)ide, tile * 32u);
    }
    if (ide + 1u == ntg) {
        work_count[0] = tile_base + n_tiles;
    }
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

// Host-visible map builders for the routed-expert counts used by DS4 graph
// shapes. Some arities are generic leftovers retained for nearby batch sizes.
template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

// Batched routed-expert matmul. It reads the expert-major map produced above,
// loads selected expert weights, and writes results back to token-major slots
// so the DS4 FFN can apply SwiGLU, weighting, and the down projection.
template<short NR1, typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        device const char * work,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    static_assert(NR1 == 32, "kernel_mul_mm_id accumulator layout supports only 32 routed rows");

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr int SA_BYTES = NR0 * NR1 * (int)sizeof(S0);

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + SA_BYTES);

    device const uint32_t * work_count = (device const uint32_t *) work;
    const uint32_t work_index = tgpig.x;
    if (work_index >= work_count[0]) {
        return;
    }
    device const uint2 * work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[work_index];
    const int im = (int)item.x;
    const int r0 = tgpig.y*NR0;
    const int r1 = (int)item.y;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    if (!ds4_tp_owns_expert(im, args.ne02, args.tp_rank, args.tp_world)) {
        /* Unowned expert under the TP split: zero this tile's output rows so
         * the downstream swiglu/sum stages stay unchanged. Each (token,slot)
         * row belongs to exactly one expert, so nothing else writes them. */
        for (short j = sgitg; j < nr1; j += 4) {
            const int idj = ids_i32[im*args.ne21 + r1 + j];

            const short ide = idj % args.ne20;
            const short idt = idj / args.ne20;

            device float * D = (device float *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;

            for (int i = tiisg; i < nr0; i += 32) {
                D[i] = 0.0f;
            }
        }
        return;
    }

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = (uint64_t)(im - args.tp_expert_base)*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

// Address-table variant used by SSD streaming.  The routing ids remain the
// model's original expert ids, but each expert's resident buffer is found via a
// GPU-address table instead of a contiguous full-layer tensor.
template<short NR1, typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id_addr(
        constant ds4_metal_args_mul_mm_id & args,
        device const uint64_t * src0_addrs,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    static_assert(NR1 == 32, "kernel_mul_mm_id_addr accumulator layout supports only 32 routed rows");

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr int SA_BYTES = NR0 * NR1 * (int)sizeof(S0);

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + SA_BYTES);

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const uint64_t base_addr = src0_addrs[im];
    if (base_addr == 0) {
        return;
    }
    device const char * src0 = reinterpret_cast<device const char *>(base_addr);

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const short offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0)) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

// Fused routed gate+up grouped matmul with the SwiGLU epilogue, generic over
// the expert quant block.  Both A streams share one staged B tile per k-step;
// each output keeps the exact MMA accumulation order of the separate GEMMs,
// and the epilogue matches kernel_dsv4_moe_swiglu_weight_f16, so the fused
// result is bit-identical to the unfused path.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm_id_pair_swiglu_f16_impl(
        constant ds4_metal_args_mul_mm_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst_mid,
        device const char * weights,
        device const char * work,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa_gate = (threadgroup half *)(shmem);
    threadgroup half *sa_up   = (threadgroup half *)(shmem + 4096);
    threadgroup half *sb      = (threadgroup half *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    device const uint32_t * work_count = (device const uint32_t *) work;
    const uint32_t work_index = tgpig.x;
    if (work_index >= work_count[0]) {
        return;
    }
    device const uint2 * work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[work_index];
    const int im = (int)item.x;
    const int r0 = tgpig.y*NR0;
    const int r1 = (int)item.y;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * xg =
        (device const block_q *)(src0_gate + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const block_q * xu =
        (device const block_q *)(src0_up + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];

    for (short i = 0; i < 8; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_gate;
        dequantize_func(xg, il, temp_gate);
        half4x4 temp_up;
        dequantize_func(xu, il, temp_up);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage the shared B tile and both expert A tiles in one pass: two
        // threadgroup barriers per k-step total instead of two per matrix.
        {
            const short sx_b = (tiitg%NL1);
            const short sy_b = (tiitg/NL1)/8;
            const short ly_b = (tiitg/NL1)%8;
            const short ib_b = 4*sx_b + sy_b;
            *(threadgroup half2x4 *)(sb + 64*ib_b + 8*ly_b) =
                (half2x4)(*((device float2x4 *) y));
        }

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa_gate + 64*ib + 8*ly + lx) = temp_gate[i/4][i%4];
            *(sa_up   + 64*ib + 8*ly + lx) = temp_up[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_gate = (sa_gate + 4*64*(sgitg%2));
        threadgroup const half * lsma_up   = (sa_up   + 4*64*(sgitg%2));
        threadgroup const half * lsmb      = (sb      + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma_g[i], lsma_gate + 64*i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_up   + 64*i, 8, 0, false);
            }
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_gate[i], mb[i/4], ma_g[i%4], mc_gate[i]);
                simdgroup_multiply_accumulate(mc_up[i],   mb[i/4], ma_u[i%4], mc_up[i]);
            }

            lsma_gate += 8*64;
            lsma_up   += 8*64;
            lsmb += 4*64;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        xg = (il < 2) ? xg + (2 + nl - 1)/nl : xg;
        xu = (il < 2) ? xu + (2 + nl - 1)/nl : xu;
        y += NK;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_gate = (threadgroup float *) shmem;
    threadgroup float * temp_up = temp_gate + NR0*NR1;
    threadgroup float * temp_gate_str =
        temp_gate + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
    threadgroup float * temp_up_str =
        temp_up + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc_gate[i], temp_gate_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        simdgroup_store(mc_up[i],   temp_up_str   + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float c = act.clamp_value;
    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device half *D = (device half *)(dst_mid +
            ((uint64_t)idt*args.ne1 + (uint64_t)ide)*act.mid_row_stride) + r0;
        device const float *w = (device const float *)(weights + (uint64_t)idj*act.weight_stride);
        const float route_weight = w[0];

        threadgroup float *Cg = temp_gate + j*NR0;
        threadgroup float *Cu = temp_up   + j*NR0;

        int i = tiisg;
        for (; i < nr0; i += 32) {
            float g = Cg[i];
            float u = Cu[i];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            D[i] = (half)(silu * u * route_weight);
        }
    }
}

typedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>) mul_mm_id_pair_swiglu_f16_iq2;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl<block_q4_K, QK_NL, dequantize_q4_K>) mul_mm_id_pair_swiglu_f16_q4;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl<block_q5_K, QK_NL, dequantize_q5_K>) mul_mm_id_pair_swiglu_f16_q5;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl<block_mxfp4, 2, dequantize_mxfp4>) mul_mm_id_pair_swiglu_f16_mxfp4;

// Host-visible fused routed pair matmuls for the DS4 expert quant formats.
template [[host_name("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16")]] kernel mul_mm_id_pair_swiglu_f16_iq2 kernel_mul_mm_id_pair_swiglu_f16_impl<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_q4_K_pair_swiglu_f16")]] kernel mul_mm_id_pair_swiglu_f16_q4 kernel_mul_mm_id_pair_swiglu_f16_impl<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_id_q5_K_pair_swiglu_f16")]] kernel mul_mm_id_pair_swiglu_f16_q5 kernel_mul_mm_id_pair_swiglu_f16_impl<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_id_mxfp4_pair_swiglu_f16")]] kernel mul_mm_id_pair_swiglu_f16_mxfp4 kernel_mul_mm_id_pair_swiglu_f16_impl<block_mxfp4, 2, dequantize_mxfp4>;

typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id;
typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>) mul_mm_id_f16_rhs;
typedef decltype(kernel_mul_mm_id<32, float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id_ff32;
typedef decltype(kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id_addr;
typedef decltype(kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>) mul_mm_id_addr_f16_rhs;

// Host-visible batched MoE matmul variants for the DS4 quant formats.
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q5_K,    QK_NL, dequantize_q5_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K,    QK_NL, dequantize_q6_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]]      kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]        kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q5_K,    QK_NL, dequantize_q5_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K,    QK_NL, dequantize_q6_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]]      kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]        kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_ff32")]]        kernel mul_mm_id_ff32 kernel_mul_mm_id<32, float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_ff32")]]        kernel mul_mm_id_ff32 kernel_mul_mm_id<32, float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q5_K, QK_NL, dequantize_q5_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_ff32")]]        kernel mul_mm_id_ff32 kernel_mul_mm_id<32, float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q6_K, QK_NL, dequantize_q6_K, float, float4x4, float, float2x4>;

template [[host_name("kernel_mul_mm_id_addr_q2_K_f32")]]    kernel mul_mm_id_addr kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_addr_q4_K_f32")]]    kernel mul_mm_id_addr kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_addr_mxfp4_f32")]]   kernel mul_mm_id_addr kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4, 2, dequantize_mxfp4, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_addr_q2_K_f16")]]    kernel mul_mm_id_addr_f16_rhs kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_addr_q4_K_f16")]]    kernel mul_mm_id_addr_f16_rhs kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_addr_mxfp4_f16")]]   kernel mul_mm_id_addr_f16_rhs kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4, 2, dequantize_mxfp4, half, half4x4, half, half2x4>;

#ifdef DS4_METAL_HAS_TENSOR
// Attention-output low-rank projection retained for Metal4 prefill. It uses
// the same direct-RHS idea as dense matmul: dequantize the Q8_0 or Q4_K low
// projection weights to a half tile, then let TensorOps read the dense head
// activations directly. Only the 64-token direct-RHS instantiations are
// exported because the staged-RHS and 32-token variants were benchmark-only
// experiments.
//
// Full tiles (the host dispatch guarantee for aligned batches) skip all bounds
// work.  The weight tile is double-buffered: the next k-step's dequant
// overlaps the current cooperative matmul, so the k-loop needs one
// threadgroup barrier per step instead of two.
template<typename block_q, short qnl,
         void (*dequantize_func)(device const block_q *, short, thread half4x4 &),
         short NR1>
kernel void kernel_attn_out_low_mpp_direct_rhs(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne21;
    const int G = args.ne1;
    const int group = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;
    const bool full_tile = r0 + NR0 <= M && r1 + NR1 <= N && (K % NK) == 0;

    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA0 = tensor(sa,          dextents<int32_t, 2>(NK, NR0));
    auto tA1 = tensor(sa + NR0*NK, dextents<int32_t, 2>(NK, NR0));

    device float *ptrB = (device float *)(srcB + args.nb11*group);
    const int strideB = args.nb12/sizeof(float);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA0), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    auto stage_tile = [&](const int loop_k, threadgroup half *buf) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (full_tile) {
                device const block_q *row_ptr =
                    (device const block_q *)(srcA + args.nb01*(r0 + row) + group*args.nb02);

                half4x4 temp_a;
                dequantize_func(row_ptr + k_pos/(16*qnl), (k_pos/16)%qnl, temp_a);
                threadgroup half4 *dst4 = (threadgroup half4 *)(buf + row*NK + k_base);
                dst4[0] = temp_a[0];
                dst4[1] = temp_a[1];
                dst4[2] = temp_a[2];
                dst4[3] = temp_a[3];
            } else if (r0 + row < M) {
                const int block_idx = k_pos/(16*qnl);
                const short il = (k_pos/16)%qnl;
                device const block_q *row_ptr =
                    (device const block_q *)(srcA + args.nb01*(r0 + row) + group*args.nb02);

                half4x4 temp_a;
                dequantize_func(row_ptr + block_idx, il, temp_a);
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    buf[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (half)0;
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    buf[row*NK + k_base + i] = (half)0;
                }
            }
        }
    };

    stage_tile(0, sa);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_sel = 0;
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        auto mA = (buf_sel ? tA1 : tA0).slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        const int next_k = loop_k + NK;
        if (next_k < K) {
            buf_sel ^= 1u;
            stage_tile(next_k, buf_sel ? sa + NR0*NK : sa);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_group = (device float *)dst + group*M;
    if (full_tile) {
        device float *dst_tile = dst_group + r0 + (uint64_t)r1*G*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, G*M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_group, dextents<int32_t, 2>(M, N), array<int, 2>({1, G*M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

// Routed-expert grouped matmul on the Metal4 TensorOps/MPP pipeline. The
// barrier after mm.run prevents the next K iteration from replacing staged
// tiles while the cooperative matmul still reads them.
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id_mpp(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        device const char * work,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    device const uint32_t *work_count = (device const uint32_t *)work;
    const uint32_t work_index = tgpig.x;
    if (work_index >= work_count[0]) {
        return;
    }
    device const uint2 *work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[work_index];
    const int im = (int)item.x;
    const int r0 = tgpig.y*NR0;
    const int r1 = (int)item.y;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    if (!ds4_tp_owns_expert(im, args.ne02, args.tp_rank, args.tp_world)) {
        for (short j = sgitg; j < nr1; j += 4) {
            const int idj = ids_i32[im*args.ne21 + r1 + j];
            const short ide = idj % args.ne20;
            const short idt = idj / args.ne20;
            device float *D = (device float *)dst + r0 + ide*args.ne0 +
                              idt*args.ne1*args.ne0;
            for (int i = tiisg; i < nr0; i += 32) D[i] = 0.0f;
        }
        return;
    }

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 =
        (uint64_t)(im - args.tp_expert_base)*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NR1, NK));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) =
                    loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;
                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                *(sb + NK*(8*sy + ly) + 8*sx + lx) =
                    loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) =
                (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = tiitg/32; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}



typedef decltype(kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id_mpp_t;
typedef decltype(kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>) mul_mm_id_mpp_f16_rhs_t;

template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_mpp")]] kernel mul_mm_id_mpp_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16_mpp")]]    kernel mul_mm_id_mpp_f16_rhs_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16_mpp")]] kernel mul_mm_id_mpp_f16_rhs_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4x4, half, half2x4>;

typedef decltype(kernel_attn_out_low_mpp_direct_rhs<
        block_q8_0, 2, dequantize_q8_0_pairs, 64>)
    attn_out_low_q8_0_mpp_direct_rhs_n64_t;
typedef decltype(kernel_attn_out_low_mpp_direct_rhs<
        block_q4_K, QK_NL, dequantize_q4_K, 64>)
    attn_out_low_q4_K_mpp_direct_rhs_n64_t;

template [[host_name("kernel_attn_out_low_q8_0_mpp_direct_rhs_n64")]]
kernel attn_out_low_q8_0_mpp_direct_rhs_n64_t
kernel_attn_out_low_mpp_direct_rhs<
        block_q8_0, 2, dequantize_q8_0_pairs, 64>;
template [[host_name("kernel_attn_out_low_q4_K_mpp_direct_rhs_n64")]]
kernel attn_out_low_q4_K_mpp_direct_rhs_n64_t
kernel_attn_out_low_mpp_direct_rhs<
        block_q4_K, QK_NL, dequantize_q4_K, 64>;

#endif

#undef QK_NL
#undef kmask_iq2xs
#undef ksigns_iq2xs
#undef iq2xxs_grid
#undef QK_K
#undef N_R0_Q2_K
#undef N_R0_Q4_K
#undef N_R0_Q8_K
#undef N_R0_GLM_Q4_PAIR2_K
#undef N_R0_GLM_Q4_PAIR_K
#undef N_R0_Q5_PAIR_K
#undef N_R0_Q5_K
#undef N_R0_Q6_K
#undef N_R0_IQ2_XXS

// Native 4K continuation tails visit output tiles before advancing the expert
// work tile. Only grid coordinates differ from the imported fused IQ2 kernel;
// every output preserves its dequantization, MMA and SwiGLU evaluation order.
// Compiled after metal/moe.metal to share its types and dequantization tables.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void rust_star_routed_prefill_gate_up_impl(
        constant ds4_metal_args_mul_mm_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst_mid,
        device const char * weights,
        device const char * work,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa_gate = (threadgroup half *)(shmem);
    threadgroup half *sa_up   = (threadgroup half *)(shmem + 4096);
    threadgroup half *sb      = (threadgroup half *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    device const uint32_t * work_count = (device const uint32_t *) work;
    const uint32_t work_index = tgpig.y;
    if (work_index >= work_count[0]) {
        return;
    }
    device const uint2 * work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[work_index];
    const int im = (int)item.x;
    const int r0 = tgpig.x*NR0;
    const int r1 = (int)item.y;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * xg =
        (device const block_q *)(src0_gate + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const block_q * xu =
        (device const block_q *)(src0_up + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];

    for (short i = 0; i < 8; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_gate;
        dequantize_func(xg, il, temp_gate);
        half4x4 temp_up;
        dequantize_func(xu, il, temp_up);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage the shared B tile and both expert A tiles in one pass: two
        // threadgroup barriers per k-step total instead of two per matrix.
        {
            const short sx_b = (tiitg%NL1);
            const short sy_b = (tiitg/NL1)/8;
            const short ly_b = (tiitg/NL1)%8;
            const short ib_b = 4*sx_b + sy_b;
            *(threadgroup half2x4 *)(sb + 64*ib_b + 8*ly_b) =
                (half2x4)(*((device float2x4 *) y));
        }

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa_gate + 64*ib + 8*ly + lx) = temp_gate[i/4][i%4];
            *(sa_up   + 64*ib + 8*ly + lx) = temp_up[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_gate = (sa_gate + 4*64*(sgitg%2));
        threadgroup const half * lsma_up   = (sa_up   + 4*64*(sgitg%2));
        threadgroup const half * lsmb      = (sb      + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma_g[i], lsma_gate + 64*i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_up   + 64*i, 8, 0, false);
            }
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_gate[i], mb[i/4], ma_g[i%4], mc_gate[i]);
                simdgroup_multiply_accumulate(mc_up[i],   mb[i/4], ma_u[i%4], mc_up[i]);
            }

            lsma_gate += 8*64;
            lsma_up   += 8*64;
            lsmb += 4*64;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        xg = (il < 2) ? xg + (2 + nl - 1)/nl : xg;
        xu = (il < 2) ? xu + (2 + nl - 1)/nl : xu;
        y += NK;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_gate = (threadgroup float *) shmem;
    threadgroup float * temp_up = temp_gate + NR0*NR1;
    threadgroup float * temp_gate_str =
        temp_gate + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
    threadgroup float * temp_up_str =
        temp_up + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc_gate[i], temp_gate_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        simdgroup_store(mc_up[i],   temp_up_str   + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float c = act.clamp_value;
    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device half *D = (device half *)(dst_mid +
            ((uint64_t)idt*args.ne1 + (uint64_t)ide)*act.mid_row_stride) + r0;
        device const float *w = (device const float *)(weights + (uint64_t)idj*act.weight_stride);
        const float route_weight = w[0];

        threadgroup float *Cg = temp_gate + j*NR0;
        threadgroup float *Cu = temp_up   + j*NR0;

        int i = tiisg;
        for (; i < nr0; i += 32) {
            float g = Cg[i];
            float u = Cu[i];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            D[i] = (half)(silu * u * route_weight);
        }
    }
}

// IQ2 uses 16 dequantization slices; the imported file undefines its QK_NL macro.
typedef decltype(rust_star_routed_prefill_gate_up_impl<block_iq2_xxs, 16, dequantize_iq2_xxs>) rust_star_routed_prefill_gate_up_type;
template [[host_name("rust_star_routed_prefill_gate_up")]] kernel rust_star_routed_prefill_gate_up_type rust_star_routed_prefill_gate_up_impl<block_iq2_xxs, 16, dequantize_iq2_xxs>;

// Same dense kernel template and accumulator types; only the RHS load type
// changes from float/float2x4 to half/half2x4 after explicit conversion.
typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, half, half2x4>) rust_star_attention_output_half_t;
template [[host_name("rust_star_attention_output_half")]] kernel rust_star_attention_output_half_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, half, half2x4>;

struct ds4_metal_args_dsv4_hc_split_sinkhorn {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_weighted_sum {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_hc_weighted_sum_norm {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float    norm_eps;
};

struct ds4_metal_args_dsv4_output_hc_weights4 {
    float post_scale;
    float eps;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum_norm {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float    eps;
    float    norm_eps;
};

struct ds4_metal_args_dsv4_hc_expand {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_add0;
    uint64_t nb_add1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    int32_t  has_add;
};

// Numerically stable sigmoid for the standalone split/sinkhorn path. The naive
// form 1/(1+exp(-z)) overflows for large negative z (exp(-z) blows up);
// replacing it with the 0.5*(tanh(z/2)+1) identity keeps the value bounded in
// [0, 1] across the entire float range. Gated by DS4_METAL_HC_STABLE so we can
// A/B vs the historical form on M5 Max where the faster ALU is more likely to
// push HC mixer inputs into the unstable regime.
//
// Do not automatically use these helpers in the fused HC decode kernels below:
// routing the fused vector sites through the tanh form produced non-finite
// logits on M5 Max, while the historical inline exp form remains finite and is
// the decode throughput baseline.
#ifdef DS4_METAL_HC_STABLE
static inline float  ds4_hc_sigmoid(float  z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
// 2 * sigmoid(z) == 1 + tanh(z/2).
static inline float  ds4_hc_twice_sigmoid(float  z) { return 1.0f + tanh(0.5f * z); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 1.0f + tanh(0.5f * z); }
#else
static inline float  ds4_hc_sigmoid(float  z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float  ds4_hc_twice_sigmoid(float  z) { return 2.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 2.0f / (1.0f + exp(-z)); }
#endif

// Splits an HC mixer row into pre weights, post gates, and the HC-to-HC
// combination matrix. The 4-channel path is specialized because DS4 Flash uses
// HC=4 in normal inference, while the scalar fallback keeps diagnostics usable.
kernel void kernel_dsv4_hc_split_sinkhorn(
        constant ds4_metal_args_dsv4_hc_split_sinkhorn & args,
        device  const float * mixes,
        device  const float * scale,
        device  const float * base,
        device        float * dst,
        uint tid [[thread_position_in_grid]]) {
    if ((int64_t) tid >= args.n_rows) {
        return;
    }

    constexpr int HC_MAX = 16;
    const int HC = args.n_hc;
    if (HC <= 0 || HC > HC_MAX) {
        return;
    }

    device const float * mix = mixes + ((int64_t) tid)*args.mix_hc;
    device       float * out = dst    + ((int64_t) tid)*args.mix_hc;

    const float epsv       = args.eps;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (HC == 4) {
        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        *((device float4 *) out) = ds4_hc_sigmoid(pre_z) + epsv;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = ds4_hc_twice_sigmoid(post_z);

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
        return;
    }

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = ds4_hc_sigmoid(z) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = ds4_hc_twice_sigmoid(z);
    }

    float c[HC_MAX*HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const int off = 2*HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = max(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const float v = exp(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc*HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc*HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC*HC; ++i) {
        out[2*HC + i] = c[i];
    }
}

// Decode-side fusion of HC split and pre-weighted HC reduction. One threadgroup
// handles one token row: lane 0 computes the HC=4 mixer split once, stores the
// post/comb data for the following HC expand, and all lanes reuse the pre
// weights from threadgroup memory to produce the embedding row.
kernel void kernel_dsv4_hc_split_weighted_sum(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        threadgroup   float * pre_shmem [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if ((int64_t) row >= args.n_rows || args.n_hc != 4) {
        return;
    }

    device const float * mix = (device const float *) (mixes + (uint64_t)row*args.nb_mix1);
    device       float * out = (device       float *) (split + (uint64_t)row*args.nb_split1);

    if (tid == 0) {
        const float epsv       = args.eps;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *) out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int64_t d = tid; d < args.n_embd; d += ntg) {
        float acc = 0.0f;
        acc += *((device const float *) (x + d*args.nb_x0 + 0*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[0];
        acc += *((device const float *) (x + d*args.nb_x0 + 1*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[1];
        acc += *((device const float *) (x + d*args.nb_x0 + 2*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[2];
        acc += *((device const float *) (x + d*args.nb_x0 + 3*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[3];
        *((device float *) (dst + d*args.nb0 + (uint64_t)row*args.nb1)) = acc;
    }
}

// Decode HC-pre plus the following RMSNorm.  DS4 uses HC=4 here.  The normal
// release path computes HC coefficients, collapses four residual streams into
// the model row, then immediately launches a weighted RMSNorm over the row.
// This kernel keeps the HC split math identical to
// kernel_dsv4_hc_split_weighted_sum, stores the HC-pre row for diagnostics, and
// reuses the just-collapsed values from threadgroup memory for the RMSNorm
// reduction.
kernel void kernel_dsv4_hc_split_weighted_sum_norm4(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        device  const char  * norm_weight,
        device        char  * norm_dst,
        threadgroup   float * shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if ((int64_t)row >= args.n_rows || args.n_hc != 4 || (args.n_embd & 3) != 0) {
        return;
    }

    const uint n_embd = uint(args.n_embd);
    const uint n4 = n_embd >> 2;
    threadgroup float4 *row_shmem = (threadgroup float4 *)shared;
    threadgroup float *pre_shmem = shared + n_embd;
    threadgroup float *sum_shmem = pre_shmem + 4;

    device const float *mix = (device const float *)(mixes + (uint64_t)row * args.nb_mix1);
    device float *out = (device float *)(split + (uint64_t)row * args.nb_split1);

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    if (tid == 0) {
        const float epsv = args.eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *)mix) * pre_scale +
            *((device const float4 *)base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *)(mix + 4)) * post_scale +
            *((device const float4 *)(base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *)(mix + 8)) * comb_scale +
            *((device const float4 *)(base + 8));
        float4 r1 =
            *((device const float4 *)(mix + 12)) * comb_scale +
            *((device const float4 *)(base + 12));
        float4 r2 =
            *((device const float4 *)(mix + 16)) * comb_scale +
            *((device const float4 *)(base + 16));
        float4 r3 =
            *((device const float4 *)(mix + 20)) * comb_scale +
            *((device const float4 *)(base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x1 = (device const float4 *)(x + 1 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x2 = (device const float4 *)(x + 2 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x3 = (device const float4 *)(x + 3 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        // Preserve the standalone HC collapse's explicit accumulation order.
        float4 v = 0.0f;
        v += x0[i] * pre_shmem[0];
        v += x1[i] * pre_shmem[1];
        v += x2[i] * pre_shmem[2];
        v += x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    // Batched prefill must match kernel_rms_norm_fuse_impl so enabling this
    // fusion does not change its scale by an ULP. Keep the established
    // single-row decode result unchanged; that path historically used rsqrt.
    const float norm_arg = sumf / float(n_embd) + args.norm_eps;
    const float norm_scale = args.n_rows > 1 ? 1.0f / sqrt(norm_arg) : rsqrt(norm_arg);

    device float4 *dst4 = (device float4 *)(dst + (uint64_t)row * args.nb1);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)(norm_dst + (uint64_t)row * args.nb_norm1);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}

// Expands an embedding-sized block back into HC channels after attention/FFN.
// The post gate scales the current block, while the Sinkhorn combination matrix
// mixes residual HC channels from the previous state.
kernel void kernel_dsv4_hc_expand(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d      = ((int64_t) gid) % args.n_embd;
    const int64_t tmp    = ((int64_t) gid) / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t      = tmp / args.n_hc;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }
    const float post_v  = *((device const float *) (post      + dst_hc*args.nb_post0 + t*args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((device const float *) (comb     + dst_hc*args.nb_comb0 + src_hc*args.nb_comb1 + t*args.nb_comb2));
        const float res_v  = *((device const float *) (residual + d*args.nb_res0 + src_hc*args.nb_res1 + t*args.nb_res2));
        acc += comb_v * res_v;
    }

    *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
}

// HC=4 specialization of the post/expand step. One thread computes all four
// destination HC streams for one token/dimension, reusing the same block output
// and residual HC values while preserving the per-stream accumulation order.
kernel void kernel_dsv4_hc_expand4(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    if (args.n_hc != 4) {
        return;
    }

    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }

    const float r0 = *((device const float *) (residual + d*args.nb_res0 + 0*args.nb_res1 + t*args.nb_res2));
    const float r1 = *((device const float *) (residual + d*args.nb_res0 + 1*args.nb_res1 + t*args.nb_res2));
    const float r2 = *((device const float *) (residual + d*args.nb_res0 + 2*args.nb_res1 + t*args.nb_res2));
    const float r3 = *((device const float *) (residual + d*args.nb_res0 + 3*args.nb_res1 + t*args.nb_res2));

    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * *((device const float *) (post + dst_hc*args.nb_post0 + t*args.nb_post1));

        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 0*args.nb_comb1 + t*args.nb_comb2)) * r0;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 1*args.nb_comb1 + t*args.nb_comb2)) * r1;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 2*args.nb_comb1 + t*args.nb_comb2)) * r2;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 3*args.nb_comb1 + t*args.nb_comb2)) * r3;

        *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
    }
}

// Decode-time FFN tail fusion:
//
//     shared_out = shared_mid @ Wshared_down
//     after_ffn_hc = HCPost(routed_out + shared_out, residual_hc, split)
//
// The Q8_0 dot reduction is intentionally copied from the normal matvec shape
// so the shared expert result is bit-identical.  The only specialization is
// that DS4 decode has one token and HC=4, so the thread that finishes each
// shared-down output row can immediately expand it into the four HC streams.
kernel void kernel_dsv4_shared_down_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * shared_mid,
        device        char * shared_out,
        device  const char * routed_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(shared_mid);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float shared_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(shared_out + (uint64_t)d * sizeof(float))) = shared_v;

            float block_v = *((device const float *)(routed_out + (uint64_t)d * hc.nb_block0));
            block_v += shared_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

// Decode-time attention output tail fusion:
//
//     attn_out = attn_low @ Wob
//     after_attn_hc = HCPost(attn_out, residual_hc, split)
//
// This is the no-add sibling of the shared-down/FFN fusion above.  It preserves
// the exact Q8_0 matvec reduction, stores `attn_out` for diagnostics, and then
// writes the four HC streams for the same embedding dimension.
kernel void kernel_dsv4_q8_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * input,
        device        char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(input);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float block_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(block_out + (uint64_t)d * sizeof(float))) = block_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

// Reduces HC channels to a normal embedding row with the learned pre weights.
// This is the input adapter before the attention block and before the FFN block.
kernel void kernel_dsv4_hc_weighted_sum(
        constant ds4_metal_args_dsv4_hc_weighted_sum & args,
        device  const char * x,
        device  const char * weights,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < args.n_hc; ++h) {
        const float xv = *((device const float *) (x       + d*args.nb_x0 + h*args.nb_x1 + t*args.nb_x2));
        const float wv = *((device const float *) (weights + h*args.nb_w0 + t*args.nb_w1));
        acc += xv * wv;
    }

    *((device float *) (dst + d*args.nb0 + t*args.nb1)) = acc;
}

// The one-row output head immediately applies a learned RMSNorm after reducing
// its four HC streams. Preserve the standalone scalar HC accumulation, write
// the collapsed row for diagnostics, then reload its F32 values from
// threadgroup memory using the standalone RMSNorm's float4 reduction mapping.
kernel void kernel_dsv4_hc_weighted_sum_norm4(
        constant ds4_metal_args_dsv4_hc_weighted_sum_norm & args,
        device  const char * x,
        device  const char * weights,
        device        char * dst,
        device  const char * norm_weight,
        device        char * norm_dst,
        threadgroup  float * shared [[threadgroup(0)]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if (args.n_tokens != 1 || args.n_hc != 4 ||
        args.n_embd <= 0 || (args.n_embd & 3) != 0) {
        return;
    }

    const uint n_embd = uint(args.n_embd);
    const uint n4 = n_embd >> 2;
    threadgroup float *row_shmem = shared;
    threadgroup float *sum_shmem = shared + n_embd;

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    for (uint d = tid; d < n_embd; d += ntg) {
        float acc = 0.0f;
        for (int64_t h = 0; h < args.n_hc; ++h) {
            const float xv = *((device const float *)(
                x + (uint64_t)d*args.nb_x0 + (uint64_t)h*args.nb_x1));
            const float wv = *((device const float *)(
                weights + (uint64_t)h*args.nb_w0));
            acc += xv * wv;
        }
        row_shmem[d] = acc;
        *((device float *)(dst + (uint64_t)d*args.nb0)) = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const float4 *row4 =
        (threadgroup const float4 *)row_shmem;
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        sumf += dot(row4[i], row4[i]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);

    const float mean = sumf/args.n_embd;
    const float scale = 1.0f/sqrt(mean + args.norm_eps);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)norm_dst;
    for (uint i = tid; i < n4; i += ntg) {
        norm4[i] = (row4[i]*scale)*w4[i];
    }
}

// The one-row HC=4 output head historically materializes four device-F32
// stages across separate launches. Collapse those launches into one tiny
// two-thread group while preserving the scalar/vector lane mapping and every
// global rounding boundary.
kernel void kernel_dsv4_output_hc_weights4(
        constant ds4_metal_args_dsv4_output_hc_weights4 & args,
        device  const float * pre,
        device  const float * hc_scale,
        device  const float * hc_base,
        device        float * dst,
        ushort tid [[thread_position_in_threadgroup]]) {
    device volatile float *stage = (device volatile float *)dst;

    for (uint i = tid; i < 4; i += 2) {
        stage[i] = pre[i] * hc_scale[0];
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint i = tid; i < 4; i += 2) {
        stage[i] = stage[i] + hc_base[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        const float4 x = *((device volatile float4 *)stage);
        *((device volatile float4 *)stage) = 1 / (1 + exp(-x));
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        const float4 x = *((device volatile float4 *)stage);
        *((device volatile float4 *)stage) =
            args.post_scale * x + args.eps;
    }
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

// Native 4K continuation tails visit output tiles before advancing the expert
// work tile. Only grid coordinates differ from the imported fused IQ2 kernel;
// every output preserves its dequantization, MMA and SwiGLU evaluation order.
// Compiled after metal/moe.metal to share its types and dequantization tables.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void probe_routed_half_gate_up_impl(
        constant ds4_metal_args_mul_mm_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst_mid,
        device const char * weights,
        device const char * work,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa_gate = (threadgroup half *)(shmem);
    threadgroup half *sa_up   = (threadgroup half *)(shmem + 4096);
    threadgroup half *sb      = (threadgroup half *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    device const uint32_t * work_count = (device const uint32_t *) work;
    const uint32_t work_index = tgpig.y;
    if (work_index >= work_count[0]) {
        return;
    }
    device const uint2 * work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[work_index];
    const int im = (int)item.x;
    const int r0 = tgpig.x*NR0;
    const int r1 = (int)item.y;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * xg =
        (device const block_q *)(src0_gate + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const block_q * xu =
        (device const block_q *)(src0_up + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const half * y = (device const half *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];

    for (short i = 0; i < 8; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_gate;
        dequantize_func(xg, il, temp_gate);
        half4x4 temp_up;
        dequantize_func(xu, il, temp_up);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage the shared B tile and both expert A tiles in one pass: two
        // threadgroup barriers per k-step total instead of two per matrix.
        {
            const short sx_b = (tiitg%NL1);
            const short sy_b = (tiitg/NL1)/8;
            const short ly_b = (tiitg/NL1)%8;
            const short ib_b = 4*sx_b + sy_b;
            *(threadgroup half2x4 *)(sb + 64*ib_b + 8*ly_b) =
                *((device half2x4 *) y);
        }

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa_gate + 64*ib + 8*ly + lx) = temp_gate[i/4][i%4];
            *(sa_up   + 64*ib + 8*ly + lx) = temp_up[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_gate = (sa_gate + 4*64*(sgitg%2));
        threadgroup const half * lsma_up   = (sa_up   + 4*64*(sgitg%2));
        threadgroup const half * lsmb      = (sb      + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma_g[i], lsma_gate + 64*i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_up   + 64*i, 8, 0, false);
            }
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_gate[i], mb[i/4], ma_g[i%4], mc_gate[i]);
                simdgroup_multiply_accumulate(mc_up[i],   mb[i/4], ma_u[i%4], mc_up[i]);
            }

            lsma_gate += 8*64;
            lsma_up   += 8*64;
            lsmb += 4*64;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        xg = (il < 2) ? xg + (2 + nl - 1)/nl : xg;
        xu = (il < 2) ? xu + (2 + nl - 1)/nl : xu;
        y += NK;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_gate = (threadgroup float *) shmem;
    threadgroup float * temp_up = temp_gate + NR0*NR1;
    threadgroup float * temp_gate_str =
        temp_gate + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
    threadgroup float * temp_up_str =
        temp_up + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc_gate[i], temp_gate_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        simdgroup_store(mc_up[i],   temp_up_str   + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float c = act.clamp_value;
    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device half *D = (device half *)(dst_mid +
            ((uint64_t)idt*args.ne1 + (uint64_t)ide)*act.mid_row_stride) + r0;
        device const float *w = (device const float *)(weights + (uint64_t)idj*act.weight_stride);
        const float route_weight = w[0];

        threadgroup float *Cg = temp_gate + j*NR0;
        threadgroup float *Cu = temp_up   + j*NR0;

        int i = tiisg;
        for (; i < nr0; i += 32) {
            float g = Cg[i];
            float u = Cu[i];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            D[i] = (half)(silu * u * route_weight);
        }
    }
}

// IQ2 uses 16 dequantization slices; the imported file undefines its QK_NL macro.
typedef decltype(probe_routed_half_gate_up_impl<block_iq2_xxs, 16, dequantize_iq2_xxs>) probe_routed_half_gate_up_type;
template [[host_name("probe_routed_half_gate_up")]] kernel probe_routed_half_gate_up_type probe_routed_half_gate_up_impl<block_iq2_xxs, 16, dequantize_iq2_xxs>;
