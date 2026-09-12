#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define QK8_0 32
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

// Imported verbatim from pinned DwarfStar's decode Q-A/KV pair. Both banks
// preserve the standalone Q8_0 traversal and reduction order; only activation
// loading and threadgroup scheduling are shared.
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
