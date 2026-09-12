#include <stdint.h>
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
