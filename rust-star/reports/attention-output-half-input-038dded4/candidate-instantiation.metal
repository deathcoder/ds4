// Same dense kernel template and accumulator types; only the RHS load type
// changes from float/float2x4 to half/half2x4 after explicit conversion.
typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, half, half2x4>) probe_q8_half_t;
template [[host_name("probe_q8_output_half")]] kernel probe_q8_half_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, half, half2x4>;
