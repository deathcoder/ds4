#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_shim.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const uint64_t kMaxElements = 16ull * 1024ull * 1024ull;
static const uint64_t kMaxIterations = 100000ull;
static const uint64_t kMaxThreadInvocations = 1000000000ull;

static NSString *const kProbeSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"kernel void rust_star_dispatch_probe(device uint *values [[buffer(0)]],\n"
    @"    uint gid [[thread_position_in_grid]]) {\n"
    @"    values[gid] = values[gid] * 1664525u + 1013904223u;\n"
    @"}\n";

// Imported from DwarfStar metal/get_rows.metal. Keeping its argument layout,
// dispatch geometry, and conversion expression unchanged makes this the first
// independently testable production kernel rather than a look-alike probe.
static NSString *const kGetRowsF16Source =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct ds4_metal_args_get_rows {\n"
    @"    int ne00t;\n"
    @"    int ne00;\n"
    @"    ulong nb01;\n"
    @"    ulong nb02;\n"
    @"    ulong nb03;\n"
    @"    int ne10;\n"
    @"    ulong nb10;\n"
    @"    ulong nb11;\n"
    @"    ulong nb12;\n"
    @"    ulong nb1;\n"
    @"    ulong nb2;\n"
    @"    ulong nb3;\n"
    @"};\n"
    @"template<typename T0, typename T>\n"
    @"kernel void kernel_get_rows_f(\n"
    @"        constant ds4_metal_args_get_rows & args,\n"
    @"        device const char * src0,\n"
    @"        device const char * src1,\n"
    @"        device char * dst,\n"
    @"        uint3 tgpig [[threadgroup_position_in_grid]],\n"
    @"        ushort tiitg [[thread_index_in_threadgroup]],\n"
    @"        ushort3 ntg [[threads_per_threadgroup]]) {\n"
    @"    const int iw0 = tgpig.x/args.ne10;\n"
    @"    const int i10 = tgpig.x%args.ne10;\n"
    @"    const int i11 = tgpig.y;\n"
    @"    const int i12 = tgpig.z;\n"
    @"    const int r = ((const device int *)(src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];\n"
    @"    const int i02 = i11;\n"
    @"    const int i03 = i12;\n"
    @"    auto psrc = (const device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + r*args.nb01);\n"
    @"    auto pdst = (device T *)(dst + i12*args.nb3 + i11*args.nb2 + i10*args.nb1);\n"
    @"    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {\n"
    @"        pdst[ind] = psrc[ind];\n"
    @"        break;\n"
    @"    }\n"
    @"}\n"
    @"typedef decltype(kernel_get_rows_f<float, float>) get_rows_f_t;\n"
    @"template [[host_name(\"kernel_get_rows_f16\")]] kernel get_rows_f_t kernel_get_rows_f<half, float>;\n";

#include "attention_ingress_source.inc"

// Imported from DwarfStar metal/dense.metal. The fixed target uses DwarfStar's
// default four simdgroups and two output rows per threadgroup.
static NSString *const kQ8ProjectionSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"#define QK8_0 32\n"
    @"#define N_SIMDWIDTH 32\n"
    @"#define N_R0_Q8_0 2\n"
    @"#define FC_MUL_MV 600\n"
    @"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
    @"constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];\n"
    @"struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n"
    @"struct ds4_metal_args_mul_mv {\n"
    @"    int ne00; int ne01; int ne02;\n"
    @"    ulong nb00; ulong nb01; ulong nb02; ulong nb03;\n"
    @"    int ne10; int ne11; int ne12;\n"
    @"    ulong nb10; ulong nb11; ulong nb12; ulong nb13;\n"
    @"    int ne0; int ne1; int nr0; short r2; short r3;\n"
    @"};\n"
    @"template<short NR0>\n"
    @"static inline void helper_mv_reduce_and_write(\n"
    @"        device float * dst_f32, float sumf[NR0], const int r0,\n"
    @"        const int ne01, ushort tiisg, ushort sgitg,\n"
    @"        threadgroup char * shmem) {\n"
    @"    constexpr short NW = N_SIMDWIDTH;\n"
    @"    threadgroup float * shmem_f32[NR0];\n"
    @"    for (short row = 0; row < NR0; ++row) {\n"
    @"        shmem_f32[row] = (threadgroup float *) shmem + NW*row;\n"
    @"        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;\n"
    @"        sumf[row] = simd_sum(sumf[row]);\n"
    @"    }\n"
    @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    @"    for (short row = 0; row < NR0; ++row) {\n"
    @"        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];\n"
    @"    }\n"
    @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    @"    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {\n"
    @"        float tot = simd_sum(shmem_f32[row][tiisg]);\n"
    @"        if (tiisg == 0 && sgitg == 0) dst_f32[r0 + row] = tot;\n"
    @"    }\n"
    @"}\n"
    @"template<short NR0>\n"
    @"static inline void kernel_mul_mv_q8_0_f32_impl(\n"
    @"        constant ds4_metal_args_mul_mv & args,\n"
    @"        device const char * src0, device const char * src1,\n"
    @"        device char * dst, threadgroup char * shmem, uint3 tgpig,\n"
    @"        ushort tiisg, ushort sgitg) {\n"
    @"    const short NSG = FC_mul_mv_nsg;\n"
    @"    constexpr short NW = N_SIMDWIDTH; constexpr short NQ = 8;\n"
    @"    const int nb = args.ne00/QK8_0;\n"
    @"    const int r0 = tgpig.x*NR0; const int r1 = tgpig.y; const int im = tgpig.z;\n"
    @"    const uint i12 = im%args.ne12; const uint i13 = im/args.ne12;\n"
    @"    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;\n"
    @"    device const float * y = (device const float *) (src1 + offset1);\n"
    @"    device const block_q8_0 * ax[NR0];\n"
    @"    FOR_UNROLL (short row = 0; row < NR0; ++row) {\n"
    @"        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;\n"
    @"        ax[row] = (device const block_q8_0 *) (src0 + offset0);\n"
    @"    }\n"
    @"    float sumf[NR0] = { 0.f };\n"
    @"    const short ix = tiisg/(NW/NQ); const short il = tiisg%(NW/NQ);\n"
    @"    const int ib0 = sgitg*NQ + ix; float yl[NQ];\n"
    @"    device const float * yb = y + ib0*QK8_0 + il*NQ;\n"
    @"    for (int ib = ib0; ib < nb; ib += NSG*NQ) {\n"
    @"        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];\n"
    @"        for (short row = 0; row < NR0; row++) {\n"
    @"            device const int8_t * qs = ax[row][ib].qs + il*NQ;\n"
    @"            float sumq = 0.f;\n"
    @"            FOR_UNROLL (short i = 0; i < NQ; ++i) sumq += qs[i] * yl[i];\n"
    @"            sumf[row] += sumq*ax[row][ib].d;\n"
    @"        }\n"
    @"        yb += NSG*NQ*QK8_0;\n"
    @"    }\n"
    @"    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;\n"
    @"    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);\n"
    @"}\n"
    @"[[host_name(\"kernel_mul_mv_q8_0_f32\")]]\n"
    @"kernel void kernel_mul_mv_q8_0_f32(\n"
    @"        constant ds4_metal_args_mul_mv & args, device const char * src0,\n"
    @"        device const char * src1, device char * dst,\n"
    @"        threadgroup char * shmem [[threadgroup(0)]],\n"
    @"        uint3 tgpig [[threadgroup_position_in_grid]],\n"
    @"        ushort tiisg [[thread_index_in_simdgroup]],\n"
    @"        ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
    @"    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);\n"
    @"}\n";

@interface RustStarMetalContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> probePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> getRowsF16Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q8ProjectionPipeline;
@property(nonatomic, strong) id<MTLLibrary> attentionIngressLibrary;
@property(nonatomic, strong) id<MTLComputePipelineState> repeatF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsNormF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> f16ProjectionPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> hcIngressPipeline;
@property(nonatomic, assign) double setupMilliseconds;
@property(nonatomic, assign) double compileMilliseconds;
@end

@implementation RustStarMetalContext
@end

static double monotonic_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static int fail_with_message(char *error, size_t error_bytes, NSString *message) {
    if (error && error_bytes > 0) {
        const char *text = message ? message.UTF8String : "unknown Metal failure";
        snprintf(error, error_bytes, "%s", text ? text : "unknown Metal failure");
    }
    return 0;
}

static int command_succeeded(
    id<MTLCommandBuffer> command,
    char *error,
    size_t error_bytes)
{
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        return fail_with_message(error, error_bytes, command.error.localizedDescription);
    }
    return 1;
}

static int encode_probe(
    RustStarMetalContext *context,
    id<MTLCommandBuffer> command,
    id<MTLBuffer> buffer,
    NSUInteger elements,
    char *error,
    size_t error_bytes)
{
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) {
        return fail_with_message(error, error_bytes, @"failed to create compute encoder");
    }
    [encoder setComputePipelineState:context.probePipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    const NSUInteger width = MIN((NSUInteger)256, context.probePipeline.maxTotalThreadsPerThreadgroup);
    if (width == 0) {
        [encoder endEncoding];
        return fail_with_message(error, error_bytes, @"Metal pipeline reported zero threadgroup width");
    }
    [encoder dispatchThreads:MTLSizeMake(elements, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    return 1;
}

static double gpu_elapsed_ms(id<MTLCommandBuffer> command) {
    const double seconds = command.GPUEndTime - command.GPUStartTime;
    return seconds > 0.0 ? seconds * 1000.0 : 0.0;
}

int rust_star_metal_create(void **context_out, char *error, size_t error_bytes) {
    if (!context_out) return fail_with_message(error, error_bytes, @"context output is null");
    *context_out = NULL;
    @autoreleasepool {
        const double setup_start = monotonic_ms();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return fail_with_message(error, error_bytes, @"Metal device is unavailable");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return fail_with_message(error, error_bytes, @"failed to create Metal command queue");
        const double setup_end = monotonic_ms();

        NSError *compile_error = nil;
        const double compile_start = monotonic_ms();
        MTLCompileOptions *options = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:kProbeSource
                                                     options:options
                                                       error:&compile_error];
        if (!library) {
            return fail_with_message(error, error_bytes, compile_error.localizedDescription);
        }
        id<MTLFunction> function = [library newFunctionWithName:@"rust_star_dispatch_probe"];
        if (!function) {
            return fail_with_message(error, error_bytes, @"probe kernel was not found in Metal library");
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&compile_error];
        if (!pipeline) {
            return fail_with_message(error, error_bytes, compile_error.localizedDescription);
        }
        const double compile_end = monotonic_ms();

        RustStarMetalContext *context = [RustStarMetalContext new];
        context.device = device;
        context.queue = queue;
        context.probePipeline = pipeline;
        context.setupMilliseconds = setup_end - setup_start;
        context.compileMilliseconds = compile_end - compile_start;
        *context_out = (__bridge_retained void *)context;
        return 1;
    }
}

static int ensure_get_rows_f16_pipeline(
    RustStarMetalContext *context,
    char *error,
    size_t error_bytes)
{
    if (context.getRowsF16Pipeline) return 1;
    NSError *compile_error = nil;
    id<MTLLibrary> library = [context.device newLibraryWithSource:kGetRowsF16Source
                                                          options:[MTLCompileOptions new]
                                                            error:&compile_error];
    if (!library) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_get_rows_f16"];
    if (!function) {
        return fail_with_message(error, error_bytes, @"kernel_get_rows_f16 was not found in Metal library");
    }
    context.getRowsF16Pipeline =
        [context.device newComputePipelineStateWithFunction:function error:&compile_error];
    if (!context.getRowsF16Pipeline) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    return 1;
}

static int ensure_q8_projection_pipeline(
    RustStarMetalContext *context,
    char *error,
    size_t error_bytes)
{
    if (context.q8ProjectionPipeline) return 1;
    NSError *compile_error = nil;
    id<MTLLibrary> library = [context.device newLibraryWithSource:kQ8ProjectionSource
                                                          options:[MTLCompileOptions new]
                                                            error:&compile_error];
    if (!library) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    int16_t simdgroups = 4;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&simdgroups type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_mul_mv_q8_0_f32"
                                             constantValues:constants
                                                      error:&compile_error];
    if (!function) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    context.q8ProjectionPipeline =
        [context.device newComputePipelineStateWithFunction:function error:&compile_error];
    if (!context.q8ProjectionPipeline) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    return 1;
}

static int ensure_attention_ingress_pipelines(
    RustStarMetalContext *context,
    char *error,
    size_t error_bytes)
{
    if (context.attentionIngressLibrary) return 1;
    NSError *compile_error = nil;
    id<MTLLibrary> library =
        [context.device newLibraryWithSource:kAttentionIngressSource
                                      options:[MTLCompileOptions new]
                                        error:&compile_error];
    if (!library) return fail_with_message(error, error_bytes, compile_error.localizedDescription);

    id<MTLFunction> repeat = [library newFunctionWithName:@"kernel_repeat_f32"];
    id<MTLFunction> norm = [library newFunctionWithName:@"kernel_rms_norm_f32_4"];
    id<MTLFunction> hc = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum_norm4"];
    int16_t simdgroups = 8;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&simdgroups type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> projection = [library newFunctionWithName:@"kernel_mul_mv_f16_f32_4"
                                               constantValues:constants
                                                        error:&compile_error];
    if (!repeat || !norm || !hc || !projection) {
        return fail_with_message(error, error_bytes,
            compile_error ? compile_error.localizedDescription : @"attention ingress kernel was not found");
    }
    context.repeatF32Pipeline = [context.device newComputePipelineStateWithFunction:repeat error:&compile_error];
    context.rmsNormF32Pipeline = [context.device newComputePipelineStateWithFunction:norm error:&compile_error];
    context.f16ProjectionPipeline = [context.device newComputePipelineStateWithFunction:projection error:&compile_error];
    context.hcIngressPipeline = [context.device newComputePipelineStateWithFunction:hc error:&compile_error];
    if (!context.repeatF32Pipeline || !context.rmsNormF32Pipeline ||
        !context.f16ProjectionPipeline || !context.hcIngressPipeline) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    context.attentionIngressLibrary = library;
    return 1;
}

int rust_star_metal_run_probe(
    void *opaque_context,
    uint64_t elements,
    uint64_t iterations,
    rust_star_metal_probe_result *result,
    char *error,
    size_t error_bytes)
{
    if (!opaque_context || !result) {
        return fail_with_message(error, error_bytes, @"probe context or result is null");
    }
    if (elements == 0 || elements > kMaxElements ||
        iterations == 0 || iterations > kMaxIterations ||
        elements > kMaxThreadInvocations / iterations) {
        return fail_with_message(error, error_bytes, @"probe dimensions are invalid");
    }
    if (elements > (uint64_t)SIZE_MAX / sizeof(uint32_t)) {
        return fail_with_message(error, error_bytes, @"probe buffer size overflows address space");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        memset(result, 0, sizeof(*result));
        const NSUInteger count = (NSUInteger)elements;
        const NSUInteger bytes = count * sizeof(uint32_t);
        id<MTLBuffer> buffer = [context.device newBufferWithLength:bytes
                                                           options:MTLResourceStorageModeShared];
        if (!buffer) return fail_with_message(error, error_bytes, @"failed to allocate shared Metal probe buffer");
        uint32_t *values = (uint32_t *)buffer.contents;
        if (!values) return fail_with_message(error, error_bytes, @"Metal probe buffer has no host mapping");
        for (NSUInteger index = 0; index < count; index++) {
            values[index] = (uint32_t)index ^ 0xa5a5a5a5u;
        }

        id<MTLCommandBuffer> warmup = [context.queue commandBuffer];
        if (!warmup || !encode_probe(context, warmup, buffer, count, error, error_bytes)) return 0;
        const double warmup_start = monotonic_ms();
        [warmup commit];
        if (!command_succeeded(warmup, error, error_bytes)) return 0;
        const double warmup_end = monotonic_ms();
        const double warmup_gpu_ms = gpu_elapsed_ms(warmup);

        for (NSUInteger index = 0; index < count; index++) {
            values[index] = (uint32_t)index ^ 0xa5a5a5a5u;
        }

        double roundtrip_gpu_ms = 0.0;
        const double roundtrip_start = monotonic_ms();
        for (uint64_t iteration = 0; iteration < iterations; iteration++) {
            @autoreleasepool {
                id<MTLCommandBuffer> command = [context.queue commandBuffer];
                if (!command || !encode_probe(context, command, buffer, count, error, error_bytes)) return 0;
                [command commit];
                if (!command_succeeded(command, error, error_bytes)) return 0;
                roundtrip_gpu_ms += gpu_elapsed_ms(command);
            }
        }
        const double roundtrip_end = monotonic_ms();

        const double batched_start = monotonic_ms();
        id<MTLCommandBuffer> batched = [context.queue commandBuffer];
        if (!batched) return fail_with_message(error, error_bytes, @"failed to create batched command buffer");
        id<MTLComputeCommandEncoder> encoder = [batched computeCommandEncoder];
        if (!encoder) return fail_with_message(error, error_bytes, @"failed to create batched compute encoder");
        [encoder setComputePipelineState:context.probePipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        const NSUInteger width = MIN((NSUInteger)256, context.probePipeline.maxTotalThreadsPerThreadgroup);
        if (width == 0) {
            [encoder endEncoding];
            return fail_with_message(error, error_bytes, @"Metal pipeline reported zero threadgroup width");
        }
        for (uint64_t iteration = 0; iteration < iterations; iteration++) {
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        }
        [encoder endEncoding];
        [batched commit];
        if (!command_succeeded(batched, error, error_bytes)) return 0;
        const double batched_end = monotonic_ms();

        uint64_t checksum = 0;
        for (NSUInteger index = 0; index < count; index++) {
            uint32_t expected = (uint32_t)index ^ 0xa5a5a5a5u;
            for (uint64_t iteration = 0; iteration < iterations * 2; iteration++) {
                expected = expected * 1664525u + 1013904223u;
            }
            if (values[index] != expected) {
                return fail_with_message(error, error_bytes, @"Metal probe produced an incorrect value");
            }
            checksum += values[index];
        }

        result->elements = elements;
        result->iterations = iterations;
        result->recommended_max_working_set_bytes = context.device.recommendedMaxWorkingSetSize;
        result->buffer_bytes = bytes;
        result->max_total_threads_per_threadgroup = context.probePipeline.maxTotalThreadsPerThreadgroup;
        result->checksum = checksum;
        if (@available(macOS 10.15, *)) {
            result->has_unified_memory = context.device.hasUnifiedMemory ? 1u : 0u;
        }
        result->setup_ms = context.setupMilliseconds;
        result->compile_ms = context.compileMilliseconds;
        result->warmup_wall_ms = warmup_end - warmup_start;
        result->warmup_gpu_ms = warmup_gpu_ms;
        result->roundtrip_wall_ms = roundtrip_end - roundtrip_start;
        result->roundtrip_gpu_ms = roundtrip_gpu_ms;
        result->batched_wall_ms = batched_end - batched_start;
        result->batched_gpu_ms = gpu_elapsed_ms(batched);
        const char *device_name = context.device.name.UTF8String;
        snprintf(
            result->device_name,
            sizeof(result->device_name),
            "%s",
            device_name ? device_name : "unknown");
        return 1;
    }
}

typedef struct rust_star_get_rows_args {
    int32_t ne00t;
    int32_t ne00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} rust_star_get_rows_args;

static uint64_t round_up_u64(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

int rust_star_metal_run_f16_get_rows(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    uint32_t n_vocab,
    uint32_t n_embd,
    const uint32_t *tokens,
    uint32_t token_count,
    float *output,
    uint64_t output_elements,
    rust_star_metal_embedding_probe_result *result,
    char *error,
    size_t error_bytes)
{
    if (!opaque_context || !model_mapping || !tokens || !output || !result) {
        return fail_with_message(error, error_bytes, @"embedding probe received a null input");
    }
    if (n_vocab == 0 || n_embd == 0 || token_count == 0 || token_count > 64) {
        return fail_with_message(error, error_bytes, @"embedding probe dimensions are invalid");
    }
    const uint64_t expected_elements = (uint64_t)n_embd * token_count;
    if (output_elements != expected_elements || output_elements > SIZE_MAX / sizeof(float)) {
        return fail_with_message(error, error_bytes, @"embedding probe output size is invalid");
    }
    if ((uint64_t)n_embd > UINT64_MAX / (uint64_t)n_vocab / sizeof(uint16_t) ||
        tensor_bytes != (uint64_t)n_embd * n_vocab * sizeof(uint16_t) ||
        tensor_offset > model_bytes || tensor_bytes > model_bytes - tensor_offset) {
        return fail_with_message(error, error_bytes, @"embedding tensor range is invalid");
    }
    for (uint32_t index = 0; index < token_count; index++) {
        if (tokens[index] >= n_vocab) {
            return fail_with_message(error, error_bytes, @"embedding token is outside vocabulary");
        }
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_get_rows_f16_pipeline(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));
        const uint64_t page = (uint64_t)getpagesize();
        const uint64_t page_offset = tensor_offset & ~(page - 1u);
        const uint64_t leading = tensor_offset - page_offset;
        if (tensor_bytes > UINT64_MAX - leading || leading + tensor_bytes > UINT64_MAX - (page - 1u)) {
            return fail_with_message(error, error_bytes, @"embedding tensor page alignment overflows");
        }
        const uint64_t required = leading + tensor_bytes;
        uint64_t buffer_bytes = round_up_u64(required, page);
        if (buffer_bytes > model_bytes - page_offset) buffer_bytes = model_bytes - page_offset;
        if (required > buffer_bytes || buffer_bytes > (uint64_t)context.device.maxBufferLength) {
            return fail_with_message(error, error_bytes, @"embedding tensor exceeds the Metal buffer range");
        }

        const uintptr_t base = (uintptr_t)model_mapping;
        void *page_pointer = (void *)(base + page_offset);
        id<MTLBuffer> weights =
            [context.device newBufferWithBytesNoCopy:page_pointer
                                              length:(NSUInteger)buffer_bytes
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
        if (!weights) {
            return fail_with_message(error, error_bytes, @"failed to wrap mmaped embedding tensor without copying");
        }
        const BOOL pointer_match = weights.contents == page_pointer;
        if (!pointer_match) {
            return fail_with_message(error, error_bytes, @"Metal buffer contents do not match the mmap pointer");
        }
        const NSUInteger token_bytes = (NSUInteger)token_count * sizeof(uint32_t);
        id<MTLBuffer> token_buffer = [context.device newBufferWithBytes:tokens
                                                                length:token_bytes
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer =
            [context.device newBufferWithLength:(NSUInteger)output_elements * sizeof(float)
                                         options:MTLResourceStorageModeShared];
        if (!token_buffer || !output_buffer) {
            return fail_with_message(error, error_bytes, @"failed to allocate embedding probe buffers");
        }

        const uint64_t src_row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
        const uint64_t dst_row_bytes = (uint64_t)n_embd * sizeof(float);
        rust_star_get_rows_args args = {
            .ne00t = (int32_t)n_embd,
            .ne00 = (int32_t)n_embd,
            .nb01 = src_row_bytes,
            .nb02 = (uint64_t)n_vocab * src_row_bytes,
            .nb03 = (uint64_t)n_vocab * src_row_bytes,
            .ne10 = (int32_t)token_count,
            .nb10 = sizeof(uint32_t),
            .nb11 = token_bytes,
            .nb12 = token_bytes,
            .nb1 = dst_row_bytes,
            .nb2 = (uint64_t)token_count * dst_row_bytes,
            .nb3 = (uint64_t)token_count * dst_row_bytes,
        };

        NSUInteger threads = (NSUInteger)n_embd;
        const NSUInteger max_threads = context.getRowsF16Pipeline.maxTotalThreadsPerThreadgroup;
        if (threads > max_threads) threads = max_threads;
        if (threads == 0) threads = 1;
        const NSUInteger width_groups = ((NSUInteger)n_embd + threads - 1u) / threads;
        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            return fail_with_message(error, error_bytes, @"failed to create embedding command encoder");
        }
        [encoder setComputePipelineState:context.getRowsF16Pipeline];
        [encoder setBytes:&args length:sizeof(args) atIndex:0];
        [encoder setBuffer:weights offset:(NSUInteger)leading atIndex:1];
        [encoder setBuffer:token_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(width_groups * token_count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        memcpy(output, output_buffer.contents, (size_t)output_elements * sizeof(float));

        result->model_bytes = model_bytes;
        result->tensor_offset = tensor_offset;
        result->tensor_bytes = tensor_bytes;
        result->page_offset = page_offset;
        result->buffer_bytes = buffer_bytes;
        result->inner_offset = leading;
        result->output_elements = output_elements;
        result->max_buffer_length = context.device.maxBufferLength;
        result->no_copy_pointer_match = pointer_match ? 1u : 0u;
        result->wall_ms = wall_end - wall_start;
        result->gpu_ms = gpu_elapsed_ms(command);
        return 1;
    }
}

typedef struct rust_star_q8_mv_args {
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
    int32_t nr0;
    int16_t r2;
    int16_t r3;
} rust_star_q8_mv_args;

typedef struct rust_star_repeat_args {
    int32_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne0, ne1, ne2, ne3;
    uint64_t nb0, nb1, nb2, nb3;
} rust_star_repeat_args;

typedef struct rust_star_norm_args {
    int32_t ne00, ne00_t;
    uint64_t nb1, nb2, nb3;
    float eps;
    int32_t nef1[3], nef2[3], nef3[3];
    uint64_t nbf1[3], nbf2[3], nbf3[3];
} rust_star_norm_args;

typedef struct rust_star_hc_ingress_args {
    int64_t n_embd;
    int32_t n_hc, sinkhorn_iters;
    int64_t n_rows, mix_hc;
    uint64_t nb_mix1, nb_split1;
    uint64_t nb_x0, nb_x1, nb_x2;
    uint64_t nb0, nb1, nb_norm1;
    float eps, norm_eps;
} rust_star_hc_ingress_args;

static id<MTLBuffer> wrap_model_range(
    RustStarMetalContext *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t offset,
    uint64_t bytes,
    NSUInteger *inner,
    BOOL *pointer_match,
    char *error,
    size_t error_bytes)
{
    if (offset > model_bytes || bytes > model_bytes - offset) {
        fail_with_message(error, error_bytes, @"attention ingress model range is invalid");
        return nil;
    }
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t page_offset = offset & ~(page - 1u);
    const uint64_t leading = offset - page_offset;
    if (bytes > UINT64_MAX - leading || leading + bytes > UINT64_MAX - (page - 1u)) {
        fail_with_message(error, error_bytes, @"attention ingress model range alignment overflows");
        return nil;
    }
    const uint64_t required = leading + bytes;
    uint64_t buffer_bytes = round_up_u64(required, page);
    if (buffer_bytes > model_bytes - page_offset) buffer_bytes = model_bytes - page_offset;
    if (required > buffer_bytes || buffer_bytes > (uint64_t)context.device.maxBufferLength) {
        fail_with_message(error, error_bytes, @"attention ingress model range exceeds Metal limits");
        return nil;
    }
    void *page_pointer = (void *)((uintptr_t)model_mapping + page_offset);
    id<MTLBuffer> buffer = [context.device newBufferWithBytesNoCopy:page_pointer
                                                           length:(NSUInteger)buffer_bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
    if (!buffer) {
        fail_with_message(error, error_bytes, @"failed to wrap attention ingress model range");
        return nil;
    }
    *inner = (NSUInteger)leading;
    *pointer_match = buffer.contents == page_pointer;
    if (!*pointer_match) {
        fail_with_message(error, error_bytes, @"attention ingress Metal buffer does not preserve mmap identity");
        return nil;
    }
    return buffer;
}

int rust_star_metal_run_q8_0_projection(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    uint32_t input_elements,
    uint32_t output_elements,
    const float *input,
    float *output,
    rust_star_metal_projection_probe_result *result,
    char *error,
    size_t error_bytes)
{
    if (!opaque_context || !model_mapping || !input || !output || !result) {
        return fail_with_message(error, error_bytes, @"projection probe received a null input");
    }
    if (input_elements == 0 || (input_elements & 31u) != 0 ||
        output_elements == 0 || (output_elements & 1u) != 0) {
        return fail_with_message(error, error_bytes, @"projection probe dimensions are invalid");
    }
    const uint64_t row_bytes = (uint64_t)(input_elements / 32u) * 34u;
    if ((uint64_t)output_elements > UINT64_MAX / row_bytes ||
        tensor_bytes != (uint64_t)output_elements * row_bytes ||
        tensor_offset > model_bytes || tensor_bytes > model_bytes - tensor_offset) {
        return fail_with_message(error, error_bytes, @"projection tensor range is invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_q8_projection_pipeline(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));

        const uint64_t page = (uint64_t)getpagesize();
        const uint64_t page_offset = tensor_offset & ~(page - 1u);
        const uint64_t leading = tensor_offset - page_offset;
        if (tensor_bytes > UINT64_MAX - leading ||
            leading + tensor_bytes > UINT64_MAX - (page - 1u)) {
            return fail_with_message(error, error_bytes, @"projection tensor page alignment overflows");
        }
        const uint64_t required = leading + tensor_bytes;
        uint64_t buffer_bytes = round_up_u64(required, page);
        if (buffer_bytes > model_bytes - page_offset) buffer_bytes = model_bytes - page_offset;
        if (required > buffer_bytes || buffer_bytes > (uint64_t)context.device.maxBufferLength) {
            return fail_with_message(error, error_bytes, @"projection tensor exceeds the Metal buffer range");
        }

        const uintptr_t base = (uintptr_t)model_mapping;
        void *page_pointer = (void *)(base + page_offset);
        id<MTLBuffer> weights =
            [context.device newBufferWithBytesNoCopy:page_pointer
                                              length:(NSUInteger)buffer_bytes
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
        if (!weights) {
            return fail_with_message(error, error_bytes, @"failed to wrap mmaped projection tensor without copying");
        }
        const BOOL pointer_match = weights.contents == page_pointer;
        if (!pointer_match) {
            return fail_with_message(error, error_bytes, @"projection buffer contents do not match the mmap pointer");
        }

        const NSUInteger input_bytes = (NSUInteger)input_elements * sizeof(float);
        const NSUInteger output_bytes = (NSUInteger)output_elements * sizeof(float);
        id<MTLBuffer> input_buffer = [context.device newBufferWithBytes:input
                                                                 length:input_bytes
                                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [context.device newBufferWithLength:output_bytes
                                                                  options:MTLResourceStorageModeShared];
        if (!input_buffer || !output_buffer) {
            return fail_with_message(error, error_bytes, @"failed to allocate projection activation buffers");
        }

        rust_star_q8_mv_args args = {
            .ne00 = (int32_t)input_elements,
            .ne01 = (int32_t)output_elements,
            .ne02 = 1,
            .nb00 = 34,
            .nb01 = row_bytes,
            .nb02 = row_bytes * output_elements,
            .nb03 = row_bytes * output_elements,
            .ne10 = (int32_t)input_elements,
            .ne11 = 1,
            .ne12 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)input_elements * sizeof(float),
            .nb12 = (uint64_t)input_elements * sizeof(float),
            .nb13 = (uint64_t)input_elements * sizeof(float),
            .ne0 = (int32_t)output_elements,
            .ne1 = 1,
            .nr0 = 2,
            .r2 = 1,
            .r3 = 1,
        };

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            return fail_with_message(error, error_bytes, @"failed to create projection command encoder");
        }
        [encoder setComputePipelineState:context.q8ProjectionPipeline];
        [encoder setBytes:&args length:sizeof(args) atIndex:0];
        [encoder setBuffer:weights offset:(NSUInteger)leading atIndex:1];
        [encoder setBuffer:input_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(((NSUInteger)output_elements + 1u) / 2u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        [encoder endEncoding];

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        memcpy(output, output_buffer.contents, output_bytes);

        result->model_bytes = model_bytes;
        result->tensor_offset = tensor_offset;
        result->tensor_bytes = tensor_bytes;
        result->page_offset = page_offset;
        result->buffer_bytes = buffer_bytes;
        result->inner_offset = leading;
        result->input_elements = input_elements;
        result->output_elements = output_elements;
        result->max_buffer_length = context.device.maxBufferLength;
        result->no_copy_pointer_match = pointer_match ? 1u : 0u;
        result->simdgroups = 4;
        result->rows_per_threadgroup = 2;
        result->wall_ms = wall_end - wall_start;
        result->gpu_ms = gpu_elapsed_ms(command);
        return 1;
    }
}

int rust_star_metal_run_attention_ingress(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint32_t token,
    uint32_t n_vocab,
    uint64_t embedding_offset,
    uint64_t embedding_bytes,
    uint64_t hc_fn_offset,
    uint64_t hc_fn_bytes,
    uint64_t hc_scale_offset,
    uint64_t hc_scale_bytes,
    uint64_t hc_base_offset,
    uint64_t hc_base_bytes,
    uint64_t attn_norm_offset,
    uint64_t attn_norm_bytes,
    uint64_t q_a_offset,
    uint64_t q_a_bytes,
    float *mixes,
    float *split,
    float *collapsed,
    float *attn_norm,
    float *q_lora,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes)
{
    const uint32_t n_embd = 4096;
    const uint32_t n_hc = 4;
    const uint32_t hc_dim = n_embd * n_hc;
    const uint32_t mix_hc = 24;
    const uint32_t q_elements = 1024;
    if (!opaque_context || !model_mapping || !mixes || !split || !collapsed ||
        !attn_norm || !q_lora || !result || n_vocab == 0 || token >= n_vocab) {
        return fail_with_message(error, error_bytes, @"attention ingress received invalid inputs");
    }
    if (embedding_bytes != (uint64_t)n_embd*n_vocab*sizeof(uint16_t) ||
        hc_fn_bytes != (uint64_t)hc_dim*mix_hc*sizeof(uint16_t) ||
        hc_scale_bytes != 3u*sizeof(float) || hc_base_bytes != mix_hc*sizeof(float) ||
        attn_norm_bytes != n_embd*sizeof(float) ||
        q_a_bytes != (uint64_t)(n_embd/32u)*34u*q_elements) {
        return fail_with_message(error, error_bytes, @"attention ingress tensor shapes are invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_get_rows_f16_pipeline(context, error, error_bytes) ||
            !ensure_attention_ingress_pipelines(context, error, error_bytes) ||
            !ensure_q8_projection_pipeline(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));

        NSUInteger embedding_inner = 0, hc_fn_inner = 0, scale_inner = 0;
        NSUInteger base_inner = 0, norm_inner = 0, q_inner = 0;
        BOOL matches[6] = {NO, NO, NO, NO, NO, NO};
        id<MTLBuffer> embedding_weights = wrap_model_range(context, model_mapping, model_bytes,
            embedding_offset, embedding_bytes, &embedding_inner, &matches[0], error, error_bytes);
        id<MTLBuffer> hc_fn_weights = wrap_model_range(context, model_mapping, model_bytes,
            hc_fn_offset, hc_fn_bytes, &hc_fn_inner, &matches[1], error, error_bytes);
        id<MTLBuffer> scale_weights = wrap_model_range(context, model_mapping, model_bytes,
            hc_scale_offset, hc_scale_bytes, &scale_inner, &matches[2], error, error_bytes);
        id<MTLBuffer> base_weights = wrap_model_range(context, model_mapping, model_bytes,
            hc_base_offset, hc_base_bytes, &base_inner, &matches[3], error, error_bytes);
        id<MTLBuffer> norm_weights = wrap_model_range(context, model_mapping, model_bytes,
            attn_norm_offset, attn_norm_bytes, &norm_inner, &matches[4], error, error_bytes);
        id<MTLBuffer> q_weights = wrap_model_range(context, model_mapping, model_bytes,
            q_a_offset, q_a_bytes, &q_inner, &matches[5], error, error_bytes);
        if (!embedding_weights || !hc_fn_weights || !scale_weights || !base_weights ||
            !norm_weights || !q_weights) return 0;

        id<MTLBuffer> embedding = [context.device newBufferWithLength:n_embd*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cur_hc = [context.device newBufferWithLength:hc_dim*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> flat_hc = [context.device newBufferWithLength:hc_dim*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> mix_buffer = [context.device newBufferWithLength:mix_hc*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> split_buffer = [context.device newBufferWithLength:mix_hc*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> collapsed_buffer = [context.device newBufferWithLength:n_embd*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> norm_buffer = [context.device newBufferWithLength:n_embd*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_buffer = [context.device newBufferWithLength:q_elements*sizeof(float) options:MTLResourceStorageModeShared];
        if (!embedding || !cur_hc || !flat_hc || !mix_buffer || !split_buffer ||
            !collapsed_buffer || !norm_buffer || !q_buffer) {
            return fail_with_message(error, error_bytes, @"failed to allocate attention ingress buffers");
        }

        const uint64_t embedding_row_bytes = (uint64_t)n_embd*sizeof(uint16_t);
        rust_star_get_rows_args get_rows = {
            .ne00t = (int32_t)n_embd, .ne00 = (int32_t)n_embd,
            .nb01 = embedding_row_bytes, .nb02 = embedding_bytes, .nb03 = embedding_bytes,
            .ne10 = 1, .nb10 = sizeof(uint32_t), .nb11 = sizeof(uint32_t), .nb12 = sizeof(uint32_t),
            .nb1 = n_embd*sizeof(float), .nb2 = n_embd*sizeof(float), .nb3 = n_embd*sizeof(float),
        };
        const uint64_t f32_row_bytes = (uint64_t)n_embd*sizeof(float);
        rust_star_repeat_args repeat = {
            .ne00=(int32_t)n_embd, .ne01=1, .ne02=1, .ne03=1,
            .nb00=sizeof(float), .nb01=f32_row_bytes, .nb02=f32_row_bytes, .nb03=f32_row_bytes,
            .ne0=(int32_t)n_embd, .ne1=(int32_t)n_hc, .ne2=1, .ne3=1,
            .nb0=sizeof(float), .nb1=f32_row_bytes, .nb2=(uint64_t)n_hc*f32_row_bytes, .nb3=(uint64_t)n_hc*f32_row_bytes,
        };
        const uint64_t hc_row_bytes = (uint64_t)hc_dim*sizeof(float);
        rust_star_norm_args norm = {
            .ne00=(int32_t)hc_dim, .ne00_t=(int32_t)(hc_dim/4u),
            .nb1=hc_row_bytes, .nb2=hc_row_bytes, .nb3=hc_row_bytes, .eps=1.0e-6f,
            .nef1={(int32_t)1,1,1}, .nef2={1,1,1}, .nef3={1,1,1},
            .nbf1={hc_row_bytes,hc_row_bytes,hc_row_bytes},
            .nbf2={hc_row_bytes,hc_row_bytes,hc_row_bytes},
            .nbf3={hc_row_bytes,hc_row_bytes,hc_row_bytes},
        };
        const uint64_t hc_fn_row_bytes = (uint64_t)hc_dim*sizeof(uint16_t);
        rust_star_q8_mv_args f16_mv = {
            .ne00=(int32_t)hc_dim, .ne01=(int32_t)mix_hc, .ne02=1,
            .nb00=sizeof(uint16_t), .nb01=hc_fn_row_bytes,
            .nb02=hc_fn_bytes, .nb03=hc_fn_bytes,
            .ne10=(int32_t)hc_dim, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=hc_row_bytes, .nb12=hc_row_bytes, .nb13=hc_row_bytes,
            .ne0=(int32_t)mix_hc, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_hc_ingress_args hc = {
            .n_embd=n_embd, .n_hc=(int32_t)n_hc, .sinkhorn_iters=20,
            .n_rows=1, .mix_hc=mix_hc,
            .nb_mix1=mix_hc*sizeof(float), .nb_split1=mix_hc*sizeof(float),
            .nb_x0=sizeof(float), .nb_x1=f32_row_bytes, .nb_x2=(uint64_t)n_hc*f32_row_bytes,
            .nb0=sizeof(float), .nb1=f32_row_bytes, .nb_norm1=f32_row_bytes,
            .eps=1.0e-6f, .norm_eps=1.0e-6f,
        };
        const uint64_t q_row_bytes = (uint64_t)(n_embd/32u)*34u;
        rust_star_q8_mv_args q_mv = {
            .ne00=(int32_t)n_embd, .ne01=(int32_t)q_elements, .ne02=1,
            .nb00=34, .nb01=q_row_bytes, .nb02=q_a_bytes, .nb03=q_a_bytes,
            .ne10=(int32_t)n_embd, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=f32_row_bytes, .nb12=f32_row_bytes, .nb13=f32_row_bytes,
            .ne0=(int32_t)q_elements, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        if (!command) return fail_with_message(error, error_bytes, @"failed to create attention ingress command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return fail_with_message(error, error_bytes, @"failed to create attention ingress encoder");

        [encoder setComputePipelineState:context.getRowsF16Pipeline];
        [encoder setBytes:&get_rows length:sizeof(get_rows) atIndex:0];
        [encoder setBuffer:embedding_weights offset:embedding_inner atIndex:1];
        int32_t signed_token = (int32_t)token;
        [encoder setBytes:&signed_token length:sizeof(signed_token) atIndex:2];
        [encoder setBuffer:embedding offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.repeatF32Pipeline];
        [encoder setBytes:&repeat length:sizeof(repeat) atIndex:0];
        [encoder setBuffer:embedding offset:0 atIndex:1];
        [encoder setBuffer:cur_hc offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(n_hc,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.rmsNormF32Pipeline];
        [encoder setBytes:&norm length:sizeof(norm) atIndex:0];
        [encoder setBuffer:cur_hc offset:0 atIndex:1];
        [encoder setBuffer:cur_hc offset:0 atIndex:2];
        [encoder setBuffer:cur_hc offset:0 atIndex:3];
        [encoder setBuffer:flat_hc offset:0 atIndex:4];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16ProjectionPipeline];
        [encoder setBytes:&f16_mv length:sizeof(f16_mv) atIndex:0];
        [encoder setBuffer:hc_fn_weights offset:hc_fn_inner atIndex:1];
        [encoder setBuffer:flat_hc offset:0 atIndex:2];
        [encoder setBuffer:mix_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((mix_hc+1u)/2u,1,1) threadsPerThreadgroup:MTLSizeMake(32,8,1)];

        [encoder setComputePipelineState:context.hcIngressPipeline];
        [encoder setBytes:&hc length:sizeof(hc) atIndex:0];
        [encoder setBuffer:mix_buffer offset:0 atIndex:1];
        [encoder setBuffer:scale_weights offset:scale_inner atIndex:2];
        [encoder setBuffer:base_weights offset:base_inner atIndex:3];
        [encoder setBuffer:cur_hc offset:0 atIndex:4];
        [encoder setBuffer:split_buffer offset:0 atIndex:5];
        [encoder setBuffer:collapsed_buffer offset:0 atIndex:6];
        [encoder setBuffer:norm_weights offset:norm_inner atIndex:7];
        [encoder setBuffer:norm_buffer offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:(n_embd+4u+32u)*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.q8ProjectionPipeline];
        [encoder setBytes:&q_mv length:sizeof(q_mv) atIndex:0];
        [encoder setBuffer:q_weights offset:q_inner atIndex:1];
        [encoder setBuffer:norm_buffer offset:0 atIndex:2];
        [encoder setBuffer:q_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(q_elements/2u,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
        [encoder endEncoding];

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        memcpy(mixes, mix_buffer.contents, mix_hc*sizeof(float));
        memcpy(split, split_buffer.contents, mix_hc*sizeof(float));
        memcpy(collapsed, collapsed_buffer.contents, n_embd*sizeof(float));
        memcpy(attn_norm, norm_buffer.contents, n_embd*sizeof(float));
        memcpy(q_lora, q_buffer.contents, q_elements*sizeof(float));
        result->model_bytes = model_bytes;
        result->max_buffer_length = context.device.maxBufferLength;
        result->wrapped_model_ranges = 6;
        for (uint32_t index = 0; index < 6; index++) if (matches[index]) result->pointer_matches++;
        result->wall_ms = wall_end-wall_start;
        result->gpu_ms = gpu_elapsed_ms(command);
        return 1;
    }
}

void rust_star_metal_destroy(void *opaque_context) {
    if (!opaque_context) return;
    @autoreleasepool {
        RustStarMetalContext *context = CFBridgingRelease(opaque_context);
        (void)context;
    }
}
