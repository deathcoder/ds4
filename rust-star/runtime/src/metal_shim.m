#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_shim.h"

#include <stdio.h>
#include <stdint.h>
#include <math.h>
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
#include "attention_output_source.inc"

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
@property(nonatomic, strong) id<MTLComputePipelineState> qkvNormPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> headNormRopePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ropeTailPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> kvStorePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> cpyF32F16Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> flashPadPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> flashVecPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> flashReducePipeline;
@property(nonatomic, strong) id<MTLLibrary> attentionOutputLibrary;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputLowPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputHcPipeline;
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
    id<MTLFunction> qkvNorm = [library newFunctionWithName:@"kernel_dsv4_qkv_rms_norm_f32_4"];
    id<MTLFunction> headNormRope = [library newFunctionWithName:@"kernel_dsv4_head_rms_norm_rope_tail_f32"];
    id<MTLFunction> ropeTail = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32"];
    id<MTLFunction> kvStore = [library newFunctionWithName:@"kernel_dsv4_kv_fp8_store_f32"];
    id<MTLFunction> cpyF32F16 = [library newFunctionWithName:@"kernel_cpy_contig_f32_f16_4"];
    bool enabled = true;
    bool disabled = false;
    int32_t ncpsg = 32;
    MTLFunctionConstantValues *padConstants = [MTLFunctionConstantValues new];
    [padConstants setConstantValue:&enabled type:MTLDataTypeBool atIndex:100];
    [padConstants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:125];
    id<MTLFunction> flashPad = [library newFunctionWithName:@"kernel_flash_attn_ext_pad"
                                            constantValues:padConstants error:&compile_error];
    int32_t headDim = 512, nsg = 1, nwg = 32;
    MTLFunctionConstantValues *vecConstants = [MTLFunctionConstantValues new];
    [vecConstants setConstantValue:&enabled type:MTLDataTypeBool atIndex:400];
    [vecConstants setConstantValue:&enabled type:MTLDataTypeBool atIndex:401];
    [vecConstants setConstantValue:&disabled type:MTLDataTypeBool atIndex:402];
    [vecConstants setConstantValue:&disabled type:MTLDataTypeBool atIndex:403];
    [vecConstants setConstantValue:&enabled type:MTLDataTypeBool atIndex:404];
    [vecConstants setConstantValue:&disabled type:MTLDataTypeBool atIndex:405];
    [vecConstants setConstantValue:&headDim type:MTLDataTypeInt atIndex:420];
    [vecConstants setConstantValue:&headDim type:MTLDataTypeInt atIndex:421];
    [vecConstants setConstantValue:&nsg type:MTLDataTypeInt atIndex:422];
    [vecConstants setConstantValue:&nwg type:MTLDataTypeInt atIndex:423];
    id<MTLFunction> flashVec = [library newFunctionWithName:@"kernel_flash_attn_ext_vec_f16_dk512_dv512"
                                            constantValues:vecConstants error:&compile_error];
    MTLFunctionConstantValues *reduceConstants = [MTLFunctionConstantValues new];
    [reduceConstants setConstantValue:&headDim type:MTLDataTypeInt atIndex:500];
    [reduceConstants setConstantValue:&nwg type:MTLDataTypeInt atIndex:501];
    id<MTLFunction> flashReduce = [library newFunctionWithName:@"kernel_flash_attn_ext_vec_reduce"
                                               constantValues:reduceConstants error:&compile_error];
    int16_t simdgroups = 8;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&simdgroups type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> projection = [library newFunctionWithName:@"kernel_mul_mv_f16_f32_4"
                                               constantValues:constants
                                                        error:&compile_error];
    if (!repeat || !norm || !hc || !qkvNorm || !headNormRope || !ropeTail ||
        !kvStore || !cpyF32F16 || !flashPad || !flashVec || !flashReduce ||
        !projection) {
        return fail_with_message(error, error_bytes,
            compile_error ? compile_error.localizedDescription : @"attention ingress kernel was not found");
    }
    context.repeatF32Pipeline = [context.device newComputePipelineStateWithFunction:repeat error:&compile_error];
    context.rmsNormF32Pipeline = [context.device newComputePipelineStateWithFunction:norm error:&compile_error];
    context.f16ProjectionPipeline = [context.device newComputePipelineStateWithFunction:projection error:&compile_error];
    context.hcIngressPipeline = [context.device newComputePipelineStateWithFunction:hc error:&compile_error];
    context.qkvNormPipeline = [context.device newComputePipelineStateWithFunction:qkvNorm error:&compile_error];
    context.headNormRopePipeline = [context.device newComputePipelineStateWithFunction:headNormRope error:&compile_error];
    context.ropeTailPipeline = [context.device newComputePipelineStateWithFunction:ropeTail error:&compile_error];
    context.kvStorePipeline = [context.device newComputePipelineStateWithFunction:kvStore error:&compile_error];
    context.cpyF32F16Pipeline = [context.device newComputePipelineStateWithFunction:cpyF32F16 error:&compile_error];
    context.flashPadPipeline = [context.device newComputePipelineStateWithFunction:flashPad error:&compile_error];
    context.flashVecPipeline = [context.device newComputePipelineStateWithFunction:flashVec error:&compile_error];
    context.flashReducePipeline = [context.device newComputePipelineStateWithFunction:flashReduce error:&compile_error];
    if (!context.repeatF32Pipeline || !context.rmsNormF32Pipeline ||
        !context.f16ProjectionPipeline || !context.hcIngressPipeline ||
        !context.qkvNormPipeline || !context.headNormRopePipeline ||
        !context.ropeTailPipeline || !context.kvStorePipeline ||
        !context.cpyF32F16Pipeline || !context.flashPadPipeline ||
        !context.flashVecPipeline || !context.flashReducePipeline) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    context.attentionIngressLibrary = library;
    return 1;
}

static int ensure_attention_output_pipelines(
    RustStarMetalContext *context,
    char *error,
    size_t error_bytes)
{
    if (context.attentionOutputLibrary) return 1;
    NSError *compile_error = nil;
    id<MTLLibrary> library =
        [context.device newLibraryWithSource:kAttentionOutputSource
                                      options:[MTLCompileOptions new]
                                        error:&compile_error];
    if (!library) return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    int16_t simdgroups = 4;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&simdgroups type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> low = [library newFunctionWithName:@"kernel_dsv4_attn_out_low_q8_0_f32"
                                        constantValues:constants error:&compile_error];
    id<MTLFunction> hc = [library newFunctionWithName:@"kernel_dsv4_q8_hc_expand4_q8_0"
                                       constantValues:constants error:&compile_error];
    if (!low || !hc) {
        return fail_with_message(error, error_bytes,
            compile_error ? compile_error.localizedDescription : @"attention-output kernel was not found");
    }
    context.attentionOutputLowPipeline =
        [context.device newComputePipelineStateWithFunction:low error:&compile_error];
    context.attentionOutputHcPipeline =
        [context.device newComputePipelineStateWithFunction:hc error:&compile_error];
    if (!context.attentionOutputLowPipeline || !context.attentionOutputHcPipeline) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    context.attentionOutputLibrary = library;
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

typedef struct rust_star_qkv_norm_args {
    int32_t q_n, q_n4, kv_n, kv_n4;
    uint64_t q_row_stride, kv_row_stride;
    float eps;
} rust_star_qkv_norm_args;

typedef struct rust_star_head_norm_rope_args {
    int32_t n_head, head_dim, head_dim4, n_dims;
    int32_t n_ctx_orig, pos0, inverse;
    float eps, freq_base, freq_scale, ext_factor, attn_factor;
    float beta_fast, beta_slow;
} rust_star_head_norm_rope_args;

typedef struct rust_star_rope_tail_args {
    int64_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    uint64_t nb0, nb1, nb2, nb3;
    int32_t n_dims, mode, n_ctx_orig, inverse;
    float freq_base, freq_scale, ext_factor, attn_factor;
    float beta_fast, beta_slow;
    bool src2;
} rust_star_rope_tail_args;

typedef struct rust_star_kv_store_args {
    int32_t head_dim, n_rot, raw_row;
} rust_star_kv_store_args;

typedef struct rust_star_flash_pad_args {
    int32_t ne11, ne_12_2, ne_12_3;
    uint64_t nb11, nb12, nb13;
    uint64_t nb21, nb22, nb23;
    int32_t ne31, ne32, ne33;
    uint64_t nb31, nb32, nb33;
} rust_star_flash_pad_args;

typedef struct rust_star_flash_vec_args {
    int32_t ne01, ne02, ne03;
    uint64_t nb01, nb02, nb03;
    int32_t ne11, ne_12_2, ne_12_3, ns10;
    uint64_t nb11, nb12, nb13;
    int32_t ns20;
    uint64_t nb21, nb22, nb23;
    int32_t ne31, ne32, ne33;
    uint64_t nb31, nb32, nb33;
    int32_t ne1, ne2, ne3;
    float scale, max_bias, m0, m1;
    int32_t n_head_log2;
    float logit_softcap;
} rust_star_flash_vec_args;

typedef struct rust_star_flash_reduce_args {
    int32_t nrows;
} rust_star_flash_reduce_args;

typedef struct rust_star_q8_mv_id_args {
    int32_t nei0, nei1;
    uint64_t nbi1;
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13;
    uint64_t nb10, nb11, nb12;
    int32_t ne0, ne1;
    uint64_t nb1;
    int32_t nr0;
    int32_t tp_rank, tp_world, tp_addend, tp_expert_base;
} rust_star_q8_mv_id_args;

typedef struct rust_star_hc_expand_args {
    int64_t n_embd, n_hc, n_tokens;
    uint64_t nb_block0, nb_block1;
    uint64_t nb_add0, nb_add1;
    uint64_t nb_res0, nb_res1, nb_res2;
    uint64_t nb_post0, nb_post1;
    uint64_t nb_comb0, nb_comb1, nb_comb2;
    uint64_t nb0, nb1, nb2;
    int32_t has_add;
} rust_star_hc_expand_args;

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
    uint64_t q_a_norm_offset,
    uint64_t q_a_norm_bytes,
    uint64_t kv_offset,
    uint64_t kv_bytes,
    uint64_t kv_norm_offset,
    uint64_t kv_norm_bytes,
    uint64_t q_b_offset,
    uint64_t q_b_bytes,
    uint64_t attn_sinks_offset,
    uint64_t attn_sinks_bytes,
    uint64_t attn_output_a_offset,
    uint64_t attn_output_a_bytes,
    uint64_t attn_output_b_offset,
    uint64_t attn_output_b_bytes,
    float *mixes,
    float *split,
    float *collapsed,
    float *attn_norm,
    float *q_lora,
    float *q_lora_norm,
    float *kv_raw,
    float *kv_norm,
    float *q_raw,
    float *q_cur,
    float *kv_rope,
    float *kv_cur,
    float *cache_rows,
    const float *cache_row0,
    float *attention_raw,
    float *attention_back,
    float *attention_low,
    float *attention_out,
    float *after_attention_hc,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes)
{
    const uint32_t n_embd = 4096;
    const uint32_t n_hc = 4;
    const uint32_t hc_dim = n_embd * n_hc;
    const uint32_t mix_hc = 24;
    const uint32_t q_elements = 1024;
    const uint32_t kv_elements = 512;
    const uint32_t q_raw_elements = 64u*512u;
    const BOOL extended = q_lora_norm && kv_raw && kv_norm && q_raw;
    const BOOL partial_extended = q_lora_norm || kv_raw || kv_norm || q_raw;
    const BOOL rope_and_store = q_cur && kv_rope && kv_cur && cache_rows;
    const BOOL partial_rope_and_store = q_cur || kv_rope || kv_cur || cache_rows;
    const BOOL attention_read = cache_row0 && attention_raw && attention_back;
    const BOOL partial_attention_read = cache_row0 || attention_raw || attention_back;
    const BOOL attention_output = attention_low && attention_out && after_attention_hc;
    const BOOL partial_attention_output = attention_low || attention_out || after_attention_hc;
    if (!opaque_context || !model_mapping || !mixes || !split || !collapsed ||
        !attn_norm || !q_lora || !result || n_vocab == 0 || token >= n_vocab) {
        return fail_with_message(error, error_bytes, @"attention ingress received invalid inputs");
    }
    if (partial_extended && !extended) {
        return fail_with_message(error, error_bytes, @"attention setup outputs must be all present or all absent");
    }
    if (partial_rope_and_store && (!rope_and_store || !extended)) {
        return fail_with_message(error, error_bytes,
            @"attention RoPE/cache outputs require the complete attention setup output set");
    }
    if (partial_attention_read && (!attention_read || !rope_and_store)) {
        return fail_with_message(error, error_bytes,
            @"attention-read outputs require the complete RoPE/cache output set");
    }
    if (partial_attention_output && (!attention_output || !attention_read)) {
        return fail_with_message(error, error_bytes,
            @"attention-output results require the complete attention-read output set");
    }
    if (embedding_bytes != (uint64_t)n_embd*n_vocab*sizeof(uint16_t) ||
        hc_fn_bytes != (uint64_t)hc_dim*mix_hc*sizeof(uint16_t) ||
        hc_scale_bytes != 3u*sizeof(float) || hc_base_bytes != mix_hc*sizeof(float) ||
        attn_norm_bytes != n_embd*sizeof(float) ||
        q_a_bytes != (uint64_t)(n_embd/32u)*34u*q_elements) {
        return fail_with_message(error, error_bytes, @"attention ingress tensor shapes are invalid");
    }
    if (extended &&
        (q_a_norm_bytes != q_elements*sizeof(float) ||
         kv_bytes != (uint64_t)(n_embd/32u)*34u*kv_elements ||
         kv_norm_bytes != kv_elements*sizeof(float) ||
         q_b_bytes != (uint64_t)(q_elements/32u)*34u*q_raw_elements)) {
        return fail_with_message(error, error_bytes, @"attention setup tensor shapes are invalid");
    }
    if (attention_read && attn_sinks_bytes != 64u*sizeof(float)) {
        return fail_with_message(error, error_bytes, @"attention sink tensor shape is invalid");
    }
    if (attention_output &&
        (attn_output_a_bytes != 8ull*1024ull*((4096ull/32ull)*34ull) ||
         attn_output_b_bytes != 4096ull*((8192ull/32ull)*34ull))) {
        return fail_with_message(error, error_bytes, @"attention-output tensor shapes are invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_get_rows_f16_pipeline(context, error, error_bytes) ||
            !ensure_attention_ingress_pipelines(context, error, error_bytes) ||
            !ensure_q8_projection_pipeline(context, error, error_bytes) ||
            (attention_output && !ensure_attention_output_pipelines(context, error, error_bytes))) return 0;
        memset(result, 0, sizeof(*result));

        NSUInteger embedding_inner = 0, hc_fn_inner = 0, scale_inner = 0;
        NSUInteger base_inner = 0, norm_inner = 0, q_inner = 0;
        NSUInteger q_norm_inner = 0, kv_inner = 0, kv_norm_inner = 0, q_b_inner = 0;
        NSUInteger sinks_inner = 0;
        NSUInteger output_a_inner = 0, output_b_inner = 0;
        BOOL matches[13] = {NO, NO, NO, NO, NO, NO, NO, NO, NO, NO, NO, NO, NO};
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
        id<MTLBuffer> q_norm_weights = nil;
        id<MTLBuffer> kv_weights = nil;
        id<MTLBuffer> kv_norm_weights = nil;
        id<MTLBuffer> q_b_weights = nil;
        id<MTLBuffer> sinks_weights = nil;
        id<MTLBuffer> output_a_weights = nil;
        id<MTLBuffer> output_b_weights = nil;
        if (extended) {
            q_norm_weights = wrap_model_range(context, model_mapping, model_bytes,
                q_a_norm_offset, q_a_norm_bytes, &q_norm_inner, &matches[6], error, error_bytes);
            kv_weights = wrap_model_range(context, model_mapping, model_bytes,
                kv_offset, kv_bytes, &kv_inner, &matches[7], error, error_bytes);
            kv_norm_weights = wrap_model_range(context, model_mapping, model_bytes,
                kv_norm_offset, kv_norm_bytes, &kv_norm_inner, &matches[8], error, error_bytes);
            q_b_weights = wrap_model_range(context, model_mapping, model_bytes,
                q_b_offset, q_b_bytes, &q_b_inner, &matches[9], error, error_bytes);
            if (!q_norm_weights || !kv_weights || !kv_norm_weights || !q_b_weights) return 0;
        }
        if (attention_read) {
            sinks_weights = wrap_model_range(context, model_mapping, model_bytes,
                attn_sinks_offset, attn_sinks_bytes, &sinks_inner, &matches[10], error, error_bytes);
            if (!sinks_weights) return 0;
        }
        if (attention_output) {
            output_a_weights = wrap_model_range(context, model_mapping, model_bytes,
                attn_output_a_offset, attn_output_a_bytes, &output_a_inner, &matches[11], error, error_bytes);
            output_b_weights = wrap_model_range(context, model_mapping, model_bytes,
                attn_output_b_offset, attn_output_b_bytes, &output_b_inner, &matches[12], error, error_bytes);
            if (!output_a_weights || !output_b_weights) return 0;
        }

        id<MTLBuffer> embedding = [context.device newBufferWithLength:n_embd*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cur_hc = [context.device newBufferWithLength:hc_dim*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> flat_hc = [context.device newBufferWithLength:hc_dim*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> mix_buffer = [context.device newBufferWithLength:mix_hc*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> split_buffer = [context.device newBufferWithLength:mix_hc*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> collapsed_buffer = [context.device newBufferWithLength:n_embd*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> norm_buffer = [context.device newBufferWithLength:n_embd*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_buffer = [context.device newBufferWithLength:q_elements*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_norm_buffer = extended ? [context.device newBufferWithLength:q_elements*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> kv_raw_buffer = extended ? [context.device newBufferWithLength:kv_elements*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> kv_norm_buffer = extended ? [context.device newBufferWithLength:kv_elements*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> q_raw_buffer = extended ? [context.device newBufferWithLength:q_raw_elements*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> q_cur_buffer = rope_and_store ? [context.device newBufferWithLength:q_raw_elements*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> kv_rope_buffer = rope_and_store ? [context.device newBufferWithLength:kv_elements*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> cache_buffer = rope_and_store ? [context.device newBufferWithLength:3u*kv_elements*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        const NSUInteger staged_kv_bytes = 2u*kv_elements*sizeof(uint16_t);
        const NSUInteger mask_bytes = 2u*sizeof(uint16_t);
        const NSUInteger flash_pad_bytes = 2u*32u*kv_elements*sizeof(uint16_t) + 32u*sizeof(uint16_t);
        const NSUInteger flash_tmp_bytes = 64u*kv_elements*32u*sizeof(float) + 64u*64u*sizeof(float);
        id<MTLBuffer> staged_kv_buffer = attention_read ? [context.device newBufferWithLength:staged_kv_bytes options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> mask_buffer = attention_read ? [context.device newBufferWithLength:mask_bytes options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> flash_pad_buffer = attention_read ? [context.device newBufferWithLength:flash_pad_bytes options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> flash_tmp_buffer = attention_read ? [context.device newBufferWithLength:flash_tmp_bytes options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> attention_raw_buffer = attention_read ? [context.device newBufferWithLength:q_raw_elements*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> attention_back_buffer = attention_read ? [context.device newBufferWithLength:q_raw_elements*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> attention_low_buffer = attention_output ? [context.device newBufferWithLength:8192u*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> attention_out_buffer = attention_output ? [context.device newBufferWithLength:n_embd*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> after_attention_hc_buffer = attention_output ? [context.device newBufferWithLength:hc_dim*sizeof(float) options:MTLResourceStorageModeShared] : nil;
        if (!embedding || !cur_hc || !flat_hc || !mix_buffer || !split_buffer ||
            !collapsed_buffer || !norm_buffer || !q_buffer) {
            return fail_with_message(error, error_bytes, @"failed to allocate attention ingress buffers");
        }
        if (extended && (!q_norm_buffer || !kv_raw_buffer || !kv_norm_buffer || !q_raw_buffer)) {
            return fail_with_message(error, error_bytes, @"failed to allocate attention setup buffers");
        }
        if (rope_and_store && (!q_cur_buffer || !kv_rope_buffer || !cache_buffer)) {
            return fail_with_message(error, error_bytes, @"failed to allocate attention RoPE/cache buffers");
        }
        if (attention_read && (!staged_kv_buffer || !mask_buffer || !flash_pad_buffer ||
            !flash_tmp_buffer || !attention_raw_buffer || !attention_back_buffer)) {
            return fail_with_message(error, error_bytes, @"failed to allocate attention-read buffers");
        }
        if (attention_output && (!attention_low_buffer || !attention_out_buffer || !after_attention_hc_buffer)) {
            return fail_with_message(error, error_bytes, @"failed to allocate attention-output buffers");
        }
        if (rope_and_store) {
            float *cache = cache_buffer.contents;
            for (uint32_t index = 0; index < 3u*kv_elements; index++) cache[index] = -12345.5f;
            if (attention_read) memcpy(cache, cache_row0, kv_elements*sizeof(float));
        }
        if (attention_read) memset(mask_buffer.contents, 0, mask_bytes);

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
        const uint64_t kv_row_bytes = (uint64_t)(n_embd/32u)*34u;
        rust_star_q8_mv_args kv_mv = {
            .ne00=(int32_t)n_embd, .ne01=(int32_t)kv_elements, .ne02=1,
            .nb00=34, .nb01=kv_row_bytes, .nb02=kv_bytes, .nb03=kv_bytes,
            .ne10=(int32_t)n_embd, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=f32_row_bytes, .nb12=f32_row_bytes, .nb13=f32_row_bytes,
            .ne0=(int32_t)kv_elements, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_qkv_norm_args qkv_norm_args = {
            .q_n=(int32_t)q_elements, .q_n4=(int32_t)(q_elements/4u),
            .kv_n=(int32_t)kv_elements, .kv_n4=(int32_t)(kv_elements/4u),
            .q_row_stride=q_elements*sizeof(float),
            .kv_row_stride=kv_elements*sizeof(float),
            .eps=1.0e-6f,
        };
        const uint64_t q_b_row_bytes = (uint64_t)(q_elements/32u)*34u;
        rust_star_q8_mv_args q_b_mv = {
            .ne00=(int32_t)q_elements, .ne01=(int32_t)q_raw_elements, .ne02=1,
            .nb00=34, .nb01=q_b_row_bytes, .nb02=q_b_bytes, .nb03=q_b_bytes,
            .ne10=(int32_t)q_elements, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=q_elements*sizeof(float),
            .nb12=q_elements*sizeof(float), .nb13=q_elements*sizeof(float),
            .ne0=(int32_t)q_raw_elements, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_head_norm_rope_args q_rope_args = {
            .n_head=64, .head_dim=512, .head_dim4=128, .n_dims=64,
            .n_ctx_orig=0, .pos0=1, .inverse=0,
            .eps=1.0e-6f, .freq_base=10000.0f, .freq_scale=1.0f,
            .ext_factor=0.0f, .attn_factor=1.0f,
            .beta_fast=32.0f, .beta_slow=1.0f,
        };
        const uint64_t kv_row_f32_bytes = (uint64_t)kv_elements*sizeof(float);
        rust_star_rope_tail_args kv_rope_args = {
            .ne00=512, .ne01=1, .ne02=1, .ne03=1,
            .nb00=sizeof(float), .nb01=kv_row_f32_bytes,
            .nb02=kv_row_f32_bytes, .nb03=kv_row_f32_bytes,
            .nb0=sizeof(float), .nb1=kv_row_f32_bytes,
            .nb2=kv_row_f32_bytes, .nb3=kv_row_f32_bytes,
            .n_dims=64, .mode=0, .n_ctx_orig=0, .inverse=0,
            .freq_base=10000.0f, .freq_scale=1.0f, .ext_factor=0.0f,
            .attn_factor=1.0f, .beta_fast=32.0f, .beta_slow=1.0f,
            .src2=false,
        };
        rust_star_kv_store_args kv_store_args = {
            .head_dim=512, .n_rot=64, .raw_row=1,
        };
        const uint64_t attention_head_bytes = (uint64_t)kv_elements*sizeof(float);
        const uint64_t staged_row_bytes = (uint64_t)kv_elements*sizeof(uint16_t);
        rust_star_flash_pad_args flash_pad_args = {
            .ne11=2, .ne_12_2=1, .ne_12_3=1,
            .nb11=staged_row_bytes, .nb12=2u*staged_row_bytes, .nb13=2u*staged_row_bytes,
            .nb21=staged_row_bytes, .nb22=2u*staged_row_bytes, .nb23=2u*staged_row_bytes,
            .ne31=1, .ne32=1, .ne33=1,
            .nb31=mask_bytes, .nb32=mask_bytes, .nb33=mask_bytes,
        };
        rust_star_flash_vec_args flash_vec_args = {
            .ne01=1, .ne02=64, .ne03=1,
            .nb01=64u*attention_head_bytes, .nb02=attention_head_bytes,
            .nb03=64u*attention_head_bytes,
            .ne11=2, .ne_12_2=1, .ne_12_3=1, .ns10=512,
            .nb11=staged_row_bytes, .nb12=2u*staged_row_bytes, .nb13=2u*staged_row_bytes,
            .ns20=512,
            .nb21=staged_row_bytes, .nb22=2u*staged_row_bytes, .nb23=2u*staged_row_bytes,
            .ne31=1, .ne32=1, .ne33=1,
            .nb31=mask_bytes, .nb32=mask_bytes, .nb33=mask_bytes,
            .ne1=64, .ne2=1, .ne3=1,
            .scale=1.0f/sqrtf(512.0f), .max_bias=0.0f, .m0=0.0f, .m1=0.0f,
            .n_head_log2=0, .logit_softcap=0.0f,
        };
        rust_star_flash_reduce_args flash_reduce_args = { .nrows=64 };
        rust_star_rope_tail_args attention_inverse_args = kv_rope_args;
        attention_inverse_args.ne01 = 64;
        attention_inverse_args.nb01 = attention_head_bytes;
        attention_inverse_args.nb02 = 64u*attention_head_bytes;
        attention_inverse_args.nb03 = 64u*attention_head_bytes;
        attention_inverse_args.nb1 = attention_head_bytes;
        attention_inverse_args.nb2 = 64u*attention_head_bytes;
        attention_inverse_args.nb3 = 64u*attention_head_bytes;
        attention_inverse_args.inverse = 1;
        const uint64_t output_a_row_bytes = (4096ull/32ull)*34ull;
        rust_star_q8_mv_id_args output_low_args = {
            .nei0=8, .nei1=1, .nbi1=0,
            .ne00=4096, .ne01=1024, .ne02=8,
            .nb00=34, .nb01=output_a_row_bytes, .nb02=1024ull*output_a_row_bytes,
            .ne10=4096, .ne11=8, .ne12=1, .ne13=1,
            .nb10=sizeof(float), .nb11=4096ull*sizeof(float),
            .nb12=8ull*4096ull*sizeof(float),
            .ne0=1024, .ne1=8, .nb1=1024ull*sizeof(float), .nr0=2,
        };
        const uint64_t output_b_row_bytes = (8192ull/32ull)*34ull;
        rust_star_q8_mv_args output_hc_mv_args = {
            .ne00=8192, .ne01=4096, .ne02=1,
            .nb00=34, .nb01=output_b_row_bytes,
            .nb02=4096ull*output_b_row_bytes, .nb03=4096ull*output_b_row_bytes,
            .ne10=8192, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=8192ull*sizeof(float),
            .nb12=8192ull*sizeof(float), .nb13=8192ull*sizeof(float),
            .ne0=4096, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_hc_expand_args output_hc_args = {
            .n_embd=4096, .n_hc=4, .n_tokens=1,
            .nb_block0=sizeof(float), .nb_block1=4096ull*sizeof(float),
            .nb_add0=sizeof(float), .nb_add1=4096ull*sizeof(float),
            .nb_res0=sizeof(float), .nb_res1=4096ull*sizeof(float),
            .nb_res2=4ull*4096ull*sizeof(float),
            .nb_post0=sizeof(float), .nb_post1=24ull*sizeof(float),
            .nb_comb0=sizeof(float), .nb_comb1=4ull*sizeof(float),
            .nb_comb2=24ull*sizeof(float),
            .nb0=sizeof(float), .nb1=4096ull*sizeof(float),
            .nb2=4ull*4096ull*sizeof(float), .has_add=0,
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

        if (extended) {
            [encoder setComputePipelineState:context.q8ProjectionPipeline];
            [encoder setBytes:&kv_mv length:sizeof(kv_mv) atIndex:0];
            [encoder setBuffer:kv_weights offset:kv_inner atIndex:1];
            [encoder setBuffer:norm_buffer offset:0 atIndex:2];
            [encoder setBuffer:kv_raw_buffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(kv_elements/2u,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];

            [encoder setComputePipelineState:context.qkvNormPipeline];
            [encoder setBytes:&qkv_norm_args length:sizeof(qkv_norm_args) atIndex:0];
            [encoder setBuffer:q_buffer offset:0 atIndex:1];
            [encoder setBuffer:q_norm_weights offset:q_norm_inner atIndex:2];
            [encoder setBuffer:q_norm_buffer offset:0 atIndex:3];
            [encoder setBuffer:kv_raw_buffer offset:0 atIndex:4];
            [encoder setBuffer:kv_norm_weights offset:kv_norm_inner atIndex:5];
            [encoder setBuffer:kv_norm_buffer offset:0 atIndex:6];
            [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,2,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            [encoder setComputePipelineState:context.q8ProjectionPipeline];
            [encoder setBytes:&q_b_mv length:sizeof(q_b_mv) atIndex:0];
            [encoder setBuffer:q_b_weights offset:q_b_inner atIndex:1];
            [encoder setBuffer:q_norm_buffer offset:0 atIndex:2];
            [encoder setBuffer:q_raw_buffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(q_raw_elements/2u,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
        }
        [encoder endEncoding];

        if (rope_and_store) {
            id<MTLBlitCommandEncoder> q_copy = [command blitCommandEncoder];
            if (!q_copy) return fail_with_message(error, error_bytes, @"failed to create Q snapshot encoder");
            [q_copy copyFromBuffer:q_raw_buffer sourceOffset:0
                          toBuffer:q_cur_buffer destinationOffset:0
                              size:q_raw_elements*sizeof(float)];
            [q_copy endEncoding];

            encoder = [command computeCommandEncoder];
            if (!encoder) return fail_with_message(error, error_bytes, @"failed to create RoPE encoder");
            [encoder setComputePipelineState:context.headNormRopePipeline];
            [encoder setBytes:&q_rope_args length:sizeof(q_rope_args) atIndex:0];
            [encoder setBuffer:q_cur_buffer offset:0 atIndex:1];
            [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(64,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            [encoder setComputePipelineState:context.ropeTailPipeline];
            [encoder setBytes:&kv_rope_args length:sizeof(kv_rope_args) atIndex:0];
            [encoder setBuffer:kv_norm_buffer offset:0 atIndex:1];
            int32_t rope_position = 1;
            [encoder setBytes:&rope_position length:sizeof(rope_position) atIndex:2];
            [encoder setBuffer:kv_norm_buffer offset:0 atIndex:3];
            [encoder setBuffer:kv_norm_buffer offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder endEncoding];

            id<MTLBlitCommandEncoder> kv_copy = [command blitCommandEncoder];
            if (!kv_copy) return fail_with_message(error, error_bytes, @"failed to create KV snapshot encoder");
            [kv_copy copyFromBuffer:kv_norm_buffer sourceOffset:0
                           toBuffer:kv_rope_buffer destinationOffset:0
                               size:kv_elements*sizeof(float)];
            [kv_copy endEncoding];

            encoder = [command computeCommandEncoder];
            if (!encoder) return fail_with_message(error, error_bytes, @"failed to create KV-store encoder");
            [encoder setComputePipelineState:context.kvStorePipeline];
            [encoder setBytes:&kv_store_args length:sizeof(kv_store_args) atIndex:0];
            [encoder setBuffer:kv_norm_buffer offset:0 atIndex:1];
            [encoder setBuffer:cache_buffer offset:0 atIndex:2];
            [encoder setThreadgroupMemoryLength:64u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                 threadsPerThreadgroup:MTLSizeMake(64,1,1)];

            if (attention_read) {
                uint32_t staged_elements = 2u*kv_elements;
                [encoder setComputePipelineState:context.cpyF32F16Pipeline];
                [encoder setBytes:&staged_elements length:sizeof(staged_elements) atIndex:0];
                [encoder setBuffer:cache_buffer offset:0 atIndex:1];
                [encoder setBuffer:staged_kv_buffer offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                     threadsPerThreadgroup:MTLSizeMake(256,1,1)];

                [encoder setComputePipelineState:context.flashPadPipeline];
                [encoder setBytes:&flash_pad_args length:sizeof(flash_pad_args) atIndex:0];
                [encoder setBuffer:staged_kv_buffer offset:0 atIndex:1];
                [encoder setBuffer:staged_kv_buffer offset:0 atIndex:2];
                [encoder setBuffer:mask_buffer offset:0 atIndex:3];
                [encoder setBuffer:flash_pad_buffer offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(32,1,1)
                     threadsPerThreadgroup:MTLSizeMake(32,1,1)];

                [encoder setComputePipelineState:context.flashVecPipeline];
                [encoder setBytes:&flash_vec_args length:sizeof(flash_vec_args) atIndex:0];
                [encoder setBuffer:q_cur_buffer offset:0 atIndex:1];
                [encoder setBuffer:staged_kv_buffer offset:0 atIndex:2];
                [encoder setBuffer:staged_kv_buffer offset:0 atIndex:3];
                [encoder setBuffer:mask_buffer offset:0 atIndex:4];
                [encoder setBuffer:sinks_weights offset:sinks_inner atIndex:5];
                [encoder setBuffer:flash_pad_buffer offset:0 atIndex:6];
                [encoder setBuffer:flash_tmp_buffer offset:0 atIndex:7];
                [encoder setThreadgroupMemoryLength:3328u atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake(1,64,32)
                     threadsPerThreadgroup:MTLSizeMake(32,1,1)];

                [encoder setComputePipelineState:context.flashReducePipeline];
                [encoder setBytes:&flash_reduce_args length:sizeof(flash_reduce_args) atIndex:0];
                [encoder setBuffer:flash_tmp_buffer offset:0 atIndex:1];
                [encoder setBuffer:attention_raw_buffer offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(64,1,1)
                     threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
            }
            [encoder endEncoding];

            if (attention_read) {
                id<MTLBlitCommandEncoder> attention_copy = [command blitCommandEncoder];
                if (!attention_copy) return fail_with_message(error, error_bytes, @"failed to create attention snapshot encoder");
                [attention_copy copyFromBuffer:attention_raw_buffer sourceOffset:0
                                      toBuffer:attention_back_buffer destinationOffset:0
                                          size:q_raw_elements*sizeof(float)];
                [attention_copy endEncoding];

                encoder = [command computeCommandEncoder];
                if (!encoder) return fail_with_message(error, error_bytes, @"failed to create inverse-RoPE encoder");
                [encoder setComputePipelineState:context.ropeTailPipeline];
                [encoder setBytes:&attention_inverse_args length:sizeof(attention_inverse_args) atIndex:0];
                [encoder setBuffer:attention_back_buffer offset:0 atIndex:1];
                int32_t inverse_position = 1;
                [encoder setBytes:&inverse_position length:sizeof(inverse_position) atIndex:2];
                [encoder setBuffer:attention_back_buffer offset:0 atIndex:3];
                [encoder setBuffer:attention_back_buffer offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(64,1,1)
                     threadsPerThreadgroup:MTLSizeMake(256,1,1)];

                if (attention_output) {
                    [encoder setComputePipelineState:context.attentionOutputLowPipeline];
                    [encoder setBytes:&output_low_args length:sizeof(output_low_args) atIndex:0];
                    [encoder setBuffer:output_a_weights offset:output_a_inner atIndex:1];
                    [encoder setBuffer:attention_back_buffer offset:0 atIndex:2];
                    [encoder setBuffer:attention_low_buffer offset:0 atIndex:3];
                    [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(512,1,8)
                         threadsPerThreadgroup:MTLSizeMake(32,4,1)];

                    [encoder setComputePipelineState:context.attentionOutputHcPipeline];
                    [encoder setBytes:&output_hc_mv_args length:sizeof(output_hc_mv_args) atIndex:0];
                    [encoder setBytes:&output_hc_args length:sizeof(output_hc_args) atIndex:1];
                    [encoder setBuffer:output_b_weights offset:output_b_inner atIndex:2];
                    [encoder setBuffer:attention_low_buffer offset:0 atIndex:3];
                    [encoder setBuffer:attention_out_buffer offset:0 atIndex:4];
                    [encoder setBuffer:cur_hc offset:0 atIndex:5];
                    [encoder setBuffer:split_buffer offset:4u*sizeof(float) atIndex:6];
                    [encoder setBuffer:split_buffer offset:8u*sizeof(float) atIndex:7];
                    [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:8];
                    [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(2048,1,1)
                         threadsPerThreadgroup:MTLSizeMake(32,4,1)];
                }
                [encoder endEncoding];
            }
        }

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        memcpy(mixes, mix_buffer.contents, mix_hc*sizeof(float));
        memcpy(split, split_buffer.contents, mix_hc*sizeof(float));
        memcpy(collapsed, collapsed_buffer.contents, n_embd*sizeof(float));
        memcpy(attn_norm, norm_buffer.contents, n_embd*sizeof(float));
        memcpy(q_lora, q_buffer.contents, q_elements*sizeof(float));
        if (extended) {
            memcpy(q_lora_norm, q_norm_buffer.contents, q_elements*sizeof(float));
            memcpy(kv_raw, kv_raw_buffer.contents, kv_elements*sizeof(float));
            memcpy(kv_norm, kv_norm_buffer.contents, kv_elements*sizeof(float));
            memcpy(q_raw, q_raw_buffer.contents, q_raw_elements*sizeof(float));
        }
        if (rope_and_store) {
            memcpy(q_cur, q_cur_buffer.contents, q_raw_elements*sizeof(float));
            memcpy(kv_rope, kv_rope_buffer.contents, kv_elements*sizeof(float));
            memcpy(kv_cur, kv_norm_buffer.contents, kv_elements*sizeof(float));
            memcpy(cache_rows, cache_buffer.contents, 3u*kv_elements*sizeof(float));
        }
        if (attention_read) {
            memcpy(attention_raw, attention_raw_buffer.contents, q_raw_elements*sizeof(float));
            memcpy(attention_back, attention_back_buffer.contents, q_raw_elements*sizeof(float));
        }
        if (attention_output) {
            memcpy(attention_low, attention_low_buffer.contents, 8192u*sizeof(float));
            memcpy(attention_out, attention_out_buffer.contents, n_embd*sizeof(float));
            memcpy(after_attention_hc, after_attention_hc_buffer.contents, hc_dim*sizeof(float));
        }
        result->model_bytes = model_bytes;
        result->max_buffer_length = context.device.maxBufferLength;
        result->wrapped_model_ranges = attention_output ? 13 : (attention_read ? 11 : (extended ? 10 : 6));
        for (uint32_t index = 0; index < result->wrapped_model_ranges; index++) if (matches[index]) result->pointer_matches++;
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
