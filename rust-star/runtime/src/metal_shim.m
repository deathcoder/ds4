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

enum {
    RUST_STAR_COMMAND_SYNCHRONIZED = 0,
    RUST_STAR_COMMAND_CHAINED_ENQUEUE = 1,
    RUST_STAR_COMMAND_CHAINED_FINAL = 2,
    RUST_STAR_COMMAND_CHAINED_COLLECT = 3,
    RUST_STAR_COMMAND_CHAINED_TIMING = 4,
};

enum {
    RUST_STAR_INITIAL_STATE_CAPTURED = 0,
    RUST_STAR_INITIAL_STATE_COLD = 1,
};

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
#include "moe_output_source.inc"

// Imported from DwarfStar metal/dense.metal. The fixed target uses DwarfStar's
// default four simdgroups and two output rows per threadgroup. DwarfStar forces
// eight simdgroups for output dimensions above 65,536, so the same source is
// also specialized separately for the full vocabulary head.
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
@property(nonatomic, strong) id<MTLComputePipelineState> q8PrefillPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> f16PrefillPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> f16AlignedPrefillPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> f16Tail4Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q8OutputProjectionPipeline;
@property(nonatomic, strong) id<MTLLibrary> attentionIngressLibrary;
@property(nonatomic, strong) id<MTLComputePipelineState> repeatF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsNormF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> f16ProjectionPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> outputHcWeightsPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> outputHcSumNormPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorPairPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorStorePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorPackPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorSoftmaxPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorMulPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorSumRowsPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorScoreBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorPackBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorPoolBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorNormPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorShiftPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> compressorFp8Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexerQatPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> hcIngressPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> qkvNormPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> headNormRopePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ropeTailPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> kvStorePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> cpyF32F16Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> cpyF16F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> flashPadPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> flashBlkPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> flashNonvecPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> flashVecPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> flashReducePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerProbabilityPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerFinalizePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerWeightsPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerSoftplusBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerSqrtBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerHashRowsBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerGatherWeightsBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerClampSumsBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerDivideBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerScaleBatchPipeline;
@property(nonatomic, strong) id<MTLLibrary> attentionOutputLibrary;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputLowPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputHcPipeline;
@property(nonatomic, strong) id<MTLLibrary> moeOutputLibrary;
@property(nonatomic, strong) id<MTLComputePipelineState> routedPairSwigluPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routedDownSumPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sharedGateUpPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sharedDownHcPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputBatchMapPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputBatchLowPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> hcExpand4Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routedBatchMapPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routedBatchPairSwigluPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routedBatchDownPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routedBatchSumPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sharedSwigluBatchPipeline;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLBuffer>> *modelViewCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLBuffer>> *activationBufferCache;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLCommandBuffer>> *chainedCommands;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSNumber *> *chainedWallStarts;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer0FullKv;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer1FullKv;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2FullKv;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2FullQNorm;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2InputHc;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2AttnSplit;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2Tokens;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2AttnCompressed;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2AttnStateKv;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2AttnStateScore;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2IndexerCompressed;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2IndexerStateKv;
@property(nonatomic, strong) id<MTLBuffer> prefillLayer2IndexerStateScore;
@property(nonatomic, assign) uint32_t prefillKvRows;
@property(nonatomic, assign) double chainedWallEnd;
@property(nonatomic, assign) BOOL chainedReady;
@property(nonatomic, assign) uint32_t chainedFinalLayer;
@property(nonatomic, assign) const void *modelMapping;
@property(nonatomic, assign) uint64_t modelBytes;
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
        context.modelViewCache = [NSMutableDictionary dictionary];
        context.activationBufferCache = [NSMutableDictionary dictionary];
        context.chainedCommands = [NSMutableDictionary dictionary];
        context.chainedWallStarts = [NSMutableDictionary dictionary];
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
    simdgroups = 8;
    constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&simdgroups type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> output_function =
        [library newFunctionWithName:@"kernel_mul_mv_q8_0_f32"
                      constantValues:constants error:&compile_error];
    if (!output_function) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    context.q8OutputProjectionPipeline =
        [context.device newComputePipelineStateWithFunction:output_function error:&compile_error];
    if (!context.q8OutputProjectionPipeline) {
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
    id<MTLFunction> cpyF16F32 = [library newFunctionWithName:@"kernel_cpy_contig_f16_f32_4"];
    bool enabled = true;
    bool disabled = false;
    int32_t ncpsg = 32;
    MTLFunctionConstantValues *padConstants = [MTLFunctionConstantValues new];
    [padConstants setConstantValue:&enabled type:MTLDataTypeBool atIndex:100];
    [padConstants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:125];
    id<MTLFunction> flashPad = [library newFunctionWithName:@"kernel_flash_attn_ext_pad"
                                            constantValues:padConstants error:&compile_error];
    int32_t nqptg = 8, nonvecNcpsg = 64;
    MTLFunctionConstantValues *blkConstants = [MTLFunctionConstantValues new];
    [blkConstants setConstantValue:&nqptg type:MTLDataTypeInt atIndex:224];
    [blkConstants setConstantValue:&nonvecNcpsg type:MTLDataTypeInt atIndex:225];
    id<MTLFunction> flashBlk = [library newFunctionWithName:@"kernel_flash_attn_ext_blk"
                                            constantValues:blkConstants error:&compile_error];
    int32_t headDim = 512, nsg = 1, nwg = 32;
    int32_t nonvecNsg = 8;
    MTLFunctionConstantValues *nonvecConstants = [MTLFunctionConstantValues new];
    [nonvecConstants setConstantValue:&enabled type:MTLDataTypeBool atIndex:300];
    [nonvecConstants setConstantValue:&enabled type:MTLDataTypeBool atIndex:301];
    [nonvecConstants setConstantValue:&disabled type:MTLDataTypeBool atIndex:302];
    [nonvecConstants setConstantValue:&disabled type:MTLDataTypeBool atIndex:303];
    [nonvecConstants setConstantValue:&disabled type:MTLDataTypeBool atIndex:304];
    [nonvecConstants setConstantValue:&disabled type:MTLDataTypeBool atIndex:310];
    [nonvecConstants setConstantValue:&headDim type:MTLDataTypeInt atIndex:320];
    [nonvecConstants setConstantValue:&headDim type:MTLDataTypeInt atIndex:321];
    [nonvecConstants setConstantValue:&nonvecNsg type:MTLDataTypeInt atIndex:322];
    id<MTLFunction> flashNonvec =
        [library newFunctionWithName:@"kernel_flash_attn_ext_f16_dk512_dv512"
                      constantValues:nonvecConstants error:&compile_error];
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
    id<MTLFunction> routerProbability = [library newFunctionWithName:@"kernel_dsv4_softplus_sqrt_f32_4"];
    id<MTLFunction> routerFinalize = [library newFunctionWithName:@"kernel_dsv4_router_finalize_one"];
    id<MTLFunction> routerWeights = [library newFunctionWithName:@"kernel_dsv4_router_weights_one"];
    id<MTLFunction> routerSoftplusBatch = [library newFunctionWithName:@"rust_star_router_softplus_batch"];
    id<MTLFunction> routerSqrtBatch = [library newFunctionWithName:@"rust_star_router_sqrt_batch"];
    id<MTLFunction> routerHashRowsBatch = [library newFunctionWithName:@"rust_star_router_hash_rows_batch"];
    id<MTLFunction> routerGatherWeightsBatch = [library newFunctionWithName:@"rust_star_router_gather_weights_batch"];
    id<MTLFunction> routerClampSumsBatch = [library newFunctionWithName:@"rust_star_router_clamp_sums_batch"];
    id<MTLFunction> routerDivideBatch = [library newFunctionWithName:@"rust_star_router_divide_batch"];
    id<MTLFunction> routerScaleBatch = [library newFunctionWithName:@"rust_star_router_scale_batch"];
    int16_t simdgroups = 8;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&simdgroups type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> projection = [library newFunctionWithName:@"kernel_mul_mv_f16_f32_4"
                                               constantValues:constants
                                                        error:&compile_error];
    id<MTLFunction> compressorPair = [library newFunctionWithName:@"kernel_mul_mv_f16_f32_pair_4"
                                                   constantValues:constants error:&compile_error];
    id<MTLFunction> compressorStore = [library newFunctionWithName:@"kernel_dsv4_compressor_store_one"];
    id<MTLFunction> compressorPack = [library newFunctionWithName:@"rust_star_compressor_pack_ratio4_one"];
    id<MTLFunction> compressorSoftmax = [library newFunctionWithName:@"kernel_soft_max_f32_4"];
    id<MTLFunction> compressorMul = [library newFunctionWithName:@"rust_star_mul_f32"];
    int16_t sumOperation = 10;
    MTLFunctionConstantValues *sumConstants = [MTLFunctionConstantValues new];
    [sumConstants setConstantValue:&sumOperation type:MTLDataTypeShort atIndex:1400];
    id<MTLFunction> compressorSumRows = [library newFunctionWithName:@"kernel_sum_rows_f32_f32"
                                                      constantValues:sumConstants error:&compile_error];
    id<MTLFunction> compressorScoreBatch =
        [library newFunctionWithName:@"rust_star_compressor_score_ape_f16_batch"];
    id<MTLFunction> compressorPackBatch =
        [library newFunctionWithName:@"rust_star_compressor_pack_ratio4_batch"];
    id<MTLFunction> compressorPoolBatch =
        [library newFunctionWithName:@"rust_star_compressor_softmax_pool_batch"];
    id<MTLFunction> compressorNorm = [library newFunctionWithName:@"kernel_rms_norm_f32_weighted_4"];
    id<MTLFunction> compressorShift = [library newFunctionWithName:@"kernel_dsv4_ratio4_shift_f32"];
    id<MTLFunction> compressorFp8 = [library newFunctionWithName:@"kernel_dsv4_fp8_kv_quantize_f32"];
    id<MTLFunction> indexerQat = [library newFunctionWithName:@"kernel_dsv4_indexer_hadamard_fp4_f32"];
    if (!repeat || !norm || !hc || !qkvNorm || !headNormRope || !ropeTail ||
        !kvStore || !cpyF32F16 || !cpyF16F32 || !flashPad || !flashBlk ||
        !flashNonvec || !flashVec || !flashReduce ||
        !projection || !compressorPair || !compressorStore || !compressorPack ||
        !compressorSoftmax || !compressorMul || !compressorSumRows ||
        !compressorScoreBatch || !compressorPackBatch || !compressorPoolBatch ||
        !compressorNorm || !compressorShift || !compressorFp8 || !indexerQat ||
        !routerProbability || !routerFinalize || !routerWeights ||
        !routerSoftplusBatch || !routerSqrtBatch || !routerHashRowsBatch ||
        !routerGatherWeightsBatch || !routerClampSumsBatch ||
        !routerDivideBatch || !routerScaleBatch) {
        return fail_with_message(error, error_bytes,
            compile_error ? compile_error.localizedDescription : @"attention ingress kernel was not found");
    }
    context.repeatF32Pipeline = [context.device newComputePipelineStateWithFunction:repeat error:&compile_error];
    context.rmsNormF32Pipeline = [context.device newComputePipelineStateWithFunction:norm error:&compile_error];
    context.f16ProjectionPipeline = [context.device newComputePipelineStateWithFunction:projection error:&compile_error];
    context.compressorPairPipeline = [context.device newComputePipelineStateWithFunction:compressorPair error:&compile_error];
    context.compressorStorePipeline = [context.device newComputePipelineStateWithFunction:compressorStore error:&compile_error];
    context.compressorPackPipeline = [context.device newComputePipelineStateWithFunction:compressorPack error:&compile_error];
    context.compressorSoftmaxPipeline = [context.device newComputePipelineStateWithFunction:compressorSoftmax error:&compile_error];
    context.compressorMulPipeline = [context.device newComputePipelineStateWithFunction:compressorMul error:&compile_error];
    context.compressorSumRowsPipeline = [context.device newComputePipelineStateWithFunction:compressorSumRows error:&compile_error];
    context.compressorScoreBatchPipeline =
        [context.device newComputePipelineStateWithFunction:compressorScoreBatch
                                                       error:&compile_error];
    context.compressorPackBatchPipeline =
        [context.device newComputePipelineStateWithFunction:compressorPackBatch
                                                       error:&compile_error];
    context.compressorPoolBatchPipeline =
        [context.device newComputePipelineStateWithFunction:compressorPoolBatch
                                                       error:&compile_error];
    context.compressorNormPipeline = [context.device newComputePipelineStateWithFunction:compressorNorm error:&compile_error];
    context.compressorShiftPipeline = [context.device newComputePipelineStateWithFunction:compressorShift error:&compile_error];
    context.compressorFp8Pipeline = [context.device newComputePipelineStateWithFunction:compressorFp8 error:&compile_error];
    context.indexerQatPipeline = [context.device newComputePipelineStateWithFunction:indexerQat error:&compile_error];
    context.hcIngressPipeline = [context.device newComputePipelineStateWithFunction:hc error:&compile_error];
    context.qkvNormPipeline = [context.device newComputePipelineStateWithFunction:qkvNorm error:&compile_error];
    context.headNormRopePipeline = [context.device newComputePipelineStateWithFunction:headNormRope error:&compile_error];
    context.ropeTailPipeline = [context.device newComputePipelineStateWithFunction:ropeTail error:&compile_error];
    context.kvStorePipeline = [context.device newComputePipelineStateWithFunction:kvStore error:&compile_error];
    context.cpyF32F16Pipeline = [context.device newComputePipelineStateWithFunction:cpyF32F16 error:&compile_error];
    context.cpyF16F32Pipeline = [context.device newComputePipelineStateWithFunction:cpyF16F32 error:&compile_error];
    context.flashPadPipeline = [context.device newComputePipelineStateWithFunction:flashPad error:&compile_error];
    context.flashBlkPipeline = [context.device newComputePipelineStateWithFunction:flashBlk error:&compile_error];
    context.flashNonvecPipeline = [context.device newComputePipelineStateWithFunction:flashNonvec error:&compile_error];
    context.flashVecPipeline = [context.device newComputePipelineStateWithFunction:flashVec error:&compile_error];
    context.flashReducePipeline = [context.device newComputePipelineStateWithFunction:flashReduce error:&compile_error];
    context.routerProbabilityPipeline = [context.device newComputePipelineStateWithFunction:routerProbability error:&compile_error];
    context.routerFinalizePipeline = [context.device newComputePipelineStateWithFunction:routerFinalize error:&compile_error];
    context.routerWeightsPipeline = [context.device newComputePipelineStateWithFunction:routerWeights error:&compile_error];
    context.routerSoftplusBatchPipeline = [context.device newComputePipelineStateWithFunction:routerSoftplusBatch error:&compile_error];
    context.routerSqrtBatchPipeline = [context.device newComputePipelineStateWithFunction:routerSqrtBatch error:&compile_error];
    context.routerHashRowsBatchPipeline = [context.device newComputePipelineStateWithFunction:routerHashRowsBatch error:&compile_error];
    context.routerGatherWeightsBatchPipeline = [context.device newComputePipelineStateWithFunction:routerGatherWeightsBatch error:&compile_error];
    context.routerClampSumsBatchPipeline = [context.device newComputePipelineStateWithFunction:routerClampSumsBatch error:&compile_error];
    context.routerDivideBatchPipeline = [context.device newComputePipelineStateWithFunction:routerDivideBatch error:&compile_error];
    context.routerScaleBatchPipeline = [context.device newComputePipelineStateWithFunction:routerScaleBatch error:&compile_error];
    if (!context.repeatF32Pipeline || !context.rmsNormF32Pipeline ||
        !context.f16ProjectionPipeline || !context.compressorPairPipeline ||
        !context.compressorStorePipeline || !context.compressorPackPipeline ||
        !context.compressorSoftmaxPipeline || !context.compressorMulPipeline ||
        !context.compressorSumRowsPipeline ||
        !context.compressorScoreBatchPipeline ||
        !context.compressorPackBatchPipeline ||
        !context.compressorPoolBatchPipeline || !context.compressorNormPipeline ||
        !context.compressorShiftPipeline || !context.compressorFp8Pipeline ||
        !context.indexerQatPipeline || !context.hcIngressPipeline ||
        !context.qkvNormPipeline || !context.headNormRopePipeline ||
        !context.ropeTailPipeline || !context.kvStorePipeline ||
        !context.cpyF32F16Pipeline || !context.cpyF16F32Pipeline ||
        !context.flashPadPipeline || !context.flashBlkPipeline ||
        !context.flashNonvecPipeline || !context.flashVecPipeline ||
        !context.flashReducePipeline ||
        !context.routerProbabilityPipeline || !context.routerFinalizePipeline ||
        !context.routerWeightsPipeline || !context.routerSoftplusBatchPipeline ||
        !context.routerSqrtBatchPipeline || !context.routerHashRowsBatchPipeline ||
        !context.routerGatherWeightsBatchPipeline ||
        !context.routerClampSumsBatchPipeline || !context.routerDivideBatchPipeline ||
        !context.routerScaleBatchPipeline) {
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

static int ensure_moe_output_pipelines(
    RustStarMetalContext *context,
    char *error,
    size_t error_bytes)
{
    if (context.moeOutputLibrary) return 1;
    NSError *compile_error = nil;
    id<MTLLibrary> library =
        [context.device newLibraryWithSource:kMoeOutputSource
                                      options:[MTLCompileOptions new]
                                        error:&compile_error];
    if (!library) return fail_with_message(error, error_bytes, compile_error.localizedDescription);

    int16_t simdgroups = 2;
    MTLFunctionConstantValues *routed_constants = [MTLFunctionConstantValues new];
    [routed_constants setConstantValue:&simdgroups type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> routed_pair =
        [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32"
                      constantValues:routed_constants error:&compile_error];
    id<MTLFunction> routed_down =
        [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_f32"
                      constantValues:routed_constants error:&compile_error];

    simdgroups = 4;
    MTLFunctionConstantValues *shared_constants = [MTLFunctionConstantValues new];
    [shared_constants setConstantValue:&simdgroups type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> shared_gate_up =
        [library newFunctionWithName:@"kernel_dsv4_shared_gate_up_swiglu_q8_0"
                      constantValues:shared_constants error:&compile_error];
    id<MTLFunction> shared_down_hc =
        [library newFunctionWithName:@"kernel_dsv4_shared_down_hc_expand4_q8_0"
                      constantValues:shared_constants error:&compile_error];
    id<MTLFunction> output_hc_weights =
        [library newFunctionWithName:@"kernel_dsv4_output_hc_weights4"];
    id<MTLFunction> output_hc_sum_norm =
        [library newFunctionWithName:@"kernel_dsv4_hc_weighted_sum_norm4"];
    bool bc_inp = false;
    bool bc_out = false;
    MTLFunctionConstantValues *q8_prefill_constants = [MTLFunctionConstantValues new];
    [q8_prefill_constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [q8_prefill_constants setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];
    id<MTLFunction> q8_prefill =
        [library newFunctionWithName:@"kernel_mul_mm_q8_0_f32"
                      constantValues:q8_prefill_constants error:&compile_error];
    bool f16_bc_out = true;
    MTLFunctionConstantValues *f16_prefill_constants = [MTLFunctionConstantValues new];
    [f16_prefill_constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [f16_prefill_constants setConstantValue:&f16_bc_out type:MTLDataTypeBool atIndex:701];
    id<MTLFunction> f16_prefill =
        [library newFunctionWithName:@"kernel_mul_mm_f16_f32"
                      constantValues:f16_prefill_constants error:&compile_error];
    bool aligned_bc_out = false;
    MTLFunctionConstantValues *f16_aligned_constants = [MTLFunctionConstantValues new];
    [f16_aligned_constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [f16_aligned_constants setConstantValue:&aligned_bc_out
                                        type:MTLDataTypeBool atIndex:701];
    id<MTLFunction> f16_aligned_prefill =
        [library newFunctionWithName:@"kernel_mul_mm_f16_f32"
                      constantValues:f16_aligned_constants error:&compile_error];
    int16_t tail_nsg = 2;
    int16_t tail_nxpsg = 8;
    MTLFunctionConstantValues *f16_tail_constants = [MTLFunctionConstantValues new];
    [f16_tail_constants setConstantValue:&tail_nsg
                                     type:MTLDataTypeShort atIndex:600];
    [f16_tail_constants setConstantValue:&tail_nxpsg
                                     type:MTLDataTypeShort atIndex:601];
    id<MTLFunction> f16_tail4 =
        [library newFunctionWithName:@"kernel_mul_mv_ext_f16_f32_r1_4"
                      constantValues:f16_tail_constants error:&compile_error];
    id<MTLFunction> attention_output_map =
        [library newFunctionWithName:@"kernel_mul_mm_id_map0_ne20_8"];
    id<MTLFunction> attention_output_low =
        [library newFunctionWithName:@"kernel_mul_mm_id_q8_0_f32"
                      constantValues:q8_prefill_constants error:&compile_error];
    id<MTLFunction> hc_expand4 =
        [library newFunctionWithName:@"kernel_dsv4_hc_expand4"];
    id<MTLFunction> routed_batch_map =
        [library newFunctionWithName:@"kernel_mul_mm_id_map0_ne20_6"];
    id<MTLFunction> routed_batch_pair =
        [library newFunctionWithName:@"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16"];
    id<MTLFunction> routed_batch_down =
        [library newFunctionWithName:@"kernel_mul_mm_id_q2_K_f16"
                      constantValues:q8_prefill_constants error:&compile_error];
    id<MTLFunction> routed_batch_sum =
        [library newFunctionWithName:@"kernel_dsv4_moe_sum6_f32"];
    id<MTLFunction> shared_swiglu_batch =
        [library newFunctionWithName:@"kernel_swiglu_flat_f32"];
    if (!routed_pair || !routed_down || !shared_gate_up || !shared_down_hc ||
        !output_hc_weights || !output_hc_sum_norm || !q8_prefill || !f16_prefill ||
        !f16_aligned_prefill || !f16_tail4 ||
        !attention_output_map || !attention_output_low || !hc_expand4 ||
        !routed_batch_map || !routed_batch_pair || !routed_batch_down ||
        !routed_batch_sum || !shared_swiglu_batch) {
        return fail_with_message(error, error_bytes,
            compile_error ? compile_error.localizedDescription : @"MoE output kernel was not found");
    }

    context.routedPairSwigluPipeline =
        [context.device newComputePipelineStateWithFunction:routed_pair error:&compile_error];
    context.routedDownSumPipeline =
        [context.device newComputePipelineStateWithFunction:routed_down error:&compile_error];
    context.sharedGateUpPipeline =
        [context.device newComputePipelineStateWithFunction:shared_gate_up error:&compile_error];
    context.sharedDownHcPipeline =
        [context.device newComputePipelineStateWithFunction:shared_down_hc error:&compile_error];
    context.outputHcWeightsPipeline =
        [context.device newComputePipelineStateWithFunction:output_hc_weights error:&compile_error];
    context.outputHcSumNormPipeline =
        [context.device newComputePipelineStateWithFunction:output_hc_sum_norm error:&compile_error];
    context.q8PrefillPipeline =
        [context.device newComputePipelineStateWithFunction:q8_prefill error:&compile_error];
    context.f16PrefillPipeline =
        [context.device newComputePipelineStateWithFunction:f16_prefill error:&compile_error];
    context.f16AlignedPrefillPipeline =
        [context.device newComputePipelineStateWithFunction:f16_aligned_prefill
                                                       error:&compile_error];
    context.f16Tail4Pipeline =
        [context.device newComputePipelineStateWithFunction:f16_tail4
                                                       error:&compile_error];
    context.attentionOutputBatchMapPipeline =
        [context.device newComputePipelineStateWithFunction:attention_output_map error:&compile_error];
    context.attentionOutputBatchLowPipeline =
        [context.device newComputePipelineStateWithFunction:attention_output_low error:&compile_error];
    context.hcExpand4Pipeline =
        [context.device newComputePipelineStateWithFunction:hc_expand4 error:&compile_error];
    context.routedBatchMapPipeline =
        [context.device newComputePipelineStateWithFunction:routed_batch_map error:&compile_error];
    context.routedBatchPairSwigluPipeline =
        [context.device newComputePipelineStateWithFunction:routed_batch_pair error:&compile_error];
    context.routedBatchDownPipeline =
        [context.device newComputePipelineStateWithFunction:routed_batch_down error:&compile_error];
    context.routedBatchSumPipeline =
        [context.device newComputePipelineStateWithFunction:routed_batch_sum error:&compile_error];
    context.sharedSwigluBatchPipeline =
        [context.device newComputePipelineStateWithFunction:shared_swiglu_batch error:&compile_error];
    if (!context.routedPairSwigluPipeline || !context.routedDownSumPipeline ||
        !context.sharedGateUpPipeline || !context.sharedDownHcPipeline ||
        !context.outputHcWeightsPipeline || !context.outputHcSumNormPipeline ||
        !context.q8PrefillPipeline || !context.f16PrefillPipeline ||
        !context.f16AlignedPrefillPipeline || !context.f16Tail4Pipeline ||
        !context.attentionOutputBatchMapPipeline ||
        !context.attentionOutputBatchLowPipeline || !context.hcExpand4Pipeline ||
        !context.routedBatchMapPipeline || !context.routedBatchPairSwigluPipeline ||
        !context.routedBatchDownPipeline || !context.routedBatchSumPipeline ||
        !context.sharedSwigluBatchPipeline) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    context.moeOutputLibrary = library;
    return 1;
}

int rust_star_metal_prepare_decoder(
    void *opaque_context,
    char *error,
    size_t error_bytes)
{
    if (!opaque_context) {
        return fail_with_message(error, error_bytes, @"decoder preparation received a null context");
    }
    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        return ensure_get_rows_f16_pipeline(context, error, error_bytes) &&
            ensure_attention_ingress_pipelines(context, error, error_bytes) &&
            ensure_q8_projection_pipeline(context, error, error_bytes) &&
            ensure_attention_output_pipelines(context, error, error_bytes) &&
            ensure_moe_output_pipelines(context, error, error_bytes);
    }
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

typedef struct rust_star_q8_mm_args {
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
} rust_star_q8_mm_args;

typedef struct rust_star_mul_mv_ext_args {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    int16_t r2, r3;
} rust_star_mul_mv_ext_args;

typedef struct rust_star_q8_mm_id_map_args {
    int32_t ne02, ne10, ne11;
    uint64_t nb11, nb12;
    int32_t ne21, ne20;
    uint64_t nb21;
} rust_star_q8_mm_id_map_args;

typedef struct rust_star_q8_mm_id_args {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne11;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne20, ne21, ne0, ne1;
    int16_t r2, r3;
    int32_t tp_rank, tp_world, tp_expert_base;
} rust_star_q8_mm_id_args;

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

typedef struct rust_star_output_hc_weights_args {
    float post_scale;
    float eps;
} rust_star_output_hc_weights_args;

typedef struct rust_star_hc_weighted_sum_norm_args {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float norm_eps;
} rust_star_hc_weighted_sum_norm_args;

typedef struct rust_star_hc_ingress_args {
    int64_t n_embd;
    int32_t n_hc, sinkhorn_iters;
    int64_t n_rows, mix_hc;
    uint64_t nb_mix1, nb_split1;
    uint64_t nb_x0, nb_x1, nb_x2;
    uint64_t nb0, nb1, nb_norm1;
    float eps, norm_eps;
} rust_star_hc_ingress_args;

typedef struct rust_star_unary_args {
    int32_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne0, ne1, ne2, ne3;
    uint64_t nb0, nb1, nb2, nb3;
    float slope, scale, bias, val, min, max;
} rust_star_unary_args;

typedef struct rust_star_router_select_one_args {
    uint32_t has_bias, hash_mode, use_token_buffer, token, hash_rows;
} rust_star_router_select_one_args;

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

typedef struct rust_star_compressor_store_args {
    uint32_t width, ratio, pos, ape_type;
} rust_star_compressor_store_args;

typedef struct rust_star_compressor_pack_args {
    uint32_t width, head_dim, ratio;
} rust_star_compressor_pack_args;

typedef struct rust_star_compressor_score_batch_args {
    uint32_t width, ratio, pos0, n_tokens;
} rust_star_compressor_score_batch_args;

typedef struct rust_star_compressor_pack_batch_args {
    uint32_t head_dim, n_comp, replay, n_threads;
} rust_star_compressor_pack_batch_args;

typedef struct rust_star_compressor_pool_batch_args {
    uint32_t head_dim, n_comp;
} rust_star_compressor_pool_batch_args;

typedef struct rust_star_softmax_args {
    int32_t ne00, ne01, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne11, ne12, ne13;
    uint64_t nb11, nb12, nb13;
    uint64_t nb1, nb2, nb3;
    float scale, max_bias, m0, m1;
    int32_t n_head_log2;
} rust_star_softmax_args;

typedef struct rust_star_sum_rows_args {
    int64_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int64_t ne0, ne1, ne2, ne3;
    uint64_t nb0, nb1, nb2, nb3;
} rust_star_sum_rows_args;

typedef struct rust_star_fp8_quantize_args {
    int64_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    uint64_t nb0, nb1, nb2, nb3;
    int32_t n_rot;
} rust_star_fp8_quantize_args;

typedef struct rust_star_indexer_qat_args {
    uint32_t n_rows, head_dim;
    uint64_t row_stride;
} rust_star_indexer_qat_args;

typedef struct rust_star_ratio4_shift_args {
    uint32_t width;
} rust_star_ratio4_shift_args;

typedef struct rust_star_flash_pad_args {
    int32_t ne11, ne_12_2, ne_12_3;
    uint64_t nb11, nb12, nb13;
    uint64_t nb21, nb22, nb23;
    int32_t ne31, ne32, ne33;
    uint64_t nb31, nb32, nb33;
} rust_star_flash_pad_args;

typedef struct rust_star_flash_blk_args {
    int32_t ne01, ne30, ne31, ne32, ne33;
    uint64_t nb31, nb32, nb33;
} rust_star_flash_blk_args;

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

typedef struct rust_star_moe_swiglu_weight_args {
    uint32_t width, rows;
    uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride;
    uint32_t write_clamped;
    float clamp_value;
} rust_star_moe_swiglu_weight_args;

typedef struct rust_star_glu_args {
    int32_t ne00;
    uint64_t nb01;
    int32_t ne10;
    uint64_t nb11;
    int32_t ne0;
    uint64_t nb1;
    int32_t i00, i10;
    float alpha, limit;
} rust_star_glu_args;

typedef struct rust_star_moe_sum_args {
    uint32_t width, tokens;
    uint64_t src_token_stride, dst_token_stride;
} rust_star_moe_sum_args;

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
    if (!context.modelMapping) {
        context.modelMapping = model_mapping;
        context.modelBytes = model_bytes;
    } else if (context.modelMapping != model_mapping || context.modelBytes != model_bytes) {
        fail_with_message(error, error_bytes, @"Metal context cannot be rebound to a different model mapping");
        return nil;
    }
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
    NSString *key = [NSString stringWithFormat:@"%llu:%llu", page_offset, buffer_bytes];
    id<MTLBuffer> buffer = context.modelViewCache[key];
    if (!buffer) {
        buffer = [context.device newBufferWithBytesNoCopy:page_pointer
                                                   length:(NSUInteger)buffer_bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        if (buffer) context.modelViewCache[key] = buffer;
    }
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

static id<MTLBuffer> persistent_buffer(
    RustStarMetalContext *context,
    NSString *key,
    NSUInteger bytes,
    char *error,
    size_t error_bytes)
{
    id<MTLBuffer> buffer = context.activationBufferCache[key];
    if (buffer) {
        if (buffer.length != bytes) {
            fail_with_message(error, error_bytes, [NSString stringWithFormat:
                @"persistent Metal buffer %@ has %llu bytes, requested %llu",
                key, (unsigned long long)buffer.length,
                (unsigned long long)bytes]);
            return nil;
        }
        return buffer;
    }
    buffer = [context.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) {
        fail_with_message(error, error_bytes, @"failed to allocate persistent Metal buffer");
        return nil;
    }
    context.activationBufferCache[key] = buffer;
    return buffer;
}

static NSString *layer_buffer_key(NSString *base, BOOL layer_scoped, uint32_t layer_index) {
    return layer_scoped
        ? [NSString stringWithFormat:@"%@_layer_%u", base, layer_index]
        : base;
}

static int encode_projected_compressor_step(
    RustStarMetalContext *context,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> projected_kv, NSUInteger projected_kv_offset,
    id<MTLBuffer> projected_score, NSUInteger projected_score_offset,
    id<MTLBuffer> ape, NSUInteger ape_offset,
    id<MTLBuffer> norm_weight, NSUInteger norm_weight_offset,
    id<MTLBuffer> state_kv,
    id<MTLBuffer> state_score,
    id<MTLBuffer> packed_kv,
    id<MTLBuffer> packed_score,
    id<MTLBuffer> softmax,
    id<MTLBuffer> output, NSUInteger output_offset,
    uint32_t width,
    uint32_t head_dim,
    uint32_t ratio,
    uint32_t position,
    BOOL emit,
    BOOL indexer)
{
    rust_star_compressor_store_args store = {
        .width=width, .ratio=ratio, .pos=position, .ape_type=1,
    };
    [encoder setComputePipelineState:context.compressorStorePipeline];
    [encoder setBytes:&store length:sizeof(store) atIndex:0];
    [encoder setBuffer:projected_kv offset:projected_kv_offset atIndex:1];
    [encoder setBuffer:projected_score offset:projected_score_offset atIndex:2];
    [encoder setBuffer:ape offset:ape_offset atIndex:3];
    [encoder setBuffer:state_kv offset:0 atIndex:4];
    [encoder setBuffer:state_score offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake((width+255u)/256u,1,1)
         threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    if (!emit) return 1;

    const uint32_t pool_rows = ratio == 4u ? 8u : ratio;
    rust_star_compressor_pack_args pack = {
        .width=width, .head_dim=head_dim, .ratio=ratio,
    };
    const uint32_t packed_elements = pool_rows*head_dim;
    [encoder setComputePipelineState:context.compressorPackPipeline];
    [encoder setBytes:&pack length:sizeof(pack) atIndex:0];
    [encoder setBuffer:state_kv offset:0 atIndex:1];
    [encoder setBuffer:state_score offset:0 atIndex:2];
    [encoder setBuffer:packed_kv offset:0 atIndex:3];
    [encoder setBuffer:packed_score offset:0 atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(packed_elements,1,1)
          threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    const uint64_t softmax_row_bytes = (uint64_t)pool_rows*sizeof(float);
    rust_star_softmax_args softmax_args = {
        .ne00=(int32_t)pool_rows, .ne01=(int32_t)head_dim, .ne02=1,
        .nb01=softmax_row_bytes,
        .nb02=(uint64_t)head_dim*softmax_row_bytes,
        .nb03=(uint64_t)head_dim*softmax_row_bytes,
        .ne11=(int32_t)pool_rows, .ne12=(int32_t)head_dim, .ne13=1,
        .nb11=softmax_row_bytes,
        .nb12=(uint64_t)head_dim*softmax_row_bytes,
        .nb13=(uint64_t)head_dim*softmax_row_bytes,
        .nb1=softmax_row_bytes,
        .nb2=(uint64_t)head_dim*softmax_row_bytes,
        .nb3=(uint64_t)head_dim*softmax_row_bytes,
        .scale=1.0f, .max_bias=0.0f, .m0=0.0f, .m1=0.0f,
        .n_head_log2=1,
    };
    [encoder setComputePipelineState:context.compressorSoftmaxPipeline];
    [encoder setBytes:&softmax_args length:sizeof(softmax_args) atIndex:0];
    [encoder setBuffer:packed_score offset:0 atIndex:1];
    [encoder setBuffer:packed_score offset:0 atIndex:2];
    [encoder setBuffer:packed_score offset:0 atIndex:3];
    [encoder setBuffer:softmax offset:0 atIndex:4];
    [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(head_dim,1,1)
         threadsPerThreadgroup:MTLSizeMake(32,1,1)];

    [encoder setComputePipelineState:context.compressorMulPipeline];
    [encoder setBuffer:packed_kv offset:0 atIndex:0];
    [encoder setBuffer:softmax offset:0 atIndex:1];
    [encoder setBuffer:packed_kv offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(packed_elements,1,1)
          threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    rust_star_sum_rows_args sum = {
        .ne00=pool_rows, .ne01=head_dim, .ne02=1, .ne03=1,
        .nb00=sizeof(float), .nb01=softmax_row_bytes,
        .nb02=(uint64_t)head_dim*softmax_row_bytes,
        .nb03=(uint64_t)head_dim*softmax_row_bytes,
        .ne0=1, .ne1=head_dim, .ne2=1, .ne3=1,
        .nb0=sizeof(float), .nb1=sizeof(float),
        .nb2=(uint64_t)head_dim*sizeof(float),
        .nb3=(uint64_t)head_dim*sizeof(float),
    };
    [encoder setComputePipelineState:context.compressorSumRowsPipeline];
    [encoder setBytes:&sum length:sizeof(sum) atIndex:0];
    [encoder setBuffer:packed_kv offset:0 atIndex:1];
    [encoder setBuffer:output offset:output_offset atIndex:2];
    [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(head_dim,1,1)
         threadsPerThreadgroup:MTLSizeMake(pool_rows,1,1)];

    const uint64_t output_bytes = (uint64_t)head_dim*sizeof(float);
    rust_star_norm_args norm = {
        .ne00=(int32_t)head_dim, .ne00_t=(int32_t)(head_dim/4u),
        .nb1=output_bytes, .nb2=output_bytes, .nb3=output_bytes,
        .eps=1.0e-6f,
        .nef1={1,1,1}, .nef2={1,1,1}, .nef3={1,1,1},
        .nbf1={output_bytes,output_bytes,output_bytes},
        .nbf2={output_bytes,output_bytes,output_bytes},
        .nbf3={output_bytes,output_bytes,output_bytes},
    };
    [encoder setComputePipelineState:context.compressorNormPipeline];
    [encoder setBytes:&norm length:sizeof(norm) atIndex:0];
    [encoder setBuffer:output offset:output_offset atIndex:1];
    [encoder setBuffer:norm_weight offset:norm_weight_offset atIndex:2];
    [encoder setBuffer:output offset:output_offset atIndex:3];
    [encoder setBuffer:output offset:output_offset atIndex:4];
    [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
    const NSUInteger norm_threads = head_dim == 512u ? 128u : 32u;
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
         threadsPerThreadgroup:MTLSizeMake(norm_threads,1,1)];

    rust_star_rope_tail_args rope = {
        .ne00=(int32_t)head_dim, .ne01=1, .ne02=1, .ne03=1,
        .nb00=sizeof(float), .nb01=output_bytes,
        .nb02=output_bytes, .nb03=output_bytes,
        .nb0=sizeof(float), .nb1=output_bytes,
        .nb2=output_bytes, .nb3=output_bytes,
        .n_dims=64, .mode=0, .n_ctx_orig=65536, .inverse=0,
        .freq_base=160000.0f, .freq_scale=1.0f/16.0f,
        .ext_factor=1.0f,
        .attn_factor=1.0f/(1.0f+0.1f*logf(16.0f)),
        .beta_fast=32.0f, .beta_slow=1.0f, .src2=false,
    };
    // The emitted row represents the first token in this compression window.
    // Position 3 therefore uses 0, position 7 uses 4, and so on.
    int32_t compressed_position = (int32_t)(position + 1u - ratio);
    [encoder setComputePipelineState:context.ropeTailPipeline];
    [encoder setBytes:&rope length:sizeof(rope) atIndex:0];
    [encoder setBuffer:output offset:output_offset atIndex:1];
    [encoder setBytes:&compressed_position length:sizeof(compressed_position) atIndex:2];
    [encoder setBuffer:output offset:output_offset atIndex:3];
    [encoder setBuffer:output offset:output_offset atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
         threadsPerThreadgroup:MTLSizeMake(MIN((uint32_t)256u,head_dim),1,1)];

    if (ratio == 4u) {
        rust_star_ratio4_shift_args shift = { .width=width };
        [encoder setComputePipelineState:context.compressorShiftPipeline];
        [encoder setBytes:&shift length:sizeof(shift) atIndex:0];
        [encoder setBuffer:state_kv offset:0 atIndex:1];
        [encoder setBuffer:state_score offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(4u*width,1,1)
              threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }

    if (indexer) {
        rust_star_indexer_qat_args qat = {
            .n_rows=1, .head_dim=head_dim, .row_stride=output_bytes,
        };
        [encoder setComputePipelineState:context.indexerQatPipeline];
        [encoder setBytes:&qat length:sizeof(qat) atIndex:0];
        [encoder setBuffer:output offset:output_offset atIndex:1];
        [encoder setThreadgroupMemoryLength:256u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    } else {
        rust_star_fp8_quantize_args fp8 = {
            .ne00=head_dim, .ne01=1, .ne02=1, .ne03=1,
            .nb00=sizeof(float), .nb01=output_bytes,
            .nb02=output_bytes, .nb03=output_bytes,
            .nb0=sizeof(float), .nb1=output_bytes,
            .nb2=output_bytes, .nb3=output_bytes, .n_rot=64,
        };
        [encoder setComputePipelineState:context.compressorFp8Pipeline];
        [encoder setBytes:&fp8 length:sizeof(fp8) atIndex:0];
        [encoder setBuffer:output offset:output_offset atIndex:1];
        [encoder setBuffer:output offset:output_offset atIndex:2];
        [encoder setThreadgroupMemoryLength:64u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    }
    return 1;
}

static int encode_compressor_step(
    RustStarMetalContext *context,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> activation, NSUInteger activation_offset,
    id<MTLBuffer> kv_weight, NSUInteger kv_weight_offset,
    id<MTLBuffer> gate_weight, NSUInteger gate_weight_offset,
    id<MTLBuffer> ape, NSUInteger ape_offset,
    id<MTLBuffer> norm_weight, NSUInteger norm_weight_offset,
    id<MTLBuffer> projected_kv,
    id<MTLBuffer> projected_score,
    id<MTLBuffer> state_kv,
    id<MTLBuffer> state_score,
    id<MTLBuffer> packed_kv,
    id<MTLBuffer> packed_score,
    id<MTLBuffer> softmax,
    id<MTLBuffer> output,
    uint32_t width,
    uint32_t head_dim,
    uint32_t ratio,
    uint32_t position,
    BOOL emit,
    BOOL indexer)
{
    rust_star_q8_mv_args projection = {
        .ne00=4096, .ne01=(int32_t)width, .ne02=1,
        .nb00=sizeof(uint16_t), .nb01=4096ull*sizeof(uint16_t),
        .nb02=4096ull*width*sizeof(uint16_t),
        .nb03=4096ull*width*sizeof(uint16_t),
        .ne10=4096, .ne11=1, .ne12=1,
        .nb10=sizeof(float), .nb11=4096ull*sizeof(float),
        .nb12=4096ull*sizeof(float), .nb13=4096ull*sizeof(float),
        .ne0=(int32_t)width, .ne1=1, .nr0=2, .r2=1, .r3=1,
    };
    [encoder setComputePipelineState:context.compressorPairPipeline];
    [encoder setBytes:&projection length:sizeof(projection) atIndex:0];
    [encoder setBuffer:kv_weight offset:kv_weight_offset atIndex:1];
    [encoder setBuffer:gate_weight offset:gate_weight_offset atIndex:2];
    [encoder setBuffer:activation offset:activation_offset atIndex:3];
    [encoder setBuffer:projected_kv offset:0 atIndex:4];
    [encoder setBuffer:projected_score offset:0 atIndex:5];
    [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((width+1u)/2u,1,1)
         threadsPerThreadgroup:MTLSizeMake(32,8,1)];

    return encode_projected_compressor_step(
        context, encoder,
        projected_kv, 0, projected_score, 0,
        ape, ape_offset, norm_weight, norm_weight_offset,
        state_kv, state_score, packed_kv, packed_score, softmax,
        output, 0, width, head_dim, ratio, position, emit, indexer);
}

static int encode_projected_compressor_batch_ratio4(
    RustStarMetalContext *context,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> activation, NSUInteger activation_offset,
    id<MTLBuffer> kv_weight, NSUInteger kv_weight_offset,
    id<MTLBuffer> gate_weight, NSUInteger gate_weight_offset,
    id<MTLBuffer> projected_kv,
    id<MTLBuffer> projected_score,
    id<MTLBuffer> score_with_ape,
    id<MTLBuffer> packed_kv,
    id<MTLBuffer> packed_score,
    id<MTLBuffer> ape, NSUInteger ape_offset,
    id<MTLBuffer> norm_weight, NSUInteger norm_weight_offset,
    id<MTLBuffer> state_kv,
    id<MTLBuffer> state_score,
    id<MTLBuffer> output, NSUInteger output_offset,
    id<MTLBuffer> compressed_positions,
    uint32_t width,
    uint32_t head_dim,
    uint32_t position_start,
    uint32_t rows,
    BOOL indexer)
{
    const uint32_t ratio = 4u;
    const uint32_t n_comp = rows/ratio;
    rust_star_compressor_score_batch_args score_args = {
        .width=width, .ratio=ratio, .pos0=position_start, .n_tokens=rows,
    };
    [encoder setComputePipelineState:context.compressorScoreBatchPipeline];
    [encoder setBytes:&score_args length:sizeof(score_args) atIndex:0];
    [encoder setBuffer:projected_score offset:0 atIndex:1];
    [encoder setBuffer:ape offset:ape_offset atIndex:2];
    [encoder setBuffer:score_with_ape offset:0 atIndex:3];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)rows*width,1,1)
          threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    const uint32_t pack_threads = MIN(head_dim, 256u);
    rust_star_compressor_pack_batch_args pack_args = {
        .head_dim=head_dim, .n_comp=n_comp,
        .replay=position_start == 0u ? 0u : 1u,
        .n_threads=pack_threads,
    };
    [encoder setComputePipelineState:context.compressorPackBatchPipeline];
    [encoder setBytes:&pack_args length:sizeof(pack_args) atIndex:0];
    [encoder setBuffer:projected_kv offset:0 atIndex:1];
    [encoder setBuffer:score_with_ape offset:0 atIndex:2];
    [encoder setBuffer:state_kv offset:0 atIndex:3];
    [encoder setBuffer:state_score offset:0 atIndex:4];
    [encoder setBuffer:packed_kv offset:0 atIndex:5];
    [encoder setBuffer:packed_score offset:0 atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(n_comp,8u,1u)
         threadsPerThreadgroup:MTLSizeMake(pack_threads,1,1)];

    rust_star_compressor_pool_batch_args pool_args = {
        .head_dim=head_dim, .n_comp=n_comp,
    };
    [encoder setComputePipelineState:context.compressorPoolBatchPipeline];
    [encoder setBytes:&pool_args length:sizeof(pool_args) atIndex:0];
    [encoder setBuffer:packed_kv offset:0 atIndex:1];
    [encoder setBuffer:packed_score offset:0 atIndex:2];
    [encoder setBuffer:output offset:output_offset atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(
            ((NSUInteger)n_comp*head_dim+255u)/256u,1,1)
         threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    const uint64_t row_bytes = (uint64_t)head_dim*sizeof(float);
    const uint64_t tile_bytes = (uint64_t)n_comp*row_bytes;
    rust_star_norm_args norm = {
        .ne00=(int32_t)head_dim, .ne00_t=(int32_t)(head_dim/4u),
        .nb1=row_bytes, .nb2=tile_bytes, .nb3=tile_bytes, .eps=1.0e-6f,
        .nef1={(int32_t)n_comp,1,1}, .nef2={1,1,1}, .nef3={1,1,1},
        .nbf1={row_bytes,row_bytes,row_bytes},
        .nbf2={tile_bytes,row_bytes,row_bytes},
        .nbf3={tile_bytes,row_bytes,row_bytes},
    };
    [encoder setComputePipelineState:context.compressorNormPipeline];
    [encoder setBytes:&norm length:sizeof(norm) atIndex:0];
    [encoder setBuffer:output offset:output_offset atIndex:1];
    [encoder setBuffer:norm_weight offset:norm_weight_offset atIndex:2];
    [encoder setBuffer:output offset:output_offset atIndex:3];
    [encoder setBuffer:output offset:output_offset atIndex:4];
    [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(n_comp,1,1)
         threadsPerThreadgroup:MTLSizeMake(head_dim == 512u ? 128u : 32u,1,1)];

    rust_star_rope_tail_args rope = {
        .ne00=(int32_t)head_dim, .ne01=1, .ne02=n_comp, .ne03=1,
        .nb00=sizeof(float), .nb01=row_bytes, .nb02=row_bytes, .nb03=tile_bytes,
        .nb0=sizeof(float), .nb1=row_bytes, .nb2=row_bytes, .nb3=tile_bytes,
        .n_dims=64, .mode=0, .n_ctx_orig=65536, .inverse=0,
        .freq_base=160000.0f, .freq_scale=1.0f/16.0f, .ext_factor=1.0f,
        .attn_factor=1.0f/(1.0f+0.1f*logf(16.0f)),
        .beta_fast=32.0f, .beta_slow=1.0f, .src2=false,
    };
    [encoder setComputePipelineState:context.ropeTailPipeline];
    [encoder setBytes:&rope length:sizeof(rope) atIndex:0];
    [encoder setBuffer:output offset:output_offset atIndex:1];
    [encoder setBuffer:compressed_positions offset:0 atIndex:2];
    [encoder setBuffer:output offset:output_offset atIndex:3];
    [encoder setBuffer:output offset:output_offset atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(1,n_comp,1)
         threadsPerThreadgroup:MTLSizeMake(MIN(256u,head_dim),1,1)];

    if (indexer) {
        rust_star_indexer_qat_args qat = {
            .n_rows=n_comp, .head_dim=head_dim, .row_stride=row_bytes,
        };
        [encoder setComputePipelineState:context.indexerQatPipeline];
        [encoder setBytes:&qat length:sizeof(qat) atIndex:0];
        [encoder setBuffer:output offset:output_offset atIndex:1];
        [encoder setThreadgroupMemoryLength:256u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(n_comp,1,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    } else {
        rust_star_fp8_quantize_args fp8 = {
            .ne00=head_dim, .ne01=1, .ne02=n_comp, .ne03=1,
            .nb00=sizeof(float), .nb01=row_bytes, .nb02=row_bytes,
            .nb03=tile_bytes, .nb0=sizeof(float), .nb1=row_bytes,
            .nb2=row_bytes, .nb3=tile_bytes, .n_rot=64,
        };
        [encoder setComputePipelineState:context.compressorFp8Pipeline];
        [encoder setBytes:&fp8 length:sizeof(fp8) atIndex:0];
        [encoder setBuffer:output offset:output_offset atIndex:1];
        [encoder setBuffer:output offset:output_offset atIndex:2];
        [encoder setThreadgroupMemoryLength:64u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(n_comp,1,1)
             threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    }

    const BOOL final_prefill_tile = position_start + rows == 2048u;
    rust_star_mul_mv_ext_args tail_projection = {
        .ne00=4096, .ne01=(int32_t)width, .ne02=1,
        .nb00=sizeof(uint16_t), .nb01=4096ull*sizeof(uint16_t),
        .nb02=4096ull*width*sizeof(uint16_t),
        .nb03=4096ull*width*sizeof(uint16_t),
        .ne10=4096, .ne11=ratio, .ne12=1,
        .nb10=sizeof(float), .nb11=4096ull*sizeof(float),
        .nb12=4096ull*ratio*sizeof(float),
        .nb13=4096ull*ratio*sizeof(float),
        .ne0=(int32_t)width, .ne1=ratio, .r2=1, .r3=1,
    };
#define RUST_STAR_ENCODE_TAIL_PROJECTION(weight, weight_offset, output) do { \
        [encoder setComputePipelineState:context.f16Tail4Pipeline]; \
        [encoder setBytes:&tail_projection length:sizeof(tail_projection) atIndex:0]; \
        [encoder setBuffer:(weight) offset:(weight_offset) atIndex:1]; \
        [encoder setBuffer:activation offset:activation_offset atIndex:2]; \
        [encoder setBuffer:(output) offset:0 atIndex:3]; \
        [encoder dispatchThreadgroups:MTLSizeMake((width+7u)/8u,1,1) \
             threadsPerThreadgroup:MTLSizeMake(32,2,1)]; \
    } while (0)
    if (final_prefill_tile) {
        RUST_STAR_ENCODE_TAIL_PROJECTION(kv_weight, kv_weight_offset, projected_kv);
        RUST_STAR_ENCODE_TAIL_PROJECTION(gate_weight, gate_weight_offset, projected_score);
    }
#undef RUST_STAR_ENCODE_TAIL_PROJECTION

    for (uint32_t row = 0; row < ratio; row++) {
        const uint32_t projected_row =
            final_prefill_tile ? row : rows-ratio+row;
        rust_star_compressor_store_args store = {
            .width=width, .ratio=ratio,
            .pos=position_start+rows-ratio+row, .ape_type=1,
        };
        [encoder setComputePipelineState:context.compressorStorePipeline];
        [encoder setBytes:&store length:sizeof(store) atIndex:0];
        [encoder setBuffer:projected_kv
                    offset:(NSUInteger)projected_row*width*sizeof(float) atIndex:1];
        [encoder setBuffer:projected_score
                    offset:(NSUInteger)projected_row*width*sizeof(float) atIndex:2];
        [encoder setBuffer:ape offset:ape_offset atIndex:3];
        [encoder setBuffer:state_kv offset:0 atIndex:4];
        [encoder setBuffer:state_score offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((width+255u)/256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    rust_star_ratio4_shift_args shift = { .width=width };
    [encoder setComputePipelineState:context.compressorShiftPipeline];
    [encoder setBytes:&shift length:sizeof(shift) atIndex:0];
    [encoder setBuffer:state_kv offset:0 atIndex:1];
    [encoder setBuffer:state_score offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(4u*width,1,1)
          threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return 1;
}

int rust_star_metal_run_ratio128_compressor_replay(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint32_t layer_index,
    uint64_t ape_offset,
    uint64_t ape_bytes,
    uint64_t kv_offset,
    uint64_t kv_bytes,
    uint64_t gate_offset,
    uint64_t gate_bytes,
    uint64_t norm_offset,
    uint64_t norm_bytes,
    const float *activation_sequence,
    uint64_t activation_elements,
    float *output,
    uint64_t output_elements,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes)
{
    const uint32_t positions = 128u;
    const uint32_t activation_width = 4096u;
    const uint32_t compressor_width = 512u;
    const uint32_t head_dim = 512u;
    const uint32_t ratio = 128u;
    if (!opaque_context || !model_mapping || !activation_sequence || !output || !result) {
        return fail_with_message(error, error_bytes,
            @"ratio-128 compressor replay received a null input");
    }
    if (layer_index < 3u || (layer_index & 1u) == 0u ||
        activation_elements != (uint64_t)positions*activation_width ||
        output_elements != head_dim ||
        ape_bytes != (uint64_t)compressor_width*ratio*sizeof(uint16_t) ||
        kv_bytes != (uint64_t)activation_width*compressor_width*sizeof(uint16_t) ||
        gate_bytes != (uint64_t)activation_width*compressor_width*sizeof(uint16_t) ||
        norm_bytes != (uint64_t)head_dim*sizeof(float)) {
        return fail_with_message(error, error_bytes,
            @"ratio-128 compressor replay dimensions are invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_attention_ingress_pipelines(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));
        memset(output, 0, output_elements*sizeof(float));

        NSUInteger ape_inner = 0, kv_inner = 0, gate_inner = 0, norm_inner = 0;
        BOOL ape_match = NO, kv_match = NO, gate_match = NO, norm_match = NO;
        id<MTLBuffer> ape = wrap_model_range(context, model_mapping, model_bytes,
            ape_offset, ape_bytes, &ape_inner, &ape_match, error, error_bytes);
        id<MTLBuffer> kv_weight = wrap_model_range(context, model_mapping, model_bytes,
            kv_offset, kv_bytes, &kv_inner, &kv_match, error, error_bytes);
        id<MTLBuffer> gate_weight = wrap_model_range(context, model_mapping, model_bytes,
            gate_offset, gate_bytes, &gate_inner, &gate_match, error, error_bytes);
        id<MTLBuffer> norm_weight = wrap_model_range(context, model_mapping, model_bytes,
            norm_offset, norm_bytes, &norm_inner, &norm_match, error, error_bytes);
        if (!ape || !kv_weight || !gate_weight || !norm_weight) return 0;

        const NSUInteger activation_bytes = (NSUInteger)activation_elements*sizeof(float);
        const NSUInteger row_bytes = compressor_width*sizeof(float);
        const NSUInteger state_bytes = (NSUInteger)ratio*row_bytes;
        const NSUInteger packed_bytes = (NSUInteger)ratio*head_dim*sizeof(float);
        id<MTLBuffer> activations = [context.device newBufferWithBytes:activation_sequence
            length:activation_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> projected_kv = [context.device newBufferWithLength:row_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> projected_score = [context.device newBufferWithLength:row_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> state_kv = [context.device newBufferWithLength:state_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> state_score = [context.device newBufferWithLength:state_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> packed_kv = [context.device newBufferWithLength:packed_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> packed_score = [context.device newBufferWithLength:packed_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> softmax = [context.device newBufferWithLength:packed_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> compressed = [context.device newBufferWithLength:head_dim*sizeof(float)
            options:MTLResourceStorageModeShared];
        if (!activations || !projected_kv || !projected_score || !state_kv ||
            !state_score || !packed_kv || !packed_score || !softmax || !compressed) {
            return fail_with_message(error, error_bytes,
                @"failed to allocate ratio-128 compressor replay buffers");
        }
        memset(state_kv.contents, 0, state_bytes);
        float *scores = state_score.contents;
        for (uint32_t index = 0; index < ratio*compressor_width; index++) {
            scores[index] = -INFINITY;
        }

        const double wall_start = monotonic_ms();
        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            return fail_with_message(error, error_bytes,
                @"failed to create ratio-128 compressor replay command buffer");
        }
        for (uint32_t position = 0; position < positions; position++) {
            const NSUInteger activation_offset =
                (NSUInteger)position*activation_width*sizeof(float);
            if (!encode_compressor_step(context, encoder,
                    activations, activation_offset,
                    kv_weight, kv_inner,
                    gate_weight, gate_inner,
                    ape, ape_inner,
                    norm_weight, norm_inner,
                    projected_kv, projected_score,
                    state_kv, state_score,
                    packed_kv, packed_score, softmax, compressed,
                    compressor_width, head_dim, ratio, position,
                    position == positions-1u, NO)) {
                return fail_with_message(error, error_bytes,
                    @"failed to encode ratio-128 compressor replay step");
            }
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return fail_with_message(error, error_bytes,
                command.error.localizedDescription ?: @"ratio-128 compressor replay failed");
        }
        memcpy(output, compressed.contents, head_dim*sizeof(float));

        result->model_bytes = model_bytes;
        result->max_buffer_length = context.device.maxBufferLength;
        result->wrapped_model_ranges = 4u;
        result->pointer_matches = (ape_match ? 1u : 0u) + (kv_match ? 1u : 0u) +
            (gate_match ? 1u : 0u) + (norm_match ? 1u : 0u);
        result->wall_ms = monotonic_ms()-wall_start;
        result->gpu_ms = gpu_elapsed_ms(command);
        return 1;
    }
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

int rust_star_metal_run_prefill_q8_boundary(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    uint32_t input_elements_per_row,
    uint32_t output_elements_per_row,
    uint32_t rows,
    const float *input,
    float *batch_output,
    float *decode_output,
    rust_star_metal_prefill_q8_probe_result *result,
    char *error,
    size_t error_bytes)
{
    if (!opaque_context || !model_mapping || !input || !batch_output ||
        !decode_output || !result) {
        return fail_with_message(error, error_bytes, @"prefill Q8 boundary received a null input");
    }
    if (rows != 128u || input_elements_per_row == 0u ||
        (input_elements_per_row % 64u) != 0u || output_elements_per_row == 0u ||
        (output_elements_per_row % 64u) != 0u) {
        return fail_with_message(error, error_bytes, @"prefill Q8 boundary dimensions are invalid");
    }
    const uint64_t row_bytes = (uint64_t)(input_elements_per_row / 32u) * 34u;
    if (tensor_bytes != row_bytes * (uint64_t)output_elements_per_row ||
        tensor_offset > model_bytes || tensor_bytes > model_bytes - tensor_offset) {
        return fail_with_message(error, error_bytes, @"prefill Q8 tensor range is invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_q8_projection_pipeline(context, error, error_bytes) ||
            !ensure_moe_output_pipelines(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));

        NSUInteger weight_inner = 0;
        BOOL pointer_match = NO;
        id<MTLBuffer> weights = wrap_model_range(
            context, model_mapping, model_bytes, tensor_offset, tensor_bytes,
            &weight_inner, &pointer_match, error, error_bytes);
        if (!weights) return 0;

        const NSUInteger input_bytes =
            (NSUInteger)rows * (NSUInteger)input_elements_per_row * sizeof(float);
        const NSUInteger batch_output_bytes =
            (NSUInteger)rows * (NSUInteger)output_elements_per_row * sizeof(float);
        const NSUInteger decode_output_bytes =
            (NSUInteger)output_elements_per_row * sizeof(float);
        id<MTLBuffer> input_buffer =
            [context.device newBufferWithBytes:input length:input_bytes
                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> batch_output_buffer =
            [context.device newBufferWithLength:batch_output_bytes
                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> decode_output_buffer =
            [context.device newBufferWithLength:decode_output_bytes
                                        options:MTLResourceStorageModeShared];
        if (!input_buffer || !batch_output_buffer || !decode_output_buffer) {
            return fail_with_message(error, error_bytes,
                @"failed to allocate prefill Q8 boundary activation buffers");
        }
        rust_star_q8_mm_args mm_args = {
            .ne00 = (int32_t)input_elements_per_row,
            .ne02 = 1,
            .nb01 = row_bytes,
            .nb02 = row_bytes * output_elements_per_row,
            .nb03 = row_bytes * output_elements_per_row,
            .ne12 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)input_elements_per_row * sizeof(float),
            .nb12 = (uint64_t)input_elements_per_row * rows * sizeof(float),
            .nb13 = (uint64_t)input_elements_per_row * rows * sizeof(float),
            .ne0 = (int32_t)output_elements_per_row,
            .ne1 = (int32_t)rows,
            .r2 = 1,
            .r3 = 1,
        };
        id<MTLCommandBuffer> batch_command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> batch_encoder = [batch_command computeCommandEncoder];
        if (!batch_command || !batch_encoder) {
            return fail_with_message(error, error_bytes,
                @"failed to create prefill Q8 batch command encoder");
        }
        [batch_encoder setComputePipelineState:context.q8PrefillPipeline];
        [batch_encoder setBytes:&mm_args length:sizeof(mm_args) atIndex:0];
        [batch_encoder setBuffer:weights offset:weight_inner atIndex:1];
        [batch_encoder setBuffer:input_buffer offset:0 atIndex:2];
        [batch_encoder setBuffer:batch_output_buffer offset:0 atIndex:3];
        [batch_encoder setThreadgroupMemoryLength:6144u atIndex:0];
        [batch_encoder dispatchThreadgroups:
            MTLSizeMake(((NSUInteger)rows + 31u) / 32u,
                        (NSUInteger)output_elements_per_row / 64u, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [batch_encoder endEncoding];
        const double batch_wall_start = monotonic_ms();
        [batch_command commit];
        if (!command_succeeded(batch_command, error, error_bytes)) return 0;
        const double batch_wall_end = monotonic_ms();

        rust_star_q8_mv_args mv_args = {
            .ne00 = (int32_t)input_elements_per_row,
            .ne01 = (int32_t)output_elements_per_row,
            .ne02 = 1,
            .nb00 = 34,
            .nb01 = row_bytes,
            .nb02 = row_bytes * output_elements_per_row,
            .nb03 = row_bytes * output_elements_per_row,
            .ne10 = (int32_t)input_elements_per_row,
            .ne11 = 1,
            .ne12 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)input_elements_per_row * sizeof(float),
            .nb12 = (uint64_t)input_elements_per_row * sizeof(float),
            .nb13 = (uint64_t)input_elements_per_row * sizeof(float),
            .ne0 = (int32_t)output_elements_per_row,
            .ne1 = 1,
            .nr0 = 2,
            .r2 = 1,
            .r3 = 1,
        };
        id<MTLCommandBuffer> decode_command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> decode_encoder = [decode_command computeCommandEncoder];
        if (!decode_command || !decode_encoder) {
            return fail_with_message(error, error_bytes,
                @"failed to create prefill Q8 decode-control command encoder");
        }
        [decode_encoder setComputePipelineState:context.q8ProjectionPipeline];
        [decode_encoder setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [decode_encoder setBuffer:weights offset:weight_inner atIndex:1];
        [decode_encoder setBuffer:input_buffer
            offset:(NSUInteger)(rows - 1u) * input_elements_per_row * sizeof(float) atIndex:2];
        [decode_encoder setBuffer:decode_output_buffer offset:0 atIndex:3];
        [decode_encoder setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
        [decode_encoder dispatchThreadgroups:
            MTLSizeMake(((NSUInteger)output_elements_per_row + 1u) / 2u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        [decode_encoder endEncoding];
        const double decode_wall_start = monotonic_ms();
        [decode_command commit];
        if (!command_succeeded(decode_command, error, error_bytes)) return 0;
        const double decode_wall_end = monotonic_ms();

        memcpy(batch_output, batch_output_buffer.contents, batch_output_bytes);
        memcpy(decode_output, decode_output_buffer.contents, decode_output_bytes);
        result->model_bytes = model_bytes;
        result->tensor_offset = tensor_offset;
        result->tensor_bytes = tensor_bytes;
        result->input_elements_per_row = input_elements_per_row;
        result->output_elements_per_row = output_elements_per_row;
        result->rows = rows;
        result->max_buffer_length = context.device.maxBufferLength;
        result->no_copy_pointer_match = pointer_match ? 1u : 0u;
        result->batch_threads_per_threadgroup = 128u;
        result->batch_threadgroups_x = (rows + 31u) / 32u;
        result->batch_threadgroups_y = output_elements_per_row / 64u;
        result->batch_wall_ms = batch_wall_end - batch_wall_start;
        result->batch_gpu_ms = gpu_elapsed_ms(batch_command);
        result->decode_wall_ms = decode_wall_end - decode_wall_start;
        result->decode_gpu_ms = gpu_elapsed_ms(decode_command);
        return 1;
    }
}

int rust_star_metal_run_prefill_qkv_boundary(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    const rust_star_metal_prefill_qkv_weights *weights,
    uint32_t rows,
    uint32_t position_start,
    const float *attn_norm,
    float *q_lora,
    float *q_lora_norm,
    float *kv_raw,
    float *kv_norm,
    float *q_raw,
    float *q_cur,
    rust_star_metal_prefill_qkv_probe_result *result,
    char *error,
    size_t error_bytes)
{
    if (!opaque_context || !model_mapping || !weights || !attn_norm ||
        !q_lora || !q_lora_norm || !kv_raw || !kv_norm || !q_raw ||
        !q_cur || !result) {
        return fail_with_message(error, error_bytes, @"prefill Q/KV boundary received a null input");
    }
    if (rows != 32u || position_start != 2016u) {
        return fail_with_message(error, error_bytes, @"prefill Q/KV boundary dimensions are invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_attention_ingress_pipelines(context, error, error_bytes) ||
            !ensure_moe_output_pipelines(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));

        const uint64_t offsets[5] = {
            weights->q_a_offset, weights->q_a_norm_offset, weights->kv_offset,
            weights->kv_norm_offset, weights->q_b_offset,
        };
        const uint64_t sizes[5] = {
            weights->q_a_bytes, weights->q_a_norm_bytes, weights->kv_bytes,
            weights->kv_norm_bytes, weights->q_b_bytes,
        };
        id<MTLBuffer> model_buffers[5] = { nil, nil, nil, nil, nil };
        NSUInteger inner[5] = { 0, 0, 0, 0, 0 };
        BOOL matches[5] = { NO, NO, NO, NO, NO };
        for (uint32_t index = 0; index < 5u; index++) {
            model_buffers[index] = wrap_model_range(
                context, model_mapping, model_bytes, offsets[index], sizes[index],
                &inner[index], &matches[index], error, error_bytes);
            if (!model_buffers[index]) return 0;
        }

        enum { n_embd = 4096, q_rank = 1024, kv_dim = 512, q_dim = 32768 };
        const NSUInteger attn_bytes = rows * n_embd * sizeof(float);
        const NSUInteger q_rank_bytes = rows * q_rank * sizeof(float);
        const NSUInteger kv_bytes = rows * kv_dim * sizeof(float);
        const NSUInteger q_bytes = rows * q_dim * sizeof(float);
        id<MTLBuffer> attn_buffer = [context.device newBufferWithBytes:attn_norm
            length:attn_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_buffer = [context.device newBufferWithLength:q_rank_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_norm_buffer = [context.device newBufferWithLength:q_rank_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_raw_buffer = [context.device newBufferWithLength:kv_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_norm_buffer = [context.device newBufferWithLength:kv_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_raw_buffer = [context.device newBufferWithLength:q_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_cur_buffer = [context.device newBufferWithLength:q_bytes
            options:MTLResourceStorageModeShared];
        if (!attn_buffer || !q_buffer || !q_norm_buffer || !kv_raw_buffer ||
            !kv_norm_buffer || !q_raw_buffer || !q_cur_buffer) {
            return fail_with_message(error, error_bytes,
                @"failed to allocate prefill Q/KV boundary activation buffers");
        }

#define RUST_STAR_Q8_MM_ARGS(in_width, out_width, weight_bytes) \
        (rust_star_q8_mm_args){ \
            .ne00=(in_width), .ne02=1, \
            .nb01=(uint64_t)((in_width)/32u)*34u, \
            .nb02=(weight_bytes), .nb03=(weight_bytes), .ne12=1, \
            .nb10=sizeof(float), .nb11=(uint64_t)(in_width)*sizeof(float), \
            .nb12=(uint64_t)(in_width)*rows*sizeof(float), \
            .nb13=(uint64_t)(in_width)*rows*sizeof(float), \
            .ne0=(out_width), .ne1=(int32_t)rows, .r2=1, .r3=1 \
        }
        rust_star_q8_mm_args q_a_args = RUST_STAR_Q8_MM_ARGS(n_embd, q_rank, weights->q_a_bytes);
        rust_star_q8_mm_args kv_args = RUST_STAR_Q8_MM_ARGS(n_embd, kv_dim, weights->kv_bytes);
        rust_star_q8_mm_args q_b_args = RUST_STAR_Q8_MM_ARGS(q_rank, q_dim, weights->q_b_bytes);
#undef RUST_STAR_Q8_MM_ARGS
        rust_star_qkv_norm_args norm_args = {
            .q_n=q_rank, .q_n4=q_rank/4, .kv_n=kv_dim, .kv_n4=kv_dim/4,
            .q_row_stride=q_rank*sizeof(float), .kv_row_stride=kv_dim*sizeof(float),
            .eps=1.0e-6f,
        };
        rust_star_head_norm_rope_args rope_args = {
            .n_head=64, .head_dim=512, .head_dim4=128, .n_dims=64,
            .n_ctx_orig=0, .pos0=(int32_t)position_start, .inverse=0,
            .eps=1.0e-6f, .freq_base=10000.0f, .freq_scale=1.0f,
            .ext_factor=0.0f, .attn_factor=1.0f,
            .beta_fast=32.0f, .beta_slow=1.0f,
        };

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            return fail_with_message(error, error_bytes,
                @"failed to create prefill Q/KV boundary command encoder");
        }
#define RUST_STAR_ENCODE_Q8(args, weight_index, input_buffer, output_buffer, out_width) do { \
        [encoder setComputePipelineState:context.q8PrefillPipeline]; \
        [encoder setBytes:&(args) length:sizeof(args) atIndex:0]; \
        [encoder setBuffer:model_buffers[(weight_index)] offset:inner[(weight_index)] atIndex:1]; \
        [encoder setBuffer:(input_buffer) offset:0 atIndex:2]; \
        [encoder setBuffer:(output_buffer) offset:0 atIndex:3]; \
        [encoder setThreadgroupMemoryLength:6144u atIndex:0]; \
        [encoder dispatchThreadgroups:MTLSizeMake(1u, (out_width)/64u, 1u) \
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)]; \
    } while (0)
        RUST_STAR_ENCODE_Q8(q_a_args, 0, attn_buffer, q_buffer, q_rank);
        RUST_STAR_ENCODE_Q8(kv_args, 2, attn_buffer, kv_raw_buffer, kv_dim);

        [encoder setComputePipelineState:context.qkvNormPipeline];
        [encoder setBytes:&norm_args length:sizeof(norm_args) atIndex:0];
        [encoder setBuffer:q_buffer offset:0 atIndex:1];
        [encoder setBuffer:model_buffers[1] offset:inner[1] atIndex:2];
        [encoder setBuffer:q_norm_buffer offset:0 atIndex:3];
        [encoder setBuffer:kv_raw_buffer offset:0 atIndex:4];
        [encoder setBuffer:model_buffers[3] offset:inner[3] atIndex:5];
        [encoder setBuffer:kv_norm_buffer offset:0 atIndex:6];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 2, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        RUST_STAR_ENCODE_Q8(q_b_args, 4, q_norm_buffer, q_raw_buffer, q_dim);
#undef RUST_STAR_ENCODE_Q8
        [encoder endEncoding];

        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (!blit) return fail_with_message(error, error_bytes,
            @"failed to create prefill Q snapshot encoder");
        [blit copyFromBuffer:q_raw_buffer sourceOffset:0 toBuffer:q_cur_buffer
            destinationOffset:0 size:q_bytes];
        [blit endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create prefill Q RoPE encoder");
        [encoder setComputePipelineState:context.headNormRopePipeline];
        [encoder setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [encoder setBuffer:q_cur_buffer offset:0 atIndex:1];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(64, rows, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        memcpy(q_lora, q_buffer.contents, q_rank_bytes);
        memcpy(q_lora_norm, q_norm_buffer.contents, q_rank_bytes);
        memcpy(kv_raw, kv_raw_buffer.contents, kv_bytes);
        memcpy(kv_norm, kv_norm_buffer.contents, kv_bytes);
        memcpy(q_raw, q_raw_buffer.contents, q_bytes);
        memcpy(q_cur, q_cur_buffer.contents, q_bytes);

        uint32_t pointer_matches = 0;
        for (uint32_t index = 0; index < 5u; index++) pointer_matches += matches[index] ? 1u : 0u;
        result->rows = rows;
        result->input_elements_per_row = n_embd;
        result->q_lora_elements_per_row = q_rank;
        result->kv_elements_per_row = kv_dim;
        result->q_elements_per_row = q_dim;
        result->dispatches = 5u;
        result->wrapped_model_ranges = 5u;
        result->pointer_matches = pointer_matches;
        result->position_start = position_start;
        result->wall_ms = wall_end - wall_start;
        result->gpu_ms = gpu_elapsed_ms(command);
        return 1;
    }
}

int rust_star_metal_run_prefill_layer0_boundary(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    const rust_star_metal_prefill_layer0_weights *weights,
    const rust_star_metal_prefill_attention_ingress_weights *next_ingress,
    const rust_star_metal_prefill_layer_weights *next_layer,
    const rust_star_metal_prefill_layer_outputs *next_outputs,
    const rust_star_metal_prefill_kvnorm_weights *layer2_kvnorm,
    const rust_star_metal_prefill_compressor_weights *layer2_compressors,
    float *layer2_kv_norm_output,
    float *layer2_kv_rope_output,
    float *layer2_kv_cur_output,
    const float *layer2_kv_prefix,
    float *layer2_attn_compressed_output,
    float *layer2_indexer_compressed_output,
    float *layer2_attn_state_kv_output,
    float *layer2_attn_state_score_output,
    float *layer2_indexer_state_kv_output,
    float *layer2_indexer_state_score_output,
    const float *layer2_attn_compressed_prefix,
    const float *layer2_indexer_compressed_prefix,
    uint32_t n_vocab,
    uint32_t rows,
    uint32_t position_start,
    uint32_t kv_state_mode,
    const uint32_t *tokens,
    float *hc_collapsed,
    float *attn_norm,
    float *q_lora,
    float *q_lora_norm,
    float *kv_raw,
    float *kv_norm,
    float *q_raw,
    float *q_cur,
    float *kv_rope,
    float *kv_cur,
    float *raw_cache,
    const float *kv_prefix,
    float *full_kv,
    float *attention_output,
    float *attention_back,
    float *attention_low,
    float *attention_out,
    float *after_attention_hc,
    float *ffn_cur,
    float *ffn_norm,
    float *router_logits,
    float *router_probs,
    int32_t *router_selected,
    float *router_weights,
    float *routed_mid,
    float *routed_out,
    float *shared_out,
    float *after_ffn_hc,
    float *next_hc_collapsed,
    float *next_attn_norm,
    float *next_q_lora,
    rust_star_metal_prefill_layer0_probe_result *result,
    char *error,
    size_t error_bytes)
{
    const BOOL complete_layer1 = next_layer != NULL || next_outputs != NULL;
    const BOOL continue_layer1 = next_ingress != NULL || complete_layer1;
    const BOOL continue_layer2 = layer2_kvnorm != NULL || layer2_kv_norm_output != NULL;
    const BOOL continue_layer2_kv_state =
        layer2_kv_rope_output != NULL || layer2_kv_cur_output != NULL ||
        layer2_kv_prefix != NULL;
    const BOOL continue_layer2_compressors =
        layer2_compressors != NULL || layer2_attn_compressed_output != NULL ||
        layer2_indexer_compressed_output != NULL ||
        layer2_attn_state_kv_output != NULL ||
        layer2_attn_state_score_output != NULL ||
        layer2_indexer_state_kv_output != NULL ||
        layer2_indexer_state_score_output != NULL ||
        layer2_attn_compressed_prefix != NULL ||
        layer2_indexer_compressed_prefix != NULL;
    const BOOL partial_layer1 = next_hc_collapsed || next_attn_norm || next_q_lora;
    if (!opaque_context || !model_mapping || !weights || !tokens ||
        !hc_collapsed || !attn_norm || !q_lora || !q_lora_norm ||
        !kv_raw || !kv_norm || !q_raw || !q_cur || !kv_rope || !kv_cur ||
        !raw_cache || !kv_prefix || !full_kv || !attention_output ||
        !attention_back || !attention_low || !attention_out ||
        !after_attention_hc || !ffn_cur || !ffn_norm || !router_logits ||
        !router_probs || !router_selected || !router_weights || !routed_mid ||
        !routed_out || !shared_out || !after_ffn_hc || !result) {
        return fail_with_message(error, error_bytes,
            @"prefill layer-0 boundary received a null input");
    }
    if ((next_ingress && complete_layer1) ||
        ((next_layer == NULL) != (next_outputs == NULL)) ||
        ((layer2_kvnorm == NULL) != (layer2_kv_norm_output == NULL)) ||
        (continue_layer2 && !complete_layer1) ||
        (continue_layer2_kv_state &&
         (!continue_layer2 || !layer2_kv_rope_output ||
          !layer2_kv_cur_output || !layer2_kv_prefix)) ||
        (continue_layer2_compressors &&
         (!continue_layer2_kv_state || !layer2_compressors ||
          !layer2_attn_compressed_output || !layer2_indexer_compressed_output ||
          !layer2_attn_state_kv_output || !layer2_attn_state_score_output ||
          !layer2_indexer_state_kv_output || !layer2_indexer_state_score_output ||
          !layer2_attn_compressed_prefix || !layer2_indexer_compressed_prefix)) ||
        partial_layer1 != continue_layer1 ||
        (continue_layer1 && (!next_hc_collapsed || !next_attn_norm || !next_q_lora))) {
        return fail_with_message(error, error_bytes,
            @"prefill layers-0/1 boundary requires the complete layer-1 ingress output set");
    }
    if (complete_layer1 &&
        (!next_outputs->kv_prefix || !next_outputs->q_lora_norm ||
         !next_outputs->kv_norm || !next_outputs->q_cur ||
         !next_outputs->kv_rope || !next_outputs->kv_cur ||
         !next_outputs->attention_output || !next_outputs->attention_back ||
         !next_outputs->attention_low || !next_outputs->attention_out ||
         !next_outputs->after_attention_hc || !next_outputs->ffn_cur ||
         !next_outputs->ffn_norm || !next_outputs->router_logits ||
         !next_outputs->router_probs || !next_outputs->router_selected ||
         !next_outputs->router_weights || !next_outputs->routed_mid ||
         !next_outputs->routed_out || !next_outputs->shared_out ||
         !next_outputs->after_ffn_hc)) {
        return fail_with_message(error, error_bytes,
            @"prefill complete layer-1 boundary received a null fixture/output pointer");
    }
    if (n_vocab != 129280u || rows != 32u || position_start % 32u != 0u ||
        position_start + rows > 2048u) {
        return fail_with_message(error, error_bytes,
            @"prefill layer-0 boundary dimensions are invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        const BOOL retain_kv_state = kv_state_mode == 1u;
        const BOOL consume_kv_state = kv_state_mode == 2u;
        if (kv_state_mode > 2u ||
            ((retain_kv_state || consume_kv_state) && !complete_layer1)) {
            return fail_with_message(error, error_bytes,
                @"prefill live-KV state mode is invalid for this tile");
        }
        if (retain_kv_state) {
            context.prefillLayer0FullKv = nil;
            context.prefillLayer1FullKv = nil;
            context.prefillLayer2FullKv = nil;
            context.prefillLayer2FullQNorm = nil;
            context.prefillLayer2InputHc = nil;
            context.prefillLayer2AttnSplit = nil;
            context.prefillLayer2Tokens = nil;
            context.prefillLayer2AttnCompressed = nil;
            context.prefillLayer2AttnStateKv = nil;
            context.prefillLayer2AttnStateScore = nil;
            context.prefillLayer2IndexerCompressed = nil;
            context.prefillLayer2IndexerStateKv = nil;
            context.prefillLayer2IndexerStateScore = nil;
            context.prefillKvRows = 0u;
        }
        if (consume_kv_state &&
            (!context.prefillLayer0FullKv || !context.prefillLayer1FullKv ||
             context.prefillKvRows != position_start)) {
            return fail_with_message(error, error_bytes,
                @"prefill live-KV continuation state is missing or noncontiguous");
        }
        if (consume_kv_state && continue_layer2_kv_state &&
            (!context.prefillLayer2FullKv || !context.prefillLayer2FullQNorm ||
             !context.prefillLayer2InputHc || !context.prefillLayer2AttnSplit ||
             !context.prefillLayer2Tokens)) {
            return fail_with_message(error, error_bytes,
                @"prefill live layer-2 KV continuation state is missing");
        }
        if (consume_kv_state && continue_layer2_compressors &&
            (!context.prefillLayer2AttnCompressed ||
             !context.prefillLayer2AttnStateKv ||
             !context.prefillLayer2AttnStateScore ||
             !context.prefillLayer2IndexerCompressed ||
             !context.prefillLayer2IndexerStateKv ||
             !context.prefillLayer2IndexerStateScore)) {
            return fail_with_message(error, error_bytes,
                @"prefill live layer-2 compressor continuation state is missing");
        }
        if (!ensure_get_rows_f16_pipeline(context, error, error_bytes) ||
            !ensure_attention_ingress_pipelines(context, error, error_bytes) ||
            !ensure_moe_output_pipelines(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));

        uint64_t offsets[65] = {
            weights->embedding_offset, weights->hc_fn_offset,
            weights->hc_scale_offset, weights->hc_base_offset,
            weights->attn_norm_offset, weights->qkv.q_a_offset,
            weights->qkv.q_a_norm_offset, weights->qkv.kv_offset,
            weights->qkv.kv_norm_offset, weights->qkv.q_b_offset,
            weights->attn_sinks_offset, weights->attn_output_a_offset,
            weights->attn_output_b_offset,
            weights->ffn.hc_fn_offset, weights->ffn.hc_scale_offset,
            weights->ffn.hc_base_offset, weights->ffn.norm_offset,
            weights->ffn.router_gate_offset, weights->ffn.router_hash_offset,
            weights->ffn.routed_gate_offset, weights->ffn.routed_up_offset,
            weights->ffn.routed_down_offset, weights->ffn.shared_gate_offset,
            weights->ffn.shared_up_offset, weights->ffn.shared_down_offset,
            0, 0, 0, 0, 0,
        };
        uint64_t sizes[65] = {
            weights->embedding_bytes, weights->hc_fn_bytes,
            weights->hc_scale_bytes, weights->hc_base_bytes,
            weights->attn_norm_bytes, weights->qkv.q_a_bytes,
            weights->qkv.q_a_norm_bytes, weights->qkv.kv_bytes,
            weights->qkv.kv_norm_bytes, weights->qkv.q_b_bytes,
            weights->attn_sinks_bytes, weights->attn_output_a_bytes,
            weights->attn_output_b_bytes,
            weights->ffn.hc_fn_bytes, weights->ffn.hc_scale_bytes,
            weights->ffn.hc_base_bytes, weights->ffn.norm_bytes,
            weights->ffn.router_gate_bytes, weights->ffn.router_hash_bytes,
            weights->ffn.routed_gate_bytes, weights->ffn.routed_up_bytes,
            weights->ffn.routed_down_bytes, weights->ffn.shared_gate_bytes,
            weights->ffn.shared_up_bytes, weights->ffn.shared_down_bytes,
            0, 0, 0, 0, 0,
        };
        const uint32_t model_range_count = continue_layer2_compressors ? 65u :
            (continue_layer2 ? 57u :
            (complete_layer1 ? 49u : (continue_layer1 ? 30u : 25u)));
        if (continue_layer1) {
            const rust_star_metal_prefill_attention_ingress_weights *ingress =
                complete_layer1 ? &next_layer->ingress : next_ingress;
            offsets[25] = ingress->hc_fn_offset;
            offsets[26] = ingress->hc_scale_offset;
            offsets[27] = ingress->hc_base_offset;
            offsets[28] = ingress->norm_offset;
            offsets[29] = ingress->q_a_offset;
            sizes[25] = ingress->hc_fn_bytes;
            sizes[26] = ingress->hc_scale_bytes;
            sizes[27] = ingress->hc_base_bytes;
            sizes[28] = ingress->norm_bytes;
            sizes[29] = ingress->q_a_bytes;
        }
        if (complete_layer1) {
            offsets[30] = next_layer->q_a_norm_offset;
            offsets[31] = next_layer->kv_offset;
            offsets[32] = next_layer->kv_norm_offset;
            offsets[33] = next_layer->q_b_offset;
            offsets[34] = next_layer->attn_sinks_offset;
            offsets[35] = next_layer->attn_output_a_offset;
            offsets[36] = next_layer->attn_output_b_offset;
            offsets[37] = next_layer->ffn.hc_fn_offset;
            offsets[38] = next_layer->ffn.hc_scale_offset;
            offsets[39] = next_layer->ffn.hc_base_offset;
            offsets[40] = next_layer->ffn.norm_offset;
            offsets[41] = next_layer->ffn.router_gate_offset;
            offsets[42] = next_layer->ffn.router_hash_offset;
            offsets[43] = next_layer->ffn.routed_gate_offset;
            offsets[44] = next_layer->ffn.routed_up_offset;
            offsets[45] = next_layer->ffn.routed_down_offset;
            offsets[46] = next_layer->ffn.shared_gate_offset;
            offsets[47] = next_layer->ffn.shared_up_offset;
            offsets[48] = next_layer->ffn.shared_down_offset;
            sizes[30] = next_layer->q_a_norm_bytes;
            sizes[31] = next_layer->kv_bytes;
            sizes[32] = next_layer->kv_norm_bytes;
            sizes[33] = next_layer->q_b_bytes;
            sizes[34] = next_layer->attn_sinks_bytes;
            sizes[35] = next_layer->attn_output_a_bytes;
            sizes[36] = next_layer->attn_output_b_bytes;
            sizes[37] = next_layer->ffn.hc_fn_bytes;
            sizes[38] = next_layer->ffn.hc_scale_bytes;
            sizes[39] = next_layer->ffn.hc_base_bytes;
            sizes[40] = next_layer->ffn.norm_bytes;
            sizes[41] = next_layer->ffn.router_gate_bytes;
            sizes[42] = next_layer->ffn.router_hash_bytes;
            sizes[43] = next_layer->ffn.routed_gate_bytes;
            sizes[44] = next_layer->ffn.routed_up_bytes;
            sizes[45] = next_layer->ffn.routed_down_bytes;
            sizes[46] = next_layer->ffn.shared_gate_bytes;
            sizes[47] = next_layer->ffn.shared_up_bytes;
            sizes[48] = next_layer->ffn.shared_down_bytes;
        }
        if (continue_layer2) {
            offsets[49] = layer2_kvnorm->ingress.hc_fn_offset;
            offsets[50] = layer2_kvnorm->ingress.hc_scale_offset;
            offsets[51] = layer2_kvnorm->ingress.hc_base_offset;
            offsets[52] = layer2_kvnorm->ingress.norm_offset;
            offsets[53] = layer2_kvnorm->ingress.q_a_offset;
            offsets[54] = layer2_kvnorm->q_a_norm_offset;
            offsets[55] = layer2_kvnorm->kv_offset;
            offsets[56] = layer2_kvnorm->kv_norm_offset;
            sizes[49] = layer2_kvnorm->ingress.hc_fn_bytes;
            sizes[50] = layer2_kvnorm->ingress.hc_scale_bytes;
            sizes[51] = layer2_kvnorm->ingress.hc_base_bytes;
            sizes[52] = layer2_kvnorm->ingress.norm_bytes;
            sizes[53] = layer2_kvnorm->ingress.q_a_bytes;
            sizes[54] = layer2_kvnorm->q_a_norm_bytes;
            sizes[55] = layer2_kvnorm->kv_bytes;
            sizes[56] = layer2_kvnorm->kv_norm_bytes;
        }
        if (continue_layer2_compressors) {
            offsets[57] = layer2_compressors->attn_ape_offset;
            offsets[58] = layer2_compressors->attn_kv_offset;
            offsets[59] = layer2_compressors->attn_gate_offset;
            offsets[60] = layer2_compressors->attn_norm_offset;
            offsets[61] = layer2_compressors->indexer_ape_offset;
            offsets[62] = layer2_compressors->indexer_kv_offset;
            offsets[63] = layer2_compressors->indexer_gate_offset;
            offsets[64] = layer2_compressors->indexer_norm_offset;
            sizes[57] = layer2_compressors->attn_ape_bytes;
            sizes[58] = layer2_compressors->attn_kv_bytes;
            sizes[59] = layer2_compressors->attn_gate_bytes;
            sizes[60] = layer2_compressors->attn_norm_bytes;
            sizes[61] = layer2_compressors->indexer_ape_bytes;
            sizes[62] = layer2_compressors->indexer_kv_bytes;
            sizes[63] = layer2_compressors->indexer_gate_bytes;
            sizes[64] = layer2_compressors->indexer_norm_bytes;
        }
        id<MTLBuffer> model_buffers[65] = { nil };
        NSUInteger inner[65] = { 0 };
        BOOL matches[65] = { NO };
        for (uint32_t index = 0; index < model_range_count; index++) {
            model_buffers[index] = wrap_model_range(
                context, model_mapping, model_bytes, offsets[index], sizes[index],
                &inner[index], &matches[index], error, error_bytes);
            if (!model_buffers[index]) return 0;
        }

        enum {
            n_embd = 4096, n_hc = 4, hc_dim = 16384, mix_hc = 24,
            q_rank = 1024, kv_dim = 512, q_dim = 32768,
            raw_cache_rows = 128, prefill_rows = 2048,
            n_head = 64, attention_window = 128,
            n_expert = 256, n_used = 6, ffn_mid = 2048,
            compressor_ratio = 4, compressor_state_rows = 8,
            compressor_rows = prefill_rows/compressor_ratio,
            compressor_tile_rows = 32/compressor_ratio,
            attn_compressor_width = 1024, attn_compressor_head = 512,
            indexer_compressor_width = 256, indexer_compressor_head = 128,
        };
        const uint32_t raw_cache_target_row = position_start % raw_cache_rows;
        const uint32_t kv_prefix_rows = position_start;
        if (weights->attn_output_a_bytes != 8ull*1024ull*((4096ull/32ull)*34ull) ||
            weights->attn_output_b_bytes != 4096ull*((8192ull/32ull)*34ull)) {
            return fail_with_message(error, error_bytes,
                @"prefill layer-0 attention-output tensor shapes are invalid");
        }
        const rust_star_metal_prefill_attention_ingress_weights *next_ingress_weights =
            complete_layer1 ? &next_layer->ingress : next_ingress;
        if (continue_layer1 &&
            (next_ingress_weights->hc_fn_bytes != 16384ull*24ull*sizeof(uint16_t) ||
             next_ingress_weights->hc_scale_bytes != 3u*sizeof(float) ||
             next_ingress_weights->hc_base_bytes != 24u*sizeof(float) ||
             next_ingress_weights->norm_bytes != 4096u*sizeof(float) ||
             next_ingress_weights->q_a_bytes != 1024ull*((4096ull/32ull)*34ull))) {
            return fail_with_message(error, error_bytes,
                @"prefill layer-1 attention-ingress tensor shapes are invalid");
        }
        if (continue_layer2 &&
            (layer2_kvnorm->ingress.hc_fn_bytes != 16384ull*24ull*sizeof(uint16_t) ||
             layer2_kvnorm->ingress.hc_scale_bytes != 3u*sizeof(float) ||
             layer2_kvnorm->ingress.hc_base_bytes != 24u*sizeof(float) ||
             layer2_kvnorm->ingress.norm_bytes != 4096u*sizeof(float) ||
             layer2_kvnorm->ingress.q_a_bytes != 1024ull*((4096ull/32ull)*34ull) ||
             layer2_kvnorm->q_a_norm_bytes != 1024u*sizeof(float) ||
             layer2_kvnorm->kv_bytes != 512ull*((4096ull/32ull)*34ull) ||
             layer2_kvnorm->kv_norm_bytes != 512u*sizeof(float))) {
            return fail_with_message(error, error_bytes,
                @"prefill layer-2 KVnorm tensor shapes are invalid");
        }
        if (continue_layer2_compressors &&
            (layer2_compressors->attn_ape_bytes != 4ull*1024ull*sizeof(uint16_t) ||
             layer2_compressors->attn_kv_bytes != 4096ull*1024ull*sizeof(uint16_t) ||
             layer2_compressors->attn_gate_bytes != 4096ull*1024ull*sizeof(uint16_t) ||
             layer2_compressors->attn_norm_bytes != 512u*sizeof(float) ||
             layer2_compressors->indexer_ape_bytes != 4ull*256ull*sizeof(uint16_t) ||
             layer2_compressors->indexer_kv_bytes != 4096ull*256ull*sizeof(uint16_t) ||
             layer2_compressors->indexer_gate_bytes != 4096ull*256ull*sizeof(uint16_t) ||
             layer2_compressors->indexer_norm_bytes != 128u*sizeof(float))) {
            return fail_with_message(error, error_bytes,
                @"prefill layer-2 compressor tensor shapes are invalid");
        }
        const NSUInteger token_bytes = rows*sizeof(uint32_t);
        const NSUInteger embedding_bytes = rows*n_embd*sizeof(float);
        const NSUInteger hc_bytes = rows*hc_dim*sizeof(float);
        const NSUInteger mix_bytes = rows*mix_hc*sizeof(float);
        const NSUInteger attn_bytes = rows*n_embd*sizeof(float);
        const NSUInteger q_rank_bytes = rows*q_rank*sizeof(float);
        const NSUInteger kv_bytes = rows*kv_dim*sizeof(float);
        const NSUInteger q_bytes = rows*q_dim*sizeof(float);
        const NSUInteger raw_cache_bytes = raw_cache_rows*kv_dim*sizeof(float);
        const NSUInteger full_kv_bytes = prefill_rows*kv_dim*sizeof(float);
        const NSUInteger full_kv_prefix_bytes = kv_prefix_rows*kv_dim*sizeof(float);
        const uint32_t compressed_prefix_rows = position_start/compressor_ratio;
        const NSUInteger attn_compressed_prefix_bytes =
            compressed_prefix_rows*attn_compressor_head*sizeof(float);
        const NSUInteger indexer_compressed_prefix_bytes =
            compressed_prefix_rows*indexer_compressor_head*sizeof(float);
        const NSUInteger attention_mask_bytes = rows*prefill_rows*sizeof(uint16_t);
        const NSUInteger attention_low_bytes = rows*8192u*sizeof(float);
        const NSUInteger attention_out_bytes = rows*n_embd*sizeof(float);
        const NSUInteger router_bytes = rows*n_expert*sizeof(float);
        const NSUInteger selected_bytes = rows*n_used*sizeof(int32_t);
        const NSUInteger routed_mid_f16_bytes = rows*n_used*ffn_mid*sizeof(uint16_t);
        const NSUInteger routed_experts_bytes = rows*n_used*n_embd*sizeof(float);
        const NSUInteger ffn_mid_bytes = rows*ffn_mid*sizeof(float);
        const NSUInteger attn_projected_bytes =
            rows*attn_compressor_width*sizeof(float);
        const NSUInteger indexer_projected_bytes =
            rows*indexer_compressor_width*sizeof(float);
        const NSUInteger attn_compressed_bytes =
            compressor_rows*attn_compressor_head*sizeof(float);
        const NSUInteger indexer_compressed_bytes =
            compressor_rows*indexer_compressor_head*sizeof(float);
        const NSUInteger attn_state_bytes =
            compressor_state_rows*attn_compressor_width*sizeof(float);
        const NSUInteger indexer_state_bytes =
            compressor_state_rows*indexer_compressor_width*sizeof(float);
        id<MTLBuffer> token_buffer = [context.device newBufferWithBytes:tokens
            length:token_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> embedding_buffer = [context.device newBufferWithLength:embedding_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> hc_buffer = [context.device newBufferWithLength:hc_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> flat_hc_buffer = [context.device newBufferWithLength:hc_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> mix_buffer = [context.device newBufferWithLength:mix_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> split_buffer = [context.device newBufferWithLength:mix_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> collapsed_buffer = [context.device newBufferWithLength:attn_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> attn_buffer = [context.device newBufferWithLength:attn_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_buffer = [context.device newBufferWithLength:q_rank_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_norm_buffer = [context.device newBufferWithLength:q_rank_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_raw_buffer = [context.device newBufferWithLength:kv_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_norm_buffer = [context.device newBufferWithLength:kv_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_norm_snapshot_buffer = [context.device newBufferWithLength:kv_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_rope_buffer = [context.device newBufferWithLength:kv_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_half_buffer = [context.device newBufferWithLength:rows*kv_dim*sizeof(uint16_t)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> raw_cache_buffer = [context.device newBufferWithLength:raw_cache_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_raw_buffer = [context.device newBufferWithLength:q_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> q_cur_buffer = [context.device newBufferWithLength:q_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> full_kv_buffer = consume_kv_state ?
            context.prefillLayer0FullKv :
            [context.device newBufferWithLength:full_kv_bytes
                options:MTLResourceStorageModeShared];
        id<MTLBuffer> full_kv_half_buffer = [context.device
            newBufferWithLength:prefill_rows*kv_dim*sizeof(uint16_t)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention_mask_buffer = [context.device
            newBufferWithLength:attention_mask_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention_block_buffer = [context.device newBufferWithLength:128u
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention_pad_buffer = [context.device newBufferWithLength:1u
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention_output_buffer = [context.device newBufferWithLength:q_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention_back_buffer = [context.device newBufferWithLength:q_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention_low_buffer = [context.device newBufferWithLength:attention_low_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention_out_buffer = [context.device newBufferWithLength:attention_out_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> after_attention_hc_buffer = [context.device newBufferWithLength:hc_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention_group_ids_buffer = [context.device
            newBufferWithLength:rows*8u*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention_group_map_buffer = [context.device newBufferWithLength:1192u
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> ffn_flat_hc_buffer = [context.device newBufferWithLength:hc_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> ffn_mix_buffer = [context.device newBufferWithLength:mix_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> ffn_split_buffer = [context.device newBufferWithLength:mix_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> ffn_cur_buffer = [context.device newBufferWithLength:attn_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> ffn_norm_buffer = [context.device newBufferWithLength:attn_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> router_logits_buffer = [context.device newBufferWithLength:router_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> router_probs_buffer = [context.device newBufferWithLength:router_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> router_selected_buffer = [context.device newBufferWithLength:selected_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> router_weights_buffer = [context.device newBufferWithLength:selected_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> router_weight_sums_buffer = [context.device newBufferWithLength:rows*sizeof(float)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> routed_mid_buffer = [context.device newBufferWithLength:routed_mid_f16_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> routed_experts_buffer = [context.device newBufferWithLength:routed_experts_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> routed_out_buffer = [context.device newBufferWithLength:attn_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> shared_gate_buffer = [context.device newBufferWithLength:ffn_mid_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> shared_up_buffer = [context.device newBufferWithLength:ffn_mid_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> shared_mid_buffer = [context.device newBufferWithLength:ffn_mid_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> shared_out_buffer = [context.device newBufferWithLength:attn_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> after_ffn_hc_buffer = [context.device newBufferWithLength:hc_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> next_flat_hc_buffer = nil;
        id<MTLBuffer> next_mix_buffer = nil;
        id<MTLBuffer> next_split_buffer = nil;
        id<MTLBuffer> next_cur_buffer = nil;
        id<MTLBuffer> next_norm_buffer = nil;
        id<MTLBuffer> next_q_lora_buffer = nil;
        id<MTLBuffer> next_q_norm_buffer = nil;
        id<MTLBuffer> next_kv_raw_buffer = nil;
        id<MTLBuffer> next_kv_norm_buffer = nil;
        id<MTLBuffer> next_kv_norm_snapshot_buffer = nil;
        id<MTLBuffer> next_kv_rope_buffer = nil;
        id<MTLBuffer> next_kv_half_buffer = nil;
        id<MTLBuffer> next_raw_cache_buffer = nil;
        id<MTLBuffer> next_q_raw_buffer = nil;
        id<MTLBuffer> next_q_cur_buffer = nil;
        id<MTLBuffer> next_full_kv_buffer = nil;
        id<MTLBuffer> next_full_kv_half_buffer = nil;
        id<MTLBuffer> next_attention_output_buffer = nil;
        id<MTLBuffer> next_attention_back_buffer = nil;
        id<MTLBuffer> next_attention_low_buffer = nil;
        id<MTLBuffer> next_attention_out_buffer = nil;
        id<MTLBuffer> next_after_attention_hc_buffer = nil;
        id<MTLBuffer> next_ffn_flat_hc_buffer = nil;
        id<MTLBuffer> next_ffn_mix_buffer = nil;
        id<MTLBuffer> next_ffn_split_buffer = nil;
        id<MTLBuffer> next_ffn_cur_buffer = nil;
        id<MTLBuffer> next_ffn_norm_buffer = nil;
        id<MTLBuffer> next_router_logits_buffer = nil;
        id<MTLBuffer> next_router_probs_buffer = nil;
        id<MTLBuffer> next_router_selected_buffer = nil;
        id<MTLBuffer> next_router_weights_buffer = nil;
        id<MTLBuffer> next_routed_mid_buffer = nil;
        id<MTLBuffer> next_routed_out_buffer = nil;
        id<MTLBuffer> next_shared_out_buffer = nil;
        id<MTLBuffer> next_after_ffn_hc_buffer = nil;
        id<MTLBuffer> layer2_flat_hc_buffer = nil;
        id<MTLBuffer> layer2_mix_buffer = nil;
        id<MTLBuffer> layer2_split_buffer = nil;
        id<MTLBuffer> layer2_cur_buffer = nil;
        id<MTLBuffer> layer2_norm_buffer = nil;
        id<MTLBuffer> layer2_q_lora_buffer = nil;
        id<MTLBuffer> layer2_q_norm_buffer = nil;
        id<MTLBuffer> layer2_kv_raw_buffer = nil;
        id<MTLBuffer> layer2_kv_norm_buffer = nil;
        id<MTLBuffer> layer2_kv_norm_snapshot_buffer = nil;
        id<MTLBuffer> layer2_kv_rope_buffer = nil;
        id<MTLBuffer> layer2_full_kv_buffer = nil;
        id<MTLBuffer> layer2_full_q_norm_buffer = nil;
        id<MTLBuffer> layer2_input_hc_buffer = nil;
        id<MTLBuffer> layer2_attn_split_buffer = nil;
        id<MTLBuffer> layer2_tokens_buffer = nil;
        id<MTLBuffer> layer2_attn_projected_kv_buffer = nil;
        id<MTLBuffer> layer2_attn_projected_score_buffer = nil;
        id<MTLBuffer> layer2_indexer_projected_kv_buffer = nil;
        id<MTLBuffer> layer2_indexer_projected_score_buffer = nil;
        id<MTLBuffer> layer2_attn_packed_kv_buffer = nil;
        id<MTLBuffer> layer2_attn_packed_score_buffer = nil;
        id<MTLBuffer> layer2_attn_softmax_buffer = nil;
        id<MTLBuffer> layer2_indexer_packed_kv_buffer = nil;
        id<MTLBuffer> layer2_indexer_packed_score_buffer = nil;
        id<MTLBuffer> layer2_indexer_softmax_buffer = nil;
        id<MTLBuffer> layer2_attn_compressed_buffer = nil;
        id<MTLBuffer> layer2_attn_state_kv_buffer = nil;
        id<MTLBuffer> layer2_attn_state_score_buffer = nil;
        id<MTLBuffer> layer2_indexer_compressed_buffer = nil;
        id<MTLBuffer> layer2_indexer_state_kv_buffer = nil;
        id<MTLBuffer> layer2_indexer_state_score_buffer = nil;
        if (continue_layer1) {
            next_flat_hc_buffer = [context.device newBufferWithLength:hc_bytes
                options:MTLResourceStorageModeShared];
            next_mix_buffer = [context.device newBufferWithLength:mix_bytes
                options:MTLResourceStorageModeShared];
            next_split_buffer = [context.device newBufferWithLength:mix_bytes
                options:MTLResourceStorageModeShared];
            next_cur_buffer = [context.device newBufferWithLength:attn_bytes
                options:MTLResourceStorageModeShared];
            next_norm_buffer = [context.device newBufferWithLength:attn_bytes
                options:MTLResourceStorageModeShared];
            next_q_lora_buffer = [context.device newBufferWithLength:q_rank_bytes
                options:MTLResourceStorageModeShared];
        }
        if (complete_layer1) {
#define RUST_STAR_NEW_L1_BUFFER(name, bytes) \
            name = [context.device newBufferWithLength:(bytes) \
                options:MTLResourceStorageModeShared]
            RUST_STAR_NEW_L1_BUFFER(next_q_norm_buffer, q_rank_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_kv_raw_buffer, kv_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_kv_norm_buffer, kv_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_kv_norm_snapshot_buffer, kv_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_kv_rope_buffer, kv_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_kv_half_buffer, rows*kv_dim*sizeof(uint16_t));
            RUST_STAR_NEW_L1_BUFFER(next_raw_cache_buffer, raw_cache_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_q_raw_buffer, q_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_q_cur_buffer, q_bytes);
            if (consume_kv_state) {
                next_full_kv_buffer = context.prefillLayer1FullKv;
            } else {
                RUST_STAR_NEW_L1_BUFFER(next_full_kv_buffer, full_kv_bytes);
            }
            RUST_STAR_NEW_L1_BUFFER(next_full_kv_half_buffer,
                prefill_rows*kv_dim*sizeof(uint16_t));
            RUST_STAR_NEW_L1_BUFFER(next_attention_output_buffer, q_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_attention_back_buffer, q_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_attention_low_buffer, attention_low_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_attention_out_buffer, attention_out_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_after_attention_hc_buffer, hc_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_ffn_flat_hc_buffer, hc_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_ffn_mix_buffer, mix_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_ffn_split_buffer, mix_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_ffn_cur_buffer, attn_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_ffn_norm_buffer, attn_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_router_logits_buffer, router_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_router_probs_buffer, router_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_router_selected_buffer, selected_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_router_weights_buffer, selected_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_routed_mid_buffer, routed_mid_f16_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_routed_out_buffer, attn_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_shared_out_buffer, attn_bytes);
            RUST_STAR_NEW_L1_BUFFER(next_after_ffn_hc_buffer, hc_bytes);
#undef RUST_STAR_NEW_L1_BUFFER
        }
        if (continue_layer2) {
#define RUST_STAR_NEW_L2_BUFFER(name, bytes) \
            name = [context.device newBufferWithLength:(bytes) \
                options:MTLResourceStorageModeShared]
            RUST_STAR_NEW_L2_BUFFER(layer2_flat_hc_buffer, hc_bytes);
            RUST_STAR_NEW_L2_BUFFER(layer2_mix_buffer, mix_bytes);
            RUST_STAR_NEW_L2_BUFFER(layer2_split_buffer, mix_bytes);
            RUST_STAR_NEW_L2_BUFFER(layer2_cur_buffer, attn_bytes);
            RUST_STAR_NEW_L2_BUFFER(layer2_norm_buffer, attn_bytes);
            RUST_STAR_NEW_L2_BUFFER(layer2_q_lora_buffer, q_rank_bytes);
            RUST_STAR_NEW_L2_BUFFER(layer2_q_norm_buffer, q_rank_bytes);
            RUST_STAR_NEW_L2_BUFFER(layer2_kv_raw_buffer, kv_bytes);
            RUST_STAR_NEW_L2_BUFFER(layer2_kv_norm_buffer, kv_bytes);
            if (continue_layer2_kv_state) {
                RUST_STAR_NEW_L2_BUFFER(layer2_kv_norm_snapshot_buffer, kv_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_kv_rope_buffer, kv_bytes);
                if (consume_kv_state) {
                    layer2_full_kv_buffer = context.prefillLayer2FullKv;
                    layer2_full_q_norm_buffer = context.prefillLayer2FullQNorm;
                    layer2_input_hc_buffer = context.prefillLayer2InputHc;
                    layer2_attn_split_buffer = context.prefillLayer2AttnSplit;
                    layer2_tokens_buffer = context.prefillLayer2Tokens;
                } else {
                    RUST_STAR_NEW_L2_BUFFER(layer2_full_kv_buffer, full_kv_bytes);
                    RUST_STAR_NEW_L2_BUFFER(layer2_full_q_norm_buffer,
                        prefill_rows*q_rank*sizeof(float));
                    RUST_STAR_NEW_L2_BUFFER(layer2_input_hc_buffer,
                        prefill_rows*hc_dim*sizeof(float));
                    RUST_STAR_NEW_L2_BUFFER(layer2_attn_split_buffer,
                        prefill_rows*mix_hc*sizeof(float));
                    RUST_STAR_NEW_L2_BUFFER(layer2_tokens_buffer,
                        prefill_rows*sizeof(uint32_t));
                }
            }
            if (continue_layer2_compressors) {
                RUST_STAR_NEW_L2_BUFFER(layer2_attn_projected_kv_buffer,
                    attn_projected_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_attn_projected_score_buffer,
                    attn_projected_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_indexer_projected_kv_buffer,
                    indexer_projected_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_indexer_projected_score_buffer,
                    indexer_projected_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_attn_packed_kv_buffer,
                    attn_projected_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_attn_packed_score_buffer,
                    attn_projected_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_attn_softmax_buffer,
                    attn_projected_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_indexer_packed_kv_buffer,
                    indexer_projected_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_indexer_packed_score_buffer,
                    indexer_projected_bytes);
                RUST_STAR_NEW_L2_BUFFER(layer2_indexer_softmax_buffer,
                    indexer_projected_bytes);
                if (consume_kv_state) {
                    layer2_attn_compressed_buffer =
                        context.prefillLayer2AttnCompressed;
                    layer2_attn_state_kv_buffer = context.prefillLayer2AttnStateKv;
                    layer2_attn_state_score_buffer =
                        context.prefillLayer2AttnStateScore;
                    layer2_indexer_compressed_buffer =
                        context.prefillLayer2IndexerCompressed;
                    layer2_indexer_state_kv_buffer =
                        context.prefillLayer2IndexerStateKv;
                    layer2_indexer_state_score_buffer =
                        context.prefillLayer2IndexerStateScore;
                } else {
                    RUST_STAR_NEW_L2_BUFFER(layer2_attn_compressed_buffer,
                        attn_compressed_bytes);
                    RUST_STAR_NEW_L2_BUFFER(layer2_attn_state_kv_buffer,
                        attn_state_bytes);
                    RUST_STAR_NEW_L2_BUFFER(layer2_attn_state_score_buffer,
                        attn_state_bytes);
                    RUST_STAR_NEW_L2_BUFFER(layer2_indexer_compressed_buffer,
                        indexer_compressed_bytes);
                    RUST_STAR_NEW_L2_BUFFER(layer2_indexer_state_kv_buffer,
                        indexer_state_bytes);
                    RUST_STAR_NEW_L2_BUFFER(layer2_indexer_state_score_buffer,
                        indexer_state_bytes);
                }
            }
#undef RUST_STAR_NEW_L2_BUFFER
        }
        const NSUInteger routed_map_tpe_bytes = n_expert*sizeof(int32_t);
        const NSUInteger routed_map_ids_bytes = n_expert*rows*sizeof(int32_t);
        const NSUInteger routed_map_work_offset =
            (routed_map_tpe_bytes+routed_map_ids_bytes+7u)&~7u;
        const NSUInteger routed_map_work_cap =
            (rows*n_used+31u*n_expert+31u)/32u;
        id<MTLBuffer> routed_map_buffer = [context.device
            newBufferWithLength:routed_map_work_offset+8u+routed_map_work_cap*8u
            options:MTLResourceStorageModeShared];
        if (!token_buffer || !embedding_buffer || !hc_buffer || !flat_hc_buffer ||
            !mix_buffer || !split_buffer || !collapsed_buffer || !attn_buffer ||
            !q_buffer || !q_norm_buffer || !kv_raw_buffer || !kv_norm_buffer ||
            !kv_norm_snapshot_buffer || !kv_rope_buffer || !kv_half_buffer ||
            !raw_cache_buffer || !q_raw_buffer || !q_cur_buffer ||
            !full_kv_buffer || !full_kv_half_buffer || !attention_mask_buffer ||
            !attention_block_buffer || !attention_pad_buffer ||
            !attention_output_buffer || !attention_back_buffer ||
            !attention_low_buffer || !attention_out_buffer ||
            !after_attention_hc_buffer || !attention_group_ids_buffer ||
            !attention_group_map_buffer || !ffn_flat_hc_buffer ||
            !ffn_mix_buffer || !ffn_split_buffer || !ffn_cur_buffer ||
            !ffn_norm_buffer || !router_logits_buffer || !router_probs_buffer ||
            !router_selected_buffer || !router_weights_buffer ||
            !router_weight_sums_buffer || !routed_mid_buffer ||
            !routed_experts_buffer || !routed_out_buffer || !shared_gate_buffer ||
            !shared_up_buffer || !shared_mid_buffer || !shared_out_buffer ||
            !after_ffn_hc_buffer || !routed_map_buffer) {
            return fail_with_message(error, error_bytes,
                @"failed to allocate prefill layer-0 boundary activation buffers");
        }
        if (continue_layer1 &&
            (!next_flat_hc_buffer || !next_mix_buffer || !next_split_buffer ||
             !next_cur_buffer || !next_norm_buffer || !next_q_lora_buffer)) {
            return fail_with_message(error, error_bytes,
                @"failed to allocate prefill layer-1 ingress activation buffers");
        }
        if (complete_layer1 &&
            (!next_q_norm_buffer || !next_kv_raw_buffer || !next_kv_norm_buffer ||
             !next_kv_norm_snapshot_buffer || !next_kv_rope_buffer ||
             !next_kv_half_buffer || !next_raw_cache_buffer ||
             !next_q_raw_buffer || !next_q_cur_buffer || !next_full_kv_buffer ||
             !next_full_kv_half_buffer || !next_attention_output_buffer ||
             !next_attention_back_buffer || !next_attention_low_buffer ||
             !next_attention_out_buffer || !next_after_attention_hc_buffer ||
             !next_ffn_flat_hc_buffer || !next_ffn_mix_buffer ||
             !next_ffn_split_buffer || !next_ffn_cur_buffer ||
             !next_ffn_norm_buffer || !next_router_logits_buffer ||
             !next_router_probs_buffer || !next_router_selected_buffer ||
             !next_router_weights_buffer || !next_routed_mid_buffer ||
             !next_routed_out_buffer || !next_shared_out_buffer ||
             !next_after_ffn_hc_buffer)) {
            return fail_with_message(error, error_bytes,
                @"failed to allocate complete prefill layer-1 activation buffers");
        }
        if (continue_layer2 &&
            (!layer2_flat_hc_buffer || !layer2_mix_buffer || !layer2_split_buffer ||
             !layer2_cur_buffer || !layer2_norm_buffer || !layer2_q_lora_buffer ||
             !layer2_q_norm_buffer || !layer2_kv_raw_buffer ||
             !layer2_kv_norm_buffer ||
             (continue_layer2_kv_state &&
              (!layer2_kv_norm_snapshot_buffer || !layer2_kv_rope_buffer ||
               !layer2_full_kv_buffer)) ||
             (continue_layer2_compressors &&
              (!layer2_attn_projected_kv_buffer ||
               !layer2_attn_projected_score_buffer ||
               !layer2_indexer_projected_kv_buffer ||
               !layer2_indexer_projected_score_buffer ||
               !layer2_attn_packed_kv_buffer ||
               !layer2_attn_packed_score_buffer ||
               !layer2_attn_softmax_buffer ||
               !layer2_indexer_packed_kv_buffer ||
               !layer2_indexer_packed_score_buffer ||
               !layer2_indexer_softmax_buffer ||
               !layer2_attn_compressed_buffer ||
               !layer2_attn_state_kv_buffer ||
               !layer2_attn_state_score_buffer ||
               !layer2_indexer_compressed_buffer ||
               !layer2_indexer_state_kv_buffer ||
               !layer2_indexer_state_score_buffer)))) {
            return fail_with_message(error, error_bytes,
                @"failed to allocate prefill layer-2 KVnorm activation buffers");
        }
        float *raw_cache_contents = raw_cache_buffer.contents;
        for (NSUInteger index = 0; index < raw_cache_rows*kv_dim; index++) {
            raw_cache_contents[index] = -12345.5f;
        }
        if (consume_kv_state) {
            if (memcmp(full_kv_buffer.contents, kv_prefix,
                       full_kv_prefix_bytes) != 0 ||
                memcmp(next_full_kv_buffer.contents, next_outputs->kv_prefix,
                       full_kv_prefix_bytes) != 0) {
                return fail_with_message(error, error_bytes,
                    @"prefill retained live-KV prefix differs from the C0 oracle");
            }
        } else {
            if (full_kv_prefix_bytes > 0) {
                memcpy(full_kv_buffer.contents, kv_prefix, full_kv_prefix_bytes);
            }
            float *full_kv_contents = full_kv_buffer.contents;
            for (NSUInteger index = kv_prefix_rows*kv_dim;
                 index < prefill_rows*kv_dim; index++) {
                full_kv_contents[index] = -23456.5f;
            }
        }
        if (complete_layer1) {
            float *next_raw_cache_contents = next_raw_cache_buffer.contents;
            for (NSUInteger index = 0; index < raw_cache_rows*kv_dim; index++) {
                next_raw_cache_contents[index] = -12345.5f;
            }
            if (!consume_kv_state) {
                if (full_kv_prefix_bytes > 0) {
                    memcpy(next_full_kv_buffer.contents, next_outputs->kv_prefix,
                        full_kv_prefix_bytes);
                }
                float *next_full_kv_contents = next_full_kv_buffer.contents;
                for (NSUInteger index = kv_prefix_rows*kv_dim;
                     index < prefill_rows*kv_dim; index++) {
                    next_full_kv_contents[index] = -23456.5f;
                }
            }
        }
        if (continue_layer2_kv_state) {
            if (consume_kv_state) {
                if (memcmp(layer2_full_kv_buffer.contents, layer2_kv_prefix,
                           full_kv_prefix_bytes) != 0) {
                    return fail_with_message(error, error_bytes,
                        @"prefill retained live layer-2 KV prefix differs from the C0 oracle");
                }
            } else {
                if (full_kv_prefix_bytes > 0) {
                    memcpy(layer2_full_kv_buffer.contents, layer2_kv_prefix,
                           full_kv_prefix_bytes);
                }
                float *layer2_full_kv_contents = layer2_full_kv_buffer.contents;
                for (NSUInteger index = kv_prefix_rows*kv_dim;
                     index < prefill_rows*kv_dim; index++) {
                    layer2_full_kv_contents[index] = -23456.5f;
                }
            }
        }
        if (continue_layer2_compressors) {
            if (consume_kv_state) {
                if (memcmp(layer2_attn_compressed_buffer.contents,
                           layer2_attn_compressed_prefix,
                           attn_compressed_prefix_bytes) != 0 ||
                    memcmp(layer2_indexer_compressed_buffer.contents,
                           layer2_indexer_compressed_prefix,
                           indexer_compressed_prefix_bytes) != 0) {
                    return fail_with_message(error, error_bytes,
                        @"prefill retained live layer-2 compressor prefix differs from the C0 oracle");
                }
            } else {
                if (attn_compressed_prefix_bytes > 0) {
                    memcpy(layer2_attn_compressed_buffer.contents,
                           layer2_attn_compressed_prefix,
                           attn_compressed_prefix_bytes);
                }
                if (indexer_compressed_prefix_bytes > 0) {
                    memcpy(layer2_indexer_compressed_buffer.contents,
                           layer2_indexer_compressed_prefix,
                           indexer_compressed_prefix_bytes);
                }
                float *attn_compressed_contents =
                    layer2_attn_compressed_buffer.contents;
                for (NSUInteger index = compressed_prefix_rows*attn_compressor_head;
                     index < compressor_rows*attn_compressor_head; index++) {
                    attn_compressed_contents[index] = -34567.5f;
                }
                float *indexer_compressed_contents =
                    layer2_indexer_compressed_buffer.contents;
                for (NSUInteger index = compressed_prefix_rows*indexer_compressor_head;
                     index < compressor_rows*indexer_compressor_head; index++) {
                    indexer_compressed_contents[index] = -34567.5f;
                }
                memset(layer2_attn_state_kv_buffer.contents, 0, attn_state_bytes);
                memset(layer2_indexer_state_kv_buffer.contents, 0,
                       indexer_state_bytes);
                float *attn_scores = layer2_attn_state_score_buffer.contents;
                for (NSUInteger index = 0;
                     index < compressor_state_rows*attn_compressor_width; index++) {
                    attn_scores[index] = -INFINITY;
                }
                float *indexer_scores =
                    layer2_indexer_state_score_buffer.contents;
                for (NSUInteger index = 0;
                     index < compressor_state_rows*indexer_compressor_width; index++) {
                    indexer_scores[index] = -INFINITY;
                }
            }
        }
        uint16_t *attention_mask = attention_mask_buffer.contents;
        for (uint32_t query = 0; query < rows; query++) {
            const uint32_t query_position = position_start + query;
            for (uint32_t key = 0; key < prefill_rows; key++) {
                const bool visible = key <= query_position &&
                    query_position - key < attention_window;
                attention_mask[(NSUInteger)query*prefill_rows + key] =
                    visible ? 0x0000u : 0xfc00u;
            }
        }
        int32_t *attention_group_ids = attention_group_ids_buffer.contents;
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t group = 0; group < 8u; group++) {
                attention_group_ids[(NSUInteger)row*8u + group] = (int32_t)group;
            }
        }

        rust_star_get_rows_args get_rows = {
            .ne00t=n_embd, .ne00=n_embd,
            .nb01=n_embd*sizeof(uint16_t),
            .nb02=(uint64_t)n_vocab*n_embd*sizeof(uint16_t),
            .nb03=(uint64_t)n_vocab*n_embd*sizeof(uint16_t),
            .ne10=(int32_t)rows,
            .nb10=sizeof(int32_t), .nb11=token_bytes, .nb12=token_bytes,
            .nb1=n_embd*sizeof(float), .nb2=embedding_bytes, .nb3=embedding_bytes,
        };
        rust_star_repeat_args repeat = {
            .ne00=n_embd, .ne01=1, .ne02=(int32_t)rows, .ne03=1,
            .nb00=sizeof(float), .nb01=n_embd*sizeof(float),
            .nb02=n_embd*sizeof(float), .nb03=embedding_bytes,
            .ne0=n_embd, .ne1=n_hc, .ne2=(int32_t)rows, .ne3=1,
            .nb0=sizeof(float), .nb1=n_embd*sizeof(float),
            .nb2=n_hc*n_embd*sizeof(float), .nb3=hc_bytes,
        };
        rust_star_norm_args hc_norm = {
            .ne00=hc_dim, .ne00_t=hc_dim/4,
            .nb1=hc_dim*sizeof(float), .nb2=hc_bytes, .nb3=hc_bytes,
            .eps=1.0e-6f,
            .nef1={(int32_t)rows,1,1}, .nef2={1,1,1}, .nef3={1,1,1},
            .nbf1={hc_dim*sizeof(float),hc_dim*sizeof(float),hc_dim*sizeof(float)},
            .nbf2={hc_bytes,hc_dim*sizeof(float),hc_dim*sizeof(float)},
            .nbf3={hc_bytes,hc_dim*sizeof(float),hc_dim*sizeof(float)},
        };
        rust_star_q8_mm_args hc_mm = {
            .ne00=hc_dim, .ne02=1,
            .nb01=hc_dim*sizeof(uint16_t),
            .nb02=(uint64_t)hc_dim*mix_hc*sizeof(uint16_t),
            .nb03=(uint64_t)hc_dim*mix_hc*sizeof(uint16_t), .ne12=1,
            .nb10=sizeof(float), .nb11=hc_dim*sizeof(float),
            .nb12=hc_bytes, .nb13=hc_bytes,
            .ne0=mix_hc, .ne1=(int32_t)rows, .r2=1, .r3=1,
        };
        rust_star_hc_ingress_args hc = {
            .n_embd=n_embd, .n_hc=n_hc, .sinkhorn_iters=20,
            .n_rows=rows, .mix_hc=mix_hc,
            .nb_mix1=mix_hc*sizeof(float), .nb_split1=mix_hc*sizeof(float),
            .nb_x0=sizeof(float), .nb_x1=n_embd*sizeof(float),
            .nb_x2=n_hc*n_embd*sizeof(float),
            .nb0=sizeof(float), .nb1=n_embd*sizeof(float),
            .nb_norm1=n_embd*sizeof(float), .eps=1.0e-6f, .norm_eps=1.0e-6f,
        };
#define RUST_STAR_LAYER0_Q8_MM_ARGS(in_width, out_width, weight_bytes) \
        (rust_star_q8_mm_args){ \
            .ne00=(in_width), .ne02=1, \
            .nb01=(uint64_t)((in_width)/32u)*34u, \
            .nb02=(weight_bytes), .nb03=(weight_bytes), .ne12=1, \
            .nb10=sizeof(float), .nb11=(uint64_t)(in_width)*sizeof(float), \
            .nb12=(uint64_t)(in_width)*rows*sizeof(float), \
            .nb13=(uint64_t)(in_width)*rows*sizeof(float), \
            .ne0=(out_width), .ne1=(int32_t)rows, .r2=1, .r3=1 \
        }
        rust_star_q8_mm_args q_a_args = RUST_STAR_LAYER0_Q8_MM_ARGS(
            n_embd, q_rank, weights->qkv.q_a_bytes);
        rust_star_q8_mm_args kv_args = RUST_STAR_LAYER0_Q8_MM_ARGS(
            n_embd, kv_dim, weights->qkv.kv_bytes);
        rust_star_q8_mm_args q_b_args = RUST_STAR_LAYER0_Q8_MM_ARGS(
            q_rank, q_dim, weights->qkv.q_b_bytes);
#undef RUST_STAR_LAYER0_Q8_MM_ARGS
#define RUST_STAR_COMPRESSOR_F16_MM_ARGS(out_width, weight_bytes) \
        (rust_star_q8_mm_args){ \
            .ne00=n_embd, .ne02=1, \
            .nb01=(uint64_t)n_embd*sizeof(uint16_t), \
            .nb02=(weight_bytes), .nb03=(weight_bytes), .ne12=1, \
            .nb10=sizeof(float), .nb11=(uint64_t)n_embd*sizeof(float), \
            .nb12=(uint64_t)n_embd*rows*sizeof(float), \
            .nb13=(uint64_t)n_embd*rows*sizeof(float), \
            .ne0=(out_width), .ne1=(int32_t)rows, .r2=1, .r3=1 \
        }
        rust_star_q8_mm_args attn_compressor_kv_args =
            RUST_STAR_COMPRESSOR_F16_MM_ARGS(attn_compressor_width,
                continue_layer2_compressors ? sizes[58] : 0u);
        rust_star_q8_mm_args attn_compressor_gate_args =
            RUST_STAR_COMPRESSOR_F16_MM_ARGS(attn_compressor_width,
                continue_layer2_compressors ? sizes[59] : 0u);
        rust_star_q8_mm_args indexer_compressor_kv_args =
            RUST_STAR_COMPRESSOR_F16_MM_ARGS(indexer_compressor_width,
                continue_layer2_compressors ? sizes[62] : 0u);
        rust_star_q8_mm_args indexer_compressor_gate_args =
            RUST_STAR_COMPRESSOR_F16_MM_ARGS(indexer_compressor_width,
                continue_layer2_compressors ? sizes[63] : 0u);
#undef RUST_STAR_COMPRESSOR_F16_MM_ARGS
        rust_star_qkv_norm_args qkv_norm_args = {
            .q_n=q_rank, .q_n4=q_rank/4, .kv_n=kv_dim, .kv_n4=kv_dim/4,
            .q_row_stride=q_rank*sizeof(float), .kv_row_stride=kv_dim*sizeof(float),
            .eps=1.0e-6f,
        };
        rust_star_head_norm_rope_args rope_args = {
            .n_head=64, .head_dim=512, .head_dim4=128, .n_dims=64,
            .n_ctx_orig=0, .pos0=(int32_t)position_start, .inverse=0,
            .eps=1.0e-6f, .freq_base=10000.0f, .freq_scale=1.0f,
            .ext_factor=0.0f, .attn_factor=1.0f,
            .beta_fast=32.0f, .beta_slow=1.0f,
        };
        rust_star_rope_tail_args kv_rope_args = {
            .ne00=kv_dim, .ne01=1, .ne02=rows, .ne03=1,
            .nb00=sizeof(float), .nb01=kv_dim*sizeof(float),
            .nb02=kv_dim*sizeof(float), .nb03=kv_bytes,
            .nb0=sizeof(float), .nb1=kv_dim*sizeof(float),
            .nb2=kv_dim*sizeof(float), .nb3=kv_bytes,
            .n_dims=64, .mode=0, .n_ctx_orig=0, .inverse=0,
            .freq_base=10000.0f, .freq_scale=1.0f,
            .ext_factor=0.0f, .attn_factor=1.0f,
            .beta_fast=32.0f, .beta_slow=1.0f, .src2=false,
        };
        const float layer2_freq_scale = 1.0f/16.0f;
        rust_star_rope_tail_args layer2_kv_rope_args = kv_rope_args;
        layer2_kv_rope_args.n_ctx_orig = 65536;
        layer2_kv_rope_args.freq_base = 160000.0f;
        layer2_kv_rope_args.freq_scale = layer2_freq_scale;
        layer2_kv_rope_args.ext_factor = 1.0f;
        layer2_kv_rope_args.attn_factor =
            1.0f/(1.0f + 0.1f*logf(1.0f/layer2_freq_scale));
        int32_t positions[32];
        for (uint32_t row = 0; row < rows; row++) {
            positions[row] = (int32_t)(position_start + row);
        }
        id<MTLBuffer> position_buffer = [context.device newBufferWithBytes:positions
            length:rows*sizeof(int32_t) options:MTLResourceStorageModeShared];
        int32_t compressed_positions[8];
        for (uint32_t row = 0; row < compressor_tile_rows; row++) {
            compressed_positions[row] =
                (int32_t)(position_start + row*compressor_ratio);
        }
        id<MTLBuffer> compressed_position_buffer = continue_layer2_compressors ?
            [context.device newBufferWithBytes:compressed_positions
                length:compressor_tile_rows*sizeof(int32_t)
                options:MTLResourceStorageModeShared] : nil;
        rust_star_fp8_quantize_args kv_fp8_args = {
            .ne00=kv_dim, .ne01=1, .ne02=rows, .ne03=1,
            .nb00=sizeof(float), .nb01=kv_dim*sizeof(float),
            .nb02=kv_dim*sizeof(float), .nb03=kv_bytes,
            .nb0=sizeof(float), .nb1=kv_dim*sizeof(float),
            .nb2=kv_dim*sizeof(float), .nb3=kv_bytes, .n_rot=64,
        };
        const uint64_t staged_row_bytes = kv_dim*sizeof(uint16_t);
        const uint64_t attention_head_bytes = kv_dim*sizeof(float);
        rust_star_flash_blk_args attention_blk_args = {
            .ne01=(int32_t)rows, .ne30=prefill_rows,
            .ne31=(int32_t)rows, .ne32=1, .ne33=1,
            .nb31=prefill_rows*sizeof(uint16_t),
            .nb32=attention_mask_bytes, .nb33=attention_mask_bytes,
        };
        rust_star_flash_vec_args attention_args = {
            .ne01=(int32_t)rows, .ne02=n_head, .ne03=1,
            .nb01=n_head*attention_head_bytes, .nb02=attention_head_bytes,
            .nb03=q_bytes,
            .ne11=prefill_rows, .ne_12_2=1, .ne_12_3=1, .ns10=kv_dim,
            .nb11=staged_row_bytes,
            .nb12=(uint64_t)prefill_rows*staged_row_bytes,
            .nb13=(uint64_t)prefill_rows*staged_row_bytes,
            .ns20=kv_dim,
            .nb21=staged_row_bytes,
            .nb22=(uint64_t)prefill_rows*staged_row_bytes,
            .nb23=(uint64_t)prefill_rows*staged_row_bytes,
            .ne31=(int32_t)rows, .ne32=1, .ne33=1,
            .nb31=prefill_rows*sizeof(uint16_t),
            .nb32=attention_mask_bytes, .nb33=attention_mask_bytes,
            .ne1=n_head, .ne2=(int32_t)rows, .ne3=1,
            .scale=1.0f/sqrtf((float)kv_dim), .max_bias=0.0f,
            .m0=0.0f, .m1=0.0f, .n_head_log2=0, .logit_softcap=0.0f,
        };
        rust_star_rope_tail_args attention_inverse_args = {
            .ne00=kv_dim, .ne01=n_head, .ne02=rows, .ne03=1,
            .nb00=sizeof(float), .nb01=attention_head_bytes,
            .nb02=n_head*attention_head_bytes, .nb03=q_bytes,
            .nb0=sizeof(float), .nb1=attention_head_bytes,
            .nb2=n_head*attention_head_bytes, .nb3=q_bytes,
            .n_dims=64, .mode=0, .n_ctx_orig=0, .inverse=1,
            .freq_base=10000.0f, .freq_scale=1.0f,
            .ext_factor=0.0f, .attn_factor=1.0f,
            .beta_fast=32.0f, .beta_slow=1.0f, .src2=false,
        };
        rust_star_q8_mm_id_map_args attention_output_map_args = {
            .ne02=8, .ne10=4096, .ne11=8,
            .nb11=4096u*sizeof(float), .nb12=8u*4096u*sizeof(float),
            .ne21=(int32_t)rows, .ne20=8, .nb21=8u*sizeof(int32_t),
        };
        rust_star_q8_mm_id_args attention_output_low_args = {
            .ne00=4096, .ne02=8,
            .nb01=(4096u/32u)*34u,
            .nb02=1024ull*((4096u/32u)*34u),
            .nb03=8ull*1024ull*((4096u/32u)*34u),
            .ne11=8, .nb10=sizeof(float),
            .nb11=4096u*sizeof(float),
            .nb12=8u*4096u*sizeof(float),
            .nb13=(uint64_t)rows*8u*4096u*sizeof(float),
            .ne20=8, .ne21=(int32_t)rows, .ne0=1024, .ne1=8,
            .r2=1, .r3=1,
        };
        rust_star_q8_mm_args attention_output_args = {
            .ne00=8192, .ne02=1,
            .nb01=(8192u/32u)*34u,
            .nb02=weights->attn_output_b_bytes,
            .nb03=weights->attn_output_b_bytes,
            .ne12=1, .nb10=sizeof(float),
            .nb11=8192u*sizeof(float),
            .nb12=(uint64_t)rows*8192u*sizeof(float),
            .nb13=(uint64_t)rows*8192u*sizeof(float),
            .ne0=n_embd, .ne1=(int32_t)rows, .r2=1, .r3=1,
        };
        rust_star_hc_expand_args attention_hc_args = {
            .n_embd=n_embd, .n_hc=n_hc, .n_tokens=rows,
            .nb_block0=sizeof(float), .nb_block1=n_embd*sizeof(float),
            .nb_add0=sizeof(float), .nb_add1=n_embd*sizeof(float),
            .nb_res0=sizeof(float), .nb_res1=n_embd*sizeof(float),
            .nb_res2=hc_dim*sizeof(float),
            .nb_post0=sizeof(float), .nb_post1=mix_hc*sizeof(float),
            .nb_comb0=sizeof(float), .nb_comb1=n_hc*sizeof(float),
            .nb_comb2=mix_hc*sizeof(float),
            .nb0=sizeof(float), .nb1=n_embd*sizeof(float),
            .nb2=hc_dim*sizeof(float), .has_add=0,
        };
        rust_star_q8_mm_args ffn_hc_mm = hc_mm;
        rust_star_hc_ingress_args ffn_hc_args = hc;
        rust_star_q8_mm_args router_mm = {
            .ne00=n_embd, .ne02=1,
            .nb01=n_embd*sizeof(uint16_t),
            .nb02=(uint64_t)n_embd*n_expert*sizeof(uint16_t),
            .nb03=(uint64_t)n_embd*n_expert*sizeof(uint16_t), .ne12=1,
            .nb10=sizeof(float), .nb11=n_embd*sizeof(float),
            .nb12=(uint64_t)rows*n_embd*sizeof(float),
            .nb13=(uint64_t)rows*n_embd*sizeof(float),
            .ne0=n_expert, .ne1=(int32_t)rows, .r2=1, .r3=1,
        };
        rust_star_sum_rows_args router_sum_args = {
            .ne00=n_used, .ne01=rows, .ne02=1, .ne03=1,
            .nb00=sizeof(float), .nb01=n_used*sizeof(float),
            .nb02=(uint64_t)rows*n_used*sizeof(float),
            .nb03=(uint64_t)rows*n_used*sizeof(float),
            .ne0=1, .ne1=rows, .ne2=1, .ne3=1,
            .nb0=sizeof(float), .nb1=sizeof(float),
            .nb2=(uint64_t)rows*sizeof(float),
            .nb3=(uint64_t)rows*sizeof(float),
        };
        rust_star_q8_mm_id_map_args routed_map_args = {
            .ne02=n_expert, .ne10=n_embd, .ne11=1,
            .nb11=n_embd*sizeof(float), .nb12=n_embd*sizeof(float),
            .ne21=(int32_t)rows, .ne20=n_used,
            .nb21=n_used*sizeof(int32_t),
        };
        const uint64_t routed_gate_row_bytes = 1056u;
        const uint64_t routed_gate_expert_bytes = ffn_mid*routed_gate_row_bytes;
        rust_star_q8_mm_id_args routed_gate_args = {
            .ne00=n_embd, .ne02=n_expert,
            .nb01=routed_gate_row_bytes, .nb02=routed_gate_expert_bytes,
            .nb03=n_expert*routed_gate_expert_bytes,
            .ne11=1, .nb10=sizeof(float), .nb11=n_embd*sizeof(float),
            .nb12=n_embd*sizeof(float),
            .nb13=(uint64_t)rows*n_embd*sizeof(float),
            .ne20=n_used, .ne21=(int32_t)rows, .ne0=ffn_mid, .ne1=n_used,
            .r2=1, .r3=1, .tp_rank=0, .tp_world=0, .tp_expert_base=0,
        };
        rust_star_moe_swiglu_weight_args routed_activation_args = {
            .width=ffn_mid, .rows=rows*n_used,
            .gate_row_stride=ffn_mid*sizeof(float),
            .up_row_stride=ffn_mid*sizeof(float),
            .mid_row_stride=ffn_mid*sizeof(uint16_t),
            .weight_stride=sizeof(float), .write_clamped=0, .clamp_value=10.0f,
        };
        const uint64_t routed_down_row_bytes = 672u;
        const uint64_t routed_down_expert_bytes = n_embd*routed_down_row_bytes;
        rust_star_q8_mm_id_args routed_down_args = {
            .ne00=ffn_mid, .ne02=n_expert,
            .nb01=routed_down_row_bytes, .nb02=routed_down_expert_bytes,
            .nb03=n_expert*routed_down_expert_bytes,
            .ne11=n_used, .nb10=sizeof(uint16_t),
            .nb11=ffn_mid*sizeof(uint16_t),
            .nb12=(uint64_t)n_used*ffn_mid*sizeof(uint16_t),
            .nb13=(uint64_t)rows*n_used*ffn_mid*sizeof(uint16_t),
            .ne20=n_used, .ne21=(int32_t)rows, .ne0=n_embd, .ne1=n_used,
            .r2=1, .r3=1, .tp_rank=0, .tp_world=0, .tp_expert_base=0,
        };
        rust_star_moe_sum_args routed_sum_args = {
            .width=n_embd, .tokens=rows,
            .src_token_stride=(uint64_t)n_used*n_embd*sizeof(float),
            .dst_token_stride=n_embd*sizeof(float),
        };
#define RUST_STAR_FFN_Q8_MM_ARGS(in_width, out_width, weight_bytes) \
        (rust_star_q8_mm_args){ \
            .ne00=(in_width), .ne02=1, \
            .nb01=(uint64_t)((in_width)/32u)*34u, \
            .nb02=(weight_bytes), .nb03=(weight_bytes), .ne12=1, \
            .nb10=sizeof(float), .nb11=(uint64_t)(in_width)*sizeof(float), \
            .nb12=(uint64_t)(in_width)*rows*sizeof(float), \
            .nb13=(uint64_t)(in_width)*rows*sizeof(float), \
            .ne0=(out_width), .ne1=(int32_t)rows, .r2=1, .r3=1 \
        }
        rust_star_q8_mm_args shared_gate_args = RUST_STAR_FFN_Q8_MM_ARGS(
            n_embd, ffn_mid, weights->ffn.shared_gate_bytes);
        rust_star_q8_mm_args shared_up_args = RUST_STAR_FFN_Q8_MM_ARGS(
            n_embd, ffn_mid, weights->ffn.shared_up_bytes);
        rust_star_q8_mm_args shared_down_args = RUST_STAR_FFN_Q8_MM_ARGS(
            ffn_mid, n_embd, weights->ffn.shared_down_bytes);
#undef RUST_STAR_FFN_Q8_MM_ARGS
        rust_star_glu_args shared_swiglu_args = {
            .ne00=(int32_t)(rows*ffn_mid),
            .nb01=(uint64_t)rows*ffn_mid*sizeof(float),
            .ne10=(int32_t)(rows*ffn_mid),
            .nb11=(uint64_t)rows*ffn_mid*sizeof(float),
            .ne0=(int32_t)(rows*ffn_mid),
            .nb1=(uint64_t)rows*ffn_mid*sizeof(float),
            .i00=0, .i10=0, .alpha=1.0f, .limit=10.0f,
        };
        rust_star_hc_expand_args ffn_hc_post_args = attention_hc_args;
        ffn_hc_post_args.has_add = 1;
        if (!position_buffer ||
            (continue_layer2_compressors && !compressed_position_buffer))
            return fail_with_message(error, error_bytes,
            @"failed to allocate prefill layer-0 KV position buffer");

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-0 boundary command encoder");

        [encoder setComputePipelineState:context.getRowsF16Pipeline];
        [encoder setBytes:&get_rows length:sizeof(get_rows) atIndex:0];
        [encoder setBuffer:model_buffers[0] offset:inner[0] atIndex:1];
        [encoder setBuffer:token_buffer offset:0 atIndex:2];
        [encoder setBuffer:embedding_buffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(4u*rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.repeatF32Pipeline];
        [encoder setBytes:&repeat length:sizeof(repeat) atIndex:0];
        [encoder setBuffer:embedding_buffer offset:0 atIndex:1];
        [encoder setBuffer:hc_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(n_hc,rows,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.rmsNormF32Pipeline];
        [encoder setBytes:&hc_norm length:sizeof(hc_norm) atIndex:0];
        [encoder setBuffer:hc_buffer offset:0 atIndex:1];
        [encoder setBuffer:hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:hc_buffer offset:0 atIndex:3];
        [encoder setBuffer:flat_hc_buffer offset:0 atIndex:4];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16PrefillPipeline];
        [encoder setBytes:&hc_mm length:sizeof(hc_mm) atIndex:0];
        [encoder setBuffer:model_buffers[1] offset:inner[1] atIndex:1];
        [encoder setBuffer:flat_hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:mix_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.hcIngressPipeline];
        [encoder setBytes:&hc length:sizeof(hc) atIndex:0];
        [encoder setBuffer:mix_buffer offset:0 atIndex:1];
        [encoder setBuffer:model_buffers[2] offset:inner[2] atIndex:2];
        [encoder setBuffer:model_buffers[3] offset:inner[3] atIndex:3];
        [encoder setBuffer:hc_buffer offset:0 atIndex:4];
        [encoder setBuffer:split_buffer offset:0 atIndex:5];
        [encoder setBuffer:collapsed_buffer offset:0 atIndex:6];
        [encoder setBuffer:model_buffers[4] offset:inner[4] atIndex:7];
        [encoder setBuffer:attn_buffer offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:(n_embd+4u+32u)*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

#define RUST_STAR_ENCODE_LAYER0_Q8(args, weight_index, input, output, out_width) do { \
        [encoder setComputePipelineState:context.q8PrefillPipeline]; \
        [encoder setBytes:&(args) length:sizeof(args) atIndex:0]; \
        [encoder setBuffer:model_buffers[(weight_index)] offset:inner[(weight_index)] atIndex:1]; \
        [encoder setBuffer:(input) offset:0 atIndex:2]; \
        [encoder setBuffer:(output) offset:0 atIndex:3]; \
        [encoder setThreadgroupMemoryLength:6144u atIndex:0]; \
        [encoder dispatchThreadgroups:MTLSizeMake(1u,(out_width)/64u,1u) \
             threadsPerThreadgroup:MTLSizeMake(128,1,1)]; \
    } while (0)
        RUST_STAR_ENCODE_LAYER0_Q8(q_a_args, 5, attn_buffer, q_buffer, q_rank);
        RUST_STAR_ENCODE_LAYER0_Q8(kv_args, 7, attn_buffer, kv_raw_buffer, kv_dim);

        [encoder setComputePipelineState:context.qkvNormPipeline];
        [encoder setBytes:&qkv_norm_args length:sizeof(qkv_norm_args) atIndex:0];
        [encoder setBuffer:q_buffer offset:0 atIndex:1];
        [encoder setBuffer:model_buffers[6] offset:inner[6] atIndex:2];
        [encoder setBuffer:q_norm_buffer offset:0 atIndex:3];
        [encoder setBuffer:kv_raw_buffer offset:0 atIndex:4];
        [encoder setBuffer:model_buffers[8] offset:inner[8] atIndex:5];
        [encoder setBuffer:kv_norm_buffer offset:0 atIndex:6];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,2,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        RUST_STAR_ENCODE_LAYER0_Q8(q_b_args, 9, q_norm_buffer, q_raw_buffer, q_dim);
#undef RUST_STAR_ENCODE_LAYER0_Q8
        [encoder endEncoding];

        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (!blit) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-0 Q snapshot encoder");
        [blit copyFromBuffer:q_raw_buffer sourceOffset:0 toBuffer:q_cur_buffer
            destinationOffset:0 size:q_bytes];
        [blit copyFromBuffer:kv_norm_buffer sourceOffset:0 toBuffer:kv_norm_snapshot_buffer
            destinationOffset:0 size:kv_bytes];
        [blit endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-0 Q RoPE encoder");
        [encoder setComputePipelineState:context.headNormRopePipeline];
        [encoder setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
        [encoder setBuffer:q_cur_buffer offset:0 atIndex:1];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(64,rows,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:context.ropeTailPipeline];
        [encoder setBytes:&kv_rope_args length:sizeof(kv_rope_args) atIndex:0];
        [encoder setBuffer:kv_norm_buffer offset:0 atIndex:1];
        [encoder setBuffer:position_buffer offset:0 atIndex:2];
        [encoder setBuffer:kv_norm_buffer offset:0 atIndex:3];
        [encoder setBuffer:kv_norm_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1,rows,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];

        blit = [command blitCommandEncoder];
        if (!blit) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-0 KV RoPE snapshot encoder");
        [blit copyFromBuffer:kv_norm_buffer sourceOffset:0 toBuffer:kv_rope_buffer
            destinationOffset:0 size:kv_bytes];
        [blit endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-0 KV storage encoder");
        [encoder setComputePipelineState:context.compressorFp8Pipeline];
        [encoder setBytes:&kv_fp8_args length:sizeof(kv_fp8_args) atIndex:0];
        [encoder setBuffer:kv_norm_buffer offset:0 atIndex:1];
        [encoder setBuffer:kv_norm_buffer offset:0 atIndex:2];
        [encoder setThreadgroupMemoryLength:64u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        uint32_t kv_elements = rows*kv_dim;
        const NSUInteger conversion_groups = (kv_elements + 1023u)/1024u;
        [encoder setComputePipelineState:context.cpyF32F16Pipeline];
        [encoder setBytes:&kv_elements length:sizeof(kv_elements) atIndex:0];
        [encoder setBuffer:kv_norm_buffer offset:0 atIndex:1];
        [encoder setBuffer:kv_half_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(conversion_groups,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:context.cpyF16F32Pipeline];
        [encoder setBytes:&kv_elements length:sizeof(kv_elements) atIndex:0];
        [encoder setBuffer:kv_half_buffer offset:0 atIndex:1];
        [encoder setBuffer:raw_cache_buffer
                     offset:raw_cache_target_row*kv_dim*sizeof(float) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(conversion_groups,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];

        blit = [command blitCommandEncoder];
        if (!blit) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-0 full-KV assembly encoder");
        [blit copyFromBuffer:kv_norm_buffer sourceOffset:0
                    toBuffer:full_kv_buffer destinationOffset:full_kv_prefix_bytes
                        size:kv_bytes];
        [blit endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-0 attention encoder");
        uint32_t full_kv_elements = prefill_rows*kv_dim;
        const NSUInteger full_conversion_groups =
            (full_kv_elements + 1023u)/1024u;
        [encoder setComputePipelineState:context.cpyF32F16Pipeline];
        [encoder setBytes:&full_kv_elements length:sizeof(full_kv_elements) atIndex:0];
        [encoder setBuffer:full_kv_buffer offset:0 atIndex:1];
        [encoder setBuffer:full_kv_half_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(full_conversion_groups,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [encoder setComputePipelineState:context.flashBlkPipeline];
        [encoder setBytes:&attention_blk_args length:sizeof(attention_blk_args) atIndex:0];
        [encoder setBuffer:attention_mask_buffer offset:0 atIndex:1];
        [encoder setBuffer:attention_block_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(32,4,1)
             threadsPerThreadgroup:MTLSizeMake(32,1,1)];

        [encoder setComputePipelineState:context.flashNonvecPipeline];
        [encoder setBytes:&attention_args length:sizeof(attention_args) atIndex:0];
        [encoder setBuffer:q_cur_buffer offset:0 atIndex:1];
        [encoder setBuffer:full_kv_half_buffer offset:0 atIndex:2];
        [encoder setBuffer:full_kv_half_buffer offset:0 atIndex:3];
        [encoder setBuffer:attention_mask_buffer offset:0 atIndex:4];
        [encoder setBuffer:model_buffers[10] offset:inner[10] atIndex:5];
        [encoder setBuffer:attention_pad_buffer offset:0 atIndex:6];
        [encoder setBuffer:attention_block_buffer offset:0 atIndex:7];
        [encoder setBuffer:attention_output_buffer offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:28672u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(4,n_head,1)
             threadsPerThreadgroup:MTLSizeMake(32,8,1)];
        [encoder endEncoding];

        blit = [command blitCommandEncoder];
        if (!blit) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-0 attention snapshot encoder");
        [blit copyFromBuffer:attention_output_buffer sourceOffset:0
                    toBuffer:attention_back_buffer destinationOffset:0 size:q_bytes];
        [blit endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-0 inverse-RoPE encoder");
        [encoder setComputePipelineState:context.ropeTailPipeline];
        [encoder setBytes:&attention_inverse_args
                   length:sizeof(attention_inverse_args) atIndex:0];
        [encoder setBuffer:attention_back_buffer offset:0 atIndex:1];
        [encoder setBuffer:position_buffer offset:0 atIndex:2];
        [encoder setBuffer:attention_back_buffer offset:0 atIndex:3];
        [encoder setBuffer:attention_back_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(n_head,rows,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [encoder setComputePipelineState:context.attentionOutputBatchMapPipeline];
        [encoder setBytes:&attention_output_map_args
                   length:sizeof(attention_output_map_args) atIndex:0];
        [encoder setBuffer:attention_group_ids_buffer offset:0 atIndex:1];
        [encoder setBuffer:attention_group_map_buffer offset:0 atIndex:2];
        [encoder setBuffer:attention_group_map_buffer offset:32u atIndex:3];
        [encoder setBuffer:attention_group_map_buffer offset:1056u atIndex:4];
        [encoder setThreadgroupMemoryLength:128u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(8,1,1)];

        [encoder setComputePipelineState:context.attentionOutputBatchLowPipeline];
        [encoder setBytes:&attention_output_low_args
                   length:sizeof(attention_output_low_args) atIndex:0];
        [encoder setBuffer:model_buffers[11] offset:inner[11] atIndex:1];
        [encoder setBuffer:attention_back_buffer offset:0 atIndex:2];
        [encoder setBuffer:attention_group_map_buffer offset:0 atIndex:3];
        [encoder setBuffer:attention_group_map_buffer offset:32u atIndex:4];
        [encoder setBuffer:attention_low_buffer offset:0 atIndex:5];
        [encoder setBuffer:attention_group_map_buffer offset:1056u atIndex:6];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(16,16,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.q8PrefillPipeline];
        [encoder setBytes:&attention_output_args
                   length:sizeof(attention_output_args) atIndex:0];
        [encoder setBuffer:model_buffers[12] offset:inner[12] atIndex:1];
        [encoder setBuffer:attention_low_buffer offset:0 atIndex:2];
        [encoder setBuffer:attention_out_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:6144u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,64,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.hcExpand4Pipeline];
        [encoder setBytes:&attention_hc_args length:sizeof(attention_hc_args) atIndex:0];
        [encoder setBuffer:attention_out_buffer offset:0 atIndex:1];
        [encoder setBuffer:hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:split_buffer offset:n_hc*sizeof(float) atIndex:3];
        [encoder setBuffer:split_buffer offset:2u*n_hc*sizeof(float) atIndex:4];
        [encoder setBuffer:attention_out_buffer offset:0 atIndex:5];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((rows*n_embd+255u)/256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [encoder setComputePipelineState:context.rmsNormF32Pipeline];
        [encoder setBytes:&hc_norm length:sizeof(hc_norm) atIndex:0];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:1];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:3];
        [encoder setBuffer:ffn_flat_hc_buffer offset:0 atIndex:4];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16PrefillPipeline];
        [encoder setBytes:&ffn_hc_mm length:sizeof(ffn_hc_mm) atIndex:0];
        [encoder setBuffer:model_buffers[13] offset:inner[13] atIndex:1];
        [encoder setBuffer:ffn_flat_hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:ffn_mix_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.hcIngressPipeline];
        [encoder setBytes:&ffn_hc_args length:sizeof(ffn_hc_args) atIndex:0];
        [encoder setBuffer:ffn_mix_buffer offset:0 atIndex:1];
        [encoder setBuffer:model_buffers[14] offset:inner[14] atIndex:2];
        [encoder setBuffer:model_buffers[15] offset:inner[15] atIndex:3];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:4];
        [encoder setBuffer:ffn_split_buffer offset:0 atIndex:5];
        [encoder setBuffer:ffn_cur_buffer offset:0 atIndex:6];
        [encoder setBuffer:model_buffers[16] offset:inner[16] atIndex:7];
        [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:(n_embd+4u+32u)*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16PrefillPipeline];
        [encoder setBytes:&router_mm length:sizeof(router_mm) atIndex:0];
        [encoder setBuffer:model_buffers[17] offset:inner[17] atIndex:1];
        [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:2];
        [encoder setBuffer:router_logits_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,4,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.routerSoftplusBatchPipeline];
        [encoder setBuffer:router_logits_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_probs_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(rows*n_expert,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:context.routerSqrtBatchPipeline];
        [encoder setBuffer:router_probs_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_probs_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(rows*n_expert,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:context.routerHashRowsBatchPipeline];
        [encoder setBuffer:model_buffers[18] offset:inner[18] atIndex:0];
        [encoder setBuffer:token_buffer offset:0 atIndex:1];
        [encoder setBuffer:router_selected_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(rows,1,1)];
        [encoder setComputePipelineState:context.routerGatherWeightsBatchPipeline];
        [encoder setBuffer:router_probs_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_selected_buffer offset:0 atIndex:1];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(n_used,rows,1)
             threadsPerThreadgroup:MTLSizeMake(n_used,1,1)];

        [encoder setComputePipelineState:context.compressorSumRowsPipeline];
        [encoder setBytes:&router_sum_args length:sizeof(router_sum_args) atIndex:0];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:1];
        [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:2];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(n_used,1,1)];
        [encoder setComputePipelineState:context.routerClampSumsBatchPipeline];
        [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(rows,1,1)];
        [encoder setComputePipelineState:context.routerDivideBatchPipeline];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:1];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(n_used,rows,1)
             threadsPerThreadgroup:MTLSizeMake(n_used,1,1)];
        [encoder setComputePipelineState:context.routerScaleBatchPipeline];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(rows*n_used,1,1)
             threadsPerThreadgroup:MTLSizeMake(192,1,1)];

        [encoder setComputePipelineState:context.routedBatchMapPipeline];
        [encoder setBytes:&routed_map_args length:sizeof(routed_map_args) atIndex:0];
        [encoder setBuffer:router_selected_buffer offset:0 atIndex:1];
        [encoder setBuffer:routed_map_buffer offset:0 atIndex:2];
        [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:3];
        [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:4];
        [encoder setThreadgroupMemoryLength:n_expert*n_used*sizeof(uint16_t) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(n_expert,1,1)];

        [encoder setComputePipelineState:context.routedBatchPairSwigluPipeline];
        [encoder setBytes:&routed_gate_args length:sizeof(routed_gate_args) atIndex:0];
        [encoder setBytes:&routed_activation_args length:sizeof(routed_activation_args) atIndex:1];
        [encoder setBuffer:model_buffers[19] offset:inner[19] atIndex:2];
        [encoder setBuffer:model_buffers[20] offset:inner[20] atIndex:3];
        [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:4];
        [encoder setBuffer:routed_map_buffer offset:0 atIndex:5];
        [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:6];
        [encoder setBuffer:routed_mid_buffer offset:0 atIndex:7];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:8];
        [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:9];
        [encoder setThreadgroupMemoryLength:16384u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(routed_map_work_cap,32,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.routedBatchDownPipeline];
        [encoder setBytes:&routed_down_args length:sizeof(routed_down_args) atIndex:0];
        [encoder setBuffer:model_buffers[21] offset:inner[21] atIndex:1];
        [encoder setBuffer:routed_mid_buffer offset:0 atIndex:2];
        [encoder setBuffer:routed_map_buffer offset:0 atIndex:3];
        [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:4];
        [encoder setBuffer:routed_experts_buffer offset:0 atIndex:5];
        [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:6];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(routed_map_work_cap,64,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [encoder setComputePipelineState:context.routedBatchSumPipeline];
        [encoder setBytes:&routed_sum_args length:sizeof(routed_sum_args) atIndex:0];
        [encoder setBuffer:routed_experts_buffer offset:0 atIndex:1];
        [encoder setBuffer:routed_out_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

#define RUST_STAR_ENCODE_FFN_Q8(args, weight_index, input, output, out_width) do { \
        [encoder setComputePipelineState:context.q8PrefillPipeline]; \
        [encoder setBytes:&(args) length:sizeof(args) atIndex:0]; \
        [encoder setBuffer:model_buffers[(weight_index)] offset:inner[(weight_index)] atIndex:1]; \
        [encoder setBuffer:(input) offset:0 atIndex:2]; \
        [encoder setBuffer:(output) offset:0 atIndex:3]; \
        [encoder setThreadgroupMemoryLength:6144u atIndex:0]; \
        [encoder dispatchThreadgroups:MTLSizeMake(1u,(out_width)/64u,1u) \
             threadsPerThreadgroup:MTLSizeMake(128,1,1)]; \
    } while (0)
        RUST_STAR_ENCODE_FFN_Q8(shared_gate_args, 22, ffn_norm_buffer,
            shared_gate_buffer, ffn_mid);
        RUST_STAR_ENCODE_FFN_Q8(shared_up_args, 23, ffn_norm_buffer,
            shared_up_buffer, ffn_mid);
        [encoder setComputePipelineState:context.sharedSwigluBatchPipeline];
        [encoder setBytes:&shared_swiglu_args length:sizeof(shared_swiglu_args) atIndex:0];
        [encoder setBuffer:shared_gate_buffer offset:0 atIndex:1];
        [encoder setBuffer:shared_up_buffer offset:0 atIndex:2];
        [encoder setBuffer:shared_mid_buffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((rows*ffn_mid+255u)/256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        RUST_STAR_ENCODE_FFN_Q8(shared_down_args, 24, shared_mid_buffer,
            shared_out_buffer, n_embd);
#undef RUST_STAR_ENCODE_FFN_Q8

        [encoder setComputePipelineState:context.hcExpand4Pipeline];
        [encoder setBytes:&ffn_hc_post_args length:sizeof(ffn_hc_post_args) atIndex:0];
        [encoder setBuffer:routed_out_buffer offset:0 atIndex:1];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:ffn_split_buffer offset:n_hc*sizeof(float) atIndex:3];
        [encoder setBuffer:ffn_split_buffer offset:2u*n_hc*sizeof(float) atIndex:4];
        [encoder setBuffer:shared_out_buffer offset:0 atIndex:5];
        [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((rows*n_embd+255u)/256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        if (continue_layer1) {
            [encoder setComputePipelineState:context.rmsNormF32Pipeline];
            [encoder setBytes:&hc_norm length:sizeof(hc_norm) atIndex:0];
            [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:1];
            [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:2];
            [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:3];
            [encoder setBuffer:next_flat_hc_buffer offset:0 atIndex:4];
            [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

            [encoder setComputePipelineState:context.f16PrefillPipeline];
            [encoder setBytes:&hc_mm length:sizeof(hc_mm) atIndex:0];
            [encoder setBuffer:model_buffers[25] offset:inner[25] atIndex:1];
            [encoder setBuffer:next_flat_hc_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_mix_buffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:8192u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [encoder setComputePipelineState:context.hcIngressPipeline];
            [encoder setBytes:&hc length:sizeof(hc) atIndex:0];
            [encoder setBuffer:next_mix_buffer offset:0 atIndex:1];
            [encoder setBuffer:model_buffers[26] offset:inner[26] atIndex:2];
            [encoder setBuffer:model_buffers[27] offset:inner[27] atIndex:3];
            [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:4];
            [encoder setBuffer:next_split_buffer offset:0 atIndex:5];
            [encoder setBuffer:next_cur_buffer offset:0 atIndex:6];
            [encoder setBuffer:model_buffers[28] offset:inner[28] atIndex:7];
            [encoder setBuffer:next_norm_buffer offset:0 atIndex:8];
            [encoder setThreadgroupMemoryLength:(n_embd+4u+32u)*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

            [encoder setComputePipelineState:context.q8PrefillPipeline];
            [encoder setBytes:&q_a_args length:sizeof(q_a_args) atIndex:0];
            [encoder setBuffer:model_buffers[29] offset:inner[29] atIndex:1];
            [encoder setBuffer:next_norm_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_q_lora_buffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:6144u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,16,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        }
        if (complete_layer1) {
    #define RUST_STAR_ENCODE_LAYER1_Q8(args, weight_index, input, output, out_width) do { \
            [encoder setComputePipelineState:context.q8PrefillPipeline]; \
            [encoder setBytes:&(args) length:sizeof(args) atIndex:0]; \
            [encoder setBuffer:model_buffers[(weight_index)] offset:inner[(weight_index)] atIndex:1]; \
            [encoder setBuffer:(input) offset:0 atIndex:2]; \
            [encoder setBuffer:(output) offset:0 atIndex:3]; \
            [encoder setThreadgroupMemoryLength:6144u atIndex:0]; \
            [encoder dispatchThreadgroups:MTLSizeMake(1u,(out_width)/64u,1u) \
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)]; \
        } while (0)
            RUST_STAR_ENCODE_LAYER1_Q8(kv_args, 31, next_norm_buffer, next_kv_raw_buffer, kv_dim);

            [encoder setComputePipelineState:context.qkvNormPipeline];
            [encoder setBytes:&qkv_norm_args length:sizeof(qkv_norm_args) atIndex:0];
            [encoder setBuffer:next_q_lora_buffer offset:0 atIndex:1];
            [encoder setBuffer:model_buffers[30] offset:inner[30] atIndex:2];
            [encoder setBuffer:next_q_norm_buffer offset:0 atIndex:3];
            [encoder setBuffer:next_kv_raw_buffer offset:0 atIndex:4];
            [encoder setBuffer:model_buffers[32] offset:inner[32] atIndex:5];
            [encoder setBuffer:next_kv_norm_buffer offset:0 atIndex:6];
            [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,2,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            RUST_STAR_ENCODE_LAYER1_Q8(q_b_args, 33, next_q_norm_buffer, next_q_raw_buffer, q_dim);
    #undef RUST_STAR_ENCODE_LAYER1_Q8
            [encoder endEncoding];

            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (!blit) return fail_with_message(error, error_bytes,
                @"failed to create prefill layer-1 Q snapshot encoder");
            [blit copyFromBuffer:next_q_raw_buffer sourceOffset:0 toBuffer:next_q_cur_buffer
                destinationOffset:0 size:q_bytes];
            [blit copyFromBuffer:next_kv_norm_buffer sourceOffset:0 toBuffer:next_kv_norm_snapshot_buffer
                destinationOffset:0 size:kv_bytes];
            [blit endEncoding];

            encoder = [command computeCommandEncoder];
            if (!encoder) return fail_with_message(error, error_bytes,
                @"failed to create prefill layer-1 Q RoPE encoder");
            [encoder setComputePipelineState:context.headNormRopePipeline];
            [encoder setBytes:&rope_args length:sizeof(rope_args) atIndex:0];
            [encoder setBuffer:next_q_cur_buffer offset:0 atIndex:1];
            [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(64,rows,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder setComputePipelineState:context.ropeTailPipeline];
            [encoder setBytes:&kv_rope_args length:sizeof(kv_rope_args) atIndex:0];
            [encoder setBuffer:next_kv_norm_buffer offset:0 atIndex:1];
            [encoder setBuffer:position_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_kv_norm_buffer offset:0 atIndex:3];
            [encoder setBuffer:next_kv_norm_buffer offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(1,rows,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder endEncoding];

            blit = [command blitCommandEncoder];
            if (!blit) return fail_with_message(error, error_bytes,
                @"failed to create prefill layer-1 KV RoPE snapshot encoder");
            [blit copyFromBuffer:next_kv_norm_buffer sourceOffset:0 toBuffer:next_kv_rope_buffer
                destinationOffset:0 size:kv_bytes];
            [blit endEncoding];

            encoder = [command computeCommandEncoder];
            if (!encoder) return fail_with_message(error, error_bytes,
                @"failed to create prefill layer-1 KV storage encoder");
            [encoder setComputePipelineState:context.compressorFp8Pipeline];
            [encoder setBytes:&kv_fp8_args length:sizeof(kv_fp8_args) atIndex:0];
            [encoder setBuffer:next_kv_norm_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_kv_norm_buffer offset:0 atIndex:2];
            [encoder setThreadgroupMemoryLength:64u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            uint32_t kv_elements = rows*kv_dim;
            const NSUInteger conversion_groups = (kv_elements + 1023u)/1024u;
            [encoder setComputePipelineState:context.cpyF32F16Pipeline];
            [encoder setBytes:&kv_elements length:sizeof(kv_elements) atIndex:0];
            [encoder setBuffer:next_kv_norm_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_kv_half_buffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(conversion_groups,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder setComputePipelineState:context.cpyF16F32Pipeline];
            [encoder setBytes:&kv_elements length:sizeof(kv_elements) atIndex:0];
            [encoder setBuffer:next_kv_half_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_raw_cache_buffer
                         offset:raw_cache_target_row*kv_dim*sizeof(float) atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(conversion_groups,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder endEncoding];

            blit = [command blitCommandEncoder];
            if (!blit) return fail_with_message(error, error_bytes,
                @"failed to create prefill layer-1 full-KV assembly encoder");
            [blit copyFromBuffer:next_kv_norm_buffer sourceOffset:0
                        toBuffer:next_full_kv_buffer destinationOffset:full_kv_prefix_bytes
                            size:kv_bytes];
            [blit endEncoding];

            encoder = [command computeCommandEncoder];
            if (!encoder) return fail_with_message(error, error_bytes,
                @"failed to create prefill layer-1 attention encoder");
            uint32_t full_kv_elements = prefill_rows*kv_dim;
            const NSUInteger full_conversion_groups =
                (full_kv_elements + 1023u)/1024u;
            [encoder setComputePipelineState:context.cpyF32F16Pipeline];
            [encoder setBytes:&full_kv_elements length:sizeof(full_kv_elements) atIndex:0];
            [encoder setBuffer:next_full_kv_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_full_kv_half_buffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(full_conversion_groups,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            [encoder setComputePipelineState:context.flashBlkPipeline];
            [encoder setBytes:&attention_blk_args length:sizeof(attention_blk_args) atIndex:0];
            [encoder setBuffer:attention_mask_buffer offset:0 atIndex:1];
            [encoder setBuffer:attention_block_buffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(32,4,1)
                 threadsPerThreadgroup:MTLSizeMake(32,1,1)];

            [encoder setComputePipelineState:context.flashNonvecPipeline];
            [encoder setBytes:&attention_args length:sizeof(attention_args) atIndex:0];
            [encoder setBuffer:next_q_cur_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_full_kv_half_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_full_kv_half_buffer offset:0 atIndex:3];
            [encoder setBuffer:attention_mask_buffer offset:0 atIndex:4];
            [encoder setBuffer:model_buffers[34] offset:inner[34] atIndex:5];
            [encoder setBuffer:attention_pad_buffer offset:0 atIndex:6];
            [encoder setBuffer:attention_block_buffer offset:0 atIndex:7];
            [encoder setBuffer:next_attention_output_buffer offset:0 atIndex:8];
            [encoder setThreadgroupMemoryLength:28672u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(4,n_head,1)
                 threadsPerThreadgroup:MTLSizeMake(32,8,1)];
            [encoder endEncoding];

            blit = [command blitCommandEncoder];
            if (!blit) return fail_with_message(error, error_bytes,
                @"failed to create prefill layer-1 attention snapshot encoder");
            [blit copyFromBuffer:next_attention_output_buffer sourceOffset:0
                        toBuffer:next_attention_back_buffer destinationOffset:0 size:q_bytes];
            [blit endEncoding];

            encoder = [command computeCommandEncoder];
            if (!encoder) return fail_with_message(error, error_bytes,
                @"failed to create prefill layer-1 inverse-RoPE encoder");
            [encoder setComputePipelineState:context.ropeTailPipeline];
            [encoder setBytes:&attention_inverse_args
                       length:sizeof(attention_inverse_args) atIndex:0];
            [encoder setBuffer:next_attention_back_buffer offset:0 atIndex:1];
            [encoder setBuffer:position_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_attention_back_buffer offset:0 atIndex:3];
            [encoder setBuffer:next_attention_back_buffer offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(n_head,rows,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            [encoder setComputePipelineState:context.attentionOutputBatchMapPipeline];
            [encoder setBytes:&attention_output_map_args
                       length:sizeof(attention_output_map_args) atIndex:0];
            [encoder setBuffer:attention_group_ids_buffer offset:0 atIndex:1];
            [encoder setBuffer:attention_group_map_buffer offset:0 atIndex:2];
            [encoder setBuffer:attention_group_map_buffer offset:32u atIndex:3];
            [encoder setBuffer:attention_group_map_buffer offset:1056u atIndex:4];
            [encoder setThreadgroupMemoryLength:128u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                 threadsPerThreadgroup:MTLSizeMake(8,1,1)];

            [encoder setComputePipelineState:context.attentionOutputBatchLowPipeline];
            [encoder setBytes:&attention_output_low_args
                       length:sizeof(attention_output_low_args) atIndex:0];
            [encoder setBuffer:model_buffers[35] offset:inner[35] atIndex:1];
            [encoder setBuffer:next_attention_back_buffer offset:0 atIndex:2];
            [encoder setBuffer:attention_group_map_buffer offset:0 atIndex:3];
            [encoder setBuffer:attention_group_map_buffer offset:32u atIndex:4];
            [encoder setBuffer:next_attention_low_buffer offset:0 atIndex:5];
            [encoder setBuffer:attention_group_map_buffer offset:1056u atIndex:6];
            [encoder setThreadgroupMemoryLength:8192u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(16,16,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [encoder setComputePipelineState:context.q8PrefillPipeline];
            [encoder setBytes:&attention_output_args
                       length:sizeof(attention_output_args) atIndex:0];
            [encoder setBuffer:model_buffers[36] offset:inner[36] atIndex:1];
            [encoder setBuffer:next_attention_low_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_attention_out_buffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:6144u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,64,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [encoder setComputePipelineState:context.hcExpand4Pipeline];
            [encoder setBytes:&attention_hc_args length:sizeof(attention_hc_args) atIndex:0];
            [encoder setBuffer:next_attention_out_buffer offset:0 atIndex:1];
            [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_split_buffer offset:n_hc*sizeof(float) atIndex:3];
            [encoder setBuffer:next_split_buffer offset:2u*n_hc*sizeof(float) atIndex:4];
            [encoder setBuffer:next_attention_out_buffer offset:0 atIndex:5];
            [encoder setBuffer:next_after_attention_hc_buffer offset:0 atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake((rows*n_embd+255u)/256u,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            [encoder setComputePipelineState:context.rmsNormF32Pipeline];
            [encoder setBytes:&hc_norm length:sizeof(hc_norm) atIndex:0];
            [encoder setBuffer:next_after_attention_hc_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_after_attention_hc_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_after_attention_hc_buffer offset:0 atIndex:3];
            [encoder setBuffer:next_ffn_flat_hc_buffer offset:0 atIndex:4];
            [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

            [encoder setComputePipelineState:context.f16PrefillPipeline];
            [encoder setBytes:&ffn_hc_mm length:sizeof(ffn_hc_mm) atIndex:0];
            [encoder setBuffer:model_buffers[37] offset:inner[37] atIndex:1];
            [encoder setBuffer:next_ffn_flat_hc_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_ffn_mix_buffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:8192u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [encoder setComputePipelineState:context.hcIngressPipeline];
            [encoder setBytes:&ffn_hc_args length:sizeof(ffn_hc_args) atIndex:0];
            [encoder setBuffer:next_ffn_mix_buffer offset:0 atIndex:1];
            [encoder setBuffer:model_buffers[38] offset:inner[38] atIndex:2];
            [encoder setBuffer:model_buffers[39] offset:inner[39] atIndex:3];
            [encoder setBuffer:next_after_attention_hc_buffer offset:0 atIndex:4];
            [encoder setBuffer:next_ffn_split_buffer offset:0 atIndex:5];
            [encoder setBuffer:next_ffn_cur_buffer offset:0 atIndex:6];
            [encoder setBuffer:model_buffers[40] offset:inner[40] atIndex:7];
            [encoder setBuffer:next_ffn_norm_buffer offset:0 atIndex:8];
            [encoder setThreadgroupMemoryLength:(n_embd+4u+32u)*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

            [encoder setComputePipelineState:context.f16PrefillPipeline];
            [encoder setBytes:&router_mm length:sizeof(router_mm) atIndex:0];
            [encoder setBuffer:model_buffers[41] offset:inner[41] atIndex:1];
            [encoder setBuffer:next_ffn_norm_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_router_logits_buffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:8192u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,4,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [encoder setComputePipelineState:context.routerSoftplusBatchPipeline];
            [encoder setBuffer:next_router_logits_buffer offset:0 atIndex:0];
            [encoder setBuffer:next_router_probs_buffer offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(rows*n_expert,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder setComputePipelineState:context.routerSqrtBatchPipeline];
            [encoder setBuffer:next_router_probs_buffer offset:0 atIndex:0];
            [encoder setBuffer:next_router_probs_buffer offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(rows*n_expert,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder setComputePipelineState:context.routerHashRowsBatchPipeline];
            [encoder setBuffer:model_buffers[42] offset:inner[42] atIndex:0];
            [encoder setBuffer:token_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_router_selected_buffer offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(rows,1,1)];
            [encoder setComputePipelineState:context.routerGatherWeightsBatchPipeline];
            [encoder setBuffer:next_router_probs_buffer offset:0 atIndex:0];
            [encoder setBuffer:next_router_selected_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_router_weights_buffer offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(n_used,rows,1)
                 threadsPerThreadgroup:MTLSizeMake(n_used,1,1)];

            [encoder setComputePipelineState:context.compressorSumRowsPipeline];
            [encoder setBytes:&router_sum_args length:sizeof(router_sum_args) atIndex:0];
            [encoder setBuffer:next_router_weights_buffer offset:0 atIndex:1];
            [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:2];
            [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(n_used,1,1)];
            [encoder setComputePipelineState:context.routerClampSumsBatchPipeline];
            [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:0];
            [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(rows,1,1)];
            [encoder setComputePipelineState:context.routerDivideBatchPipeline];
            [encoder setBuffer:next_router_weights_buffer offset:0 atIndex:0];
            [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_router_weights_buffer offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(n_used,rows,1)
                 threadsPerThreadgroup:MTLSizeMake(n_used,1,1)];
            [encoder setComputePipelineState:context.routerScaleBatchPipeline];
            [encoder setBuffer:next_router_weights_buffer offset:0 atIndex:0];
            [encoder setBuffer:next_router_weights_buffer offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(rows*n_used,1,1)
                 threadsPerThreadgroup:MTLSizeMake(192,1,1)];

            [encoder setComputePipelineState:context.routedBatchMapPipeline];
            [encoder setBytes:&routed_map_args length:sizeof(routed_map_args) atIndex:0];
            [encoder setBuffer:next_router_selected_buffer offset:0 atIndex:1];
            [encoder setBuffer:routed_map_buffer offset:0 atIndex:2];
            [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:3];
            [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:4];
            [encoder setThreadgroupMemoryLength:n_expert*n_used*sizeof(uint16_t) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                 threadsPerThreadgroup:MTLSizeMake(n_expert,1,1)];

            [encoder setComputePipelineState:context.routedBatchPairSwigluPipeline];
            [encoder setBytes:&routed_gate_args length:sizeof(routed_gate_args) atIndex:0];
            [encoder setBytes:&routed_activation_args length:sizeof(routed_activation_args) atIndex:1];
            [encoder setBuffer:model_buffers[43] offset:inner[43] atIndex:2];
            [encoder setBuffer:model_buffers[44] offset:inner[44] atIndex:3];
            [encoder setBuffer:next_ffn_norm_buffer offset:0 atIndex:4];
            [encoder setBuffer:routed_map_buffer offset:0 atIndex:5];
            [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:6];
            [encoder setBuffer:next_routed_mid_buffer offset:0 atIndex:7];
            [encoder setBuffer:next_router_weights_buffer offset:0 atIndex:8];
            [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:9];
            [encoder setThreadgroupMemoryLength:16384u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(routed_map_work_cap,32,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [encoder setComputePipelineState:context.routedBatchDownPipeline];
            [encoder setBytes:&routed_down_args length:sizeof(routed_down_args) atIndex:0];
            [encoder setBuffer:model_buffers[45] offset:inner[45] atIndex:1];
            [encoder setBuffer:next_routed_mid_buffer offset:0 atIndex:2];
            [encoder setBuffer:routed_map_buffer offset:0 atIndex:3];
            [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:4];
            [encoder setBuffer:routed_experts_buffer offset:0 atIndex:5];
            [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:6];
            [encoder setThreadgroupMemoryLength:8192u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(routed_map_work_cap,64,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [encoder setComputePipelineState:context.routedBatchSumPipeline];
            [encoder setBytes:&routed_sum_args length:sizeof(routed_sum_args) atIndex:0];
            [encoder setBuffer:routed_experts_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_routed_out_buffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    #define RUST_STAR_ENCODE_LAYER1_FFN_Q8(args, weight_index, input, output, out_width) do { \
            [encoder setComputePipelineState:context.q8PrefillPipeline]; \
            [encoder setBytes:&(args) length:sizeof(args) atIndex:0]; \
            [encoder setBuffer:model_buffers[(weight_index)] offset:inner[(weight_index)] atIndex:1]; \
            [encoder setBuffer:(input) offset:0 atIndex:2]; \
            [encoder setBuffer:(output) offset:0 atIndex:3]; \
            [encoder setThreadgroupMemoryLength:6144u atIndex:0]; \
            [encoder dispatchThreadgroups:MTLSizeMake(1u,(out_width)/64u,1u) \
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)]; \
        } while (0)
            RUST_STAR_ENCODE_LAYER1_FFN_Q8(shared_gate_args, 46, next_ffn_norm_buffer,
                shared_gate_buffer, ffn_mid);
            RUST_STAR_ENCODE_LAYER1_FFN_Q8(shared_up_args, 47, next_ffn_norm_buffer,
                shared_up_buffer, ffn_mid);
            [encoder setComputePipelineState:context.sharedSwigluBatchPipeline];
            [encoder setBytes:&shared_swiglu_args length:sizeof(shared_swiglu_args) atIndex:0];
            [encoder setBuffer:shared_gate_buffer offset:0 atIndex:1];
            [encoder setBuffer:shared_up_buffer offset:0 atIndex:2];
            [encoder setBuffer:shared_mid_buffer offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((rows*ffn_mid+255u)/256u,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            RUST_STAR_ENCODE_LAYER1_FFN_Q8(shared_down_args, 48, shared_mid_buffer,
                next_shared_out_buffer, n_embd);
    #undef RUST_STAR_ENCODE_LAYER1_FFN_Q8

            [encoder setComputePipelineState:context.hcExpand4Pipeline];
            [encoder setBytes:&ffn_hc_post_args length:sizeof(ffn_hc_post_args) atIndex:0];
            [encoder setBuffer:next_routed_out_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_after_attention_hc_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_ffn_split_buffer offset:n_hc*sizeof(float) atIndex:3];
            [encoder setBuffer:next_ffn_split_buffer offset:2u*n_hc*sizeof(float) atIndex:4];
            [encoder setBuffer:next_shared_out_buffer offset:0 atIndex:5];
            [encoder setBuffer:next_after_ffn_hc_buffer offset:0 atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake((rows*n_embd+255u)/256u,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        }
        if (continue_layer2) {
            [encoder setComputePipelineState:context.rmsNormF32Pipeline];
            [encoder setBytes:&hc_norm length:sizeof(hc_norm) atIndex:0];
            [encoder setBuffer:next_after_ffn_hc_buffer offset:0 atIndex:1];
            [encoder setBuffer:next_after_ffn_hc_buffer offset:0 atIndex:2];
            [encoder setBuffer:next_after_ffn_hc_buffer offset:0 atIndex:3];
            [encoder setBuffer:layer2_flat_hc_buffer offset:0 atIndex:4];
            [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

            [encoder setComputePipelineState:context.f16PrefillPipeline];
            [encoder setBytes:&hc_mm length:sizeof(hc_mm) atIndex:0];
            [encoder setBuffer:model_buffers[49] offset:inner[49] atIndex:1];
            [encoder setBuffer:layer2_flat_hc_buffer offset:0 atIndex:2];
            [encoder setBuffer:layer2_mix_buffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:8192u atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [encoder setComputePipelineState:context.hcIngressPipeline];
            [encoder setBytes:&hc length:sizeof(hc) atIndex:0];
            [encoder setBuffer:layer2_mix_buffer offset:0 atIndex:1];
            [encoder setBuffer:model_buffers[50] offset:inner[50] atIndex:2];
            [encoder setBuffer:model_buffers[51] offset:inner[51] atIndex:3];
            [encoder setBuffer:next_after_ffn_hc_buffer offset:0 atIndex:4];
            [encoder setBuffer:layer2_split_buffer offset:0 atIndex:5];
            [encoder setBuffer:layer2_cur_buffer offset:0 atIndex:6];
            [encoder setBuffer:model_buffers[52] offset:inner[52] atIndex:7];
            [encoder setBuffer:layer2_norm_buffer offset:0 atIndex:8];
            [encoder setThreadgroupMemoryLength:(n_embd+4u+32u)*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                 threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

#define RUST_STAR_ENCODE_LAYER2_Q8(args, weight_index, input, output, out_width) do { \
            [encoder setComputePipelineState:context.q8PrefillPipeline]; \
            [encoder setBytes:&(args) length:sizeof(args) atIndex:0]; \
            [encoder setBuffer:model_buffers[(weight_index)] offset:inner[(weight_index)] atIndex:1]; \
            [encoder setBuffer:(input) offset:0 atIndex:2]; \
            [encoder setBuffer:(output) offset:0 atIndex:3]; \
            [encoder setThreadgroupMemoryLength:6144u atIndex:0]; \
            [encoder dispatchThreadgroups:MTLSizeMake(1u,(out_width)/64u,1u) \
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)]; \
        } while (0)
            RUST_STAR_ENCODE_LAYER2_Q8(q_a_args, 53, layer2_norm_buffer,
                layer2_q_lora_buffer, q_rank);
            RUST_STAR_ENCODE_LAYER2_Q8(kv_args, 55, layer2_norm_buffer,
                layer2_kv_raw_buffer, kv_dim);
#undef RUST_STAR_ENCODE_LAYER2_Q8

            [encoder setComputePipelineState:context.qkvNormPipeline];
            [encoder setBytes:&qkv_norm_args length:sizeof(qkv_norm_args) atIndex:0];
            [encoder setBuffer:layer2_q_lora_buffer offset:0 atIndex:1];
            [encoder setBuffer:model_buffers[54] offset:inner[54] atIndex:2];
            [encoder setBuffer:layer2_q_norm_buffer offset:0 atIndex:3];
            [encoder setBuffer:layer2_kv_raw_buffer offset:0 atIndex:4];
            [encoder setBuffer:model_buffers[56] offset:inner[56] atIndex:5];
            [encoder setBuffer:layer2_kv_norm_buffer offset:0 atIndex:6];
            [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(rows,2,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            if (continue_layer2_compressors) {
#define RUST_STAR_ENCODE_COMPRESSOR_F16(args, weight_index, output, out_width) do { \
                [encoder setComputePipelineState:context.f16AlignedPrefillPipeline]; \
                [encoder setBytes:&(args) length:sizeof(args) atIndex:0]; \
                [encoder setBuffer:model_buffers[(weight_index)] \
                           offset:inner[(weight_index)] atIndex:1]; \
                [encoder setBuffer:layer2_norm_buffer offset:0 atIndex:2]; \
                [encoder setBuffer:(output) offset:0 atIndex:3]; \
                [encoder setThreadgroupMemoryLength:6144u atIndex:0]; \
                [encoder dispatchThreadgroups:MTLSizeMake(1u,(out_width)/64u,1u) \
                     threadsPerThreadgroup:MTLSizeMake(128,1,1)]; \
            } while (0)
                RUST_STAR_ENCODE_COMPRESSOR_F16(attn_compressor_kv_args, 58,
                    layer2_attn_projected_kv_buffer, attn_compressor_width);
                RUST_STAR_ENCODE_COMPRESSOR_F16(attn_compressor_gate_args, 59,
                    layer2_attn_projected_score_buffer, attn_compressor_width);
                RUST_STAR_ENCODE_COMPRESSOR_F16(indexer_compressor_kv_args, 62,
                    layer2_indexer_projected_kv_buffer,
                    indexer_compressor_width);
                RUST_STAR_ENCODE_COMPRESSOR_F16(indexer_compressor_gate_args, 63,
                    layer2_indexer_projected_score_buffer,
                    indexer_compressor_width);
#undef RUST_STAR_ENCODE_COMPRESSOR_F16
                if (!encode_projected_compressor_batch_ratio4(
                        context, encoder,
                        layer2_norm_buffer,
                        (NSUInteger)(rows-compressor_ratio)*n_embd*sizeof(float),
                        model_buffers[58], inner[58],
                        model_buffers[59], inner[59],
                        layer2_attn_projected_kv_buffer,
                        layer2_attn_projected_score_buffer,
                        layer2_attn_softmax_buffer,
                        layer2_attn_packed_kv_buffer,
                        layer2_attn_packed_score_buffer,
                        model_buffers[57], inner[57],
                        model_buffers[60], inner[60],
                        layer2_attn_state_kv_buffer,
                        layer2_attn_state_score_buffer,
                        layer2_attn_compressed_buffer,
                        attn_compressed_prefix_bytes,
                        compressed_position_buffer,
                        attn_compressor_width, attn_compressor_head,
                        position_start, rows, NO) ||
                    !encode_projected_compressor_batch_ratio4(
                        context, encoder,
                        layer2_norm_buffer,
                        (NSUInteger)(rows-compressor_ratio)*n_embd*sizeof(float),
                        model_buffers[62], inner[62],
                        model_buffers[63], inner[63],
                        layer2_indexer_projected_kv_buffer,
                        layer2_indexer_projected_score_buffer,
                        layer2_indexer_softmax_buffer,
                        layer2_indexer_packed_kv_buffer,
                        layer2_indexer_packed_score_buffer,
                        model_buffers[61], inner[61],
                        model_buffers[64], inner[64],
                        layer2_indexer_state_kv_buffer,
                        layer2_indexer_state_score_buffer,
                        layer2_indexer_compressed_buffer,
                        indexer_compressed_prefix_bytes,
                        compressed_position_buffer,
                        indexer_compressor_width, indexer_compressor_head,
                        position_start, rows, YES)) {
                    return fail_with_message(error, error_bytes,
                        @"failed to encode prefill layer-2 compressor batches");
                }
            }
            if (continue_layer2_kv_state) {
                [encoder endEncoding];

                blit = [command blitCommandEncoder];
                if (!blit) return fail_with_message(error, error_bytes,
                    @"failed to create prefill layer-2 KVnorm snapshot encoder");
                [blit copyFromBuffer:layer2_kv_norm_buffer sourceOffset:0
                            toBuffer:layer2_kv_norm_snapshot_buffer destinationOffset:0
                                size:kv_bytes];
                [blit endEncoding];

                encoder = [command computeCommandEncoder];
                if (!encoder) return fail_with_message(error, error_bytes,
                    @"failed to create prefill layer-2 KV RoPE encoder");
                [encoder setComputePipelineState:context.ropeTailPipeline];
                [encoder setBytes:&layer2_kv_rope_args
                           length:sizeof(layer2_kv_rope_args) atIndex:0];
                [encoder setBuffer:layer2_kv_norm_buffer offset:0 atIndex:1];
                [encoder setBuffer:position_buffer offset:0 atIndex:2];
                [encoder setBuffer:layer2_kv_norm_buffer offset:0 atIndex:3];
                [encoder setBuffer:layer2_kv_norm_buffer offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(1,rows,1)
                     threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [encoder endEncoding];

                blit = [command blitCommandEncoder];
                if (!blit) return fail_with_message(error, error_bytes,
                    @"failed to create prefill layer-2 KVrope snapshot encoder");
                [blit copyFromBuffer:layer2_kv_norm_buffer sourceOffset:0
                            toBuffer:layer2_kv_rope_buffer destinationOffset:0
                                size:kv_bytes];
                [blit endEncoding];

                encoder = [command computeCommandEncoder];
                if (!encoder) return fail_with_message(error, error_bytes,
                    @"failed to create prefill layer-2 KV finalization encoder");
                [encoder setComputePipelineState:context.compressorFp8Pipeline];
                [encoder setBytes:&kv_fp8_args length:sizeof(kv_fp8_args) atIndex:0];
                [encoder setBuffer:layer2_kv_norm_buffer offset:0 atIndex:1];
                [encoder setBuffer:layer2_kv_norm_buffer offset:0 atIndex:2];
                [encoder setThreadgroupMemoryLength:64u*sizeof(float) atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
                     threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                [encoder endEncoding];

                blit = [command blitCommandEncoder];
                if (!blit) return fail_with_message(error, error_bytes,
                    @"failed to create prefill layer-2 live-KV append encoder");
                [blit copyFromBuffer:layer2_kv_norm_buffer sourceOffset:0
                            toBuffer:layer2_full_kv_buffer
                   destinationOffset:full_kv_prefix_bytes size:kv_bytes];
                [blit copyFromBuffer:layer2_q_norm_buffer sourceOffset:0
                            toBuffer:layer2_full_q_norm_buffer
                   destinationOffset:(NSUInteger)position_start*q_rank*sizeof(float)
                                size:q_rank_bytes];
                [blit copyFromBuffer:next_after_ffn_hc_buffer sourceOffset:0
                            toBuffer:layer2_input_hc_buffer
                   destinationOffset:(NSUInteger)position_start*hc_dim*sizeof(float)
                                size:hc_bytes];
                [blit copyFromBuffer:layer2_split_buffer sourceOffset:0
                            toBuffer:layer2_attn_split_buffer
                   destinationOffset:(NSUInteger)position_start*mix_hc*sizeof(float)
                                size:mix_bytes];
                [blit copyFromBuffer:token_buffer sourceOffset:0
                            toBuffer:layer2_tokens_buffer
                   destinationOffset:(NSUInteger)position_start*sizeof(uint32_t)
                                size:token_bytes];
                [blit endEncoding];
            } else {
                [encoder endEncoding];
            }
        } else {
            [encoder endEncoding];
        }

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        if (continue_layer2_compressors) {
            memset((uint8_t *)layer2_attn_state_kv_buffer.contents +
                       4u*attn_compressor_width*sizeof(float),
                   0, 4u*attn_compressor_width*sizeof(float));
            memset((uint8_t *)layer2_indexer_state_kv_buffer.contents +
                       4u*indexer_compressor_width*sizeof(float),
                   0, 4u*indexer_compressor_width*sizeof(float));
            float *attn_scores = layer2_attn_state_score_buffer.contents;
            for (NSUInteger index = 4u*attn_compressor_width;
                 index < compressor_state_rows*attn_compressor_width; index++) {
                attn_scores[index] = -INFINITY;
            }
            float *indexer_scores =
                layer2_indexer_state_score_buffer.contents;
            for (NSUInteger index = 4u*indexer_compressor_width;
                 index < compressor_state_rows*indexer_compressor_width; index++) {
                indexer_scores[index] = -INFINITY;
            }
        }
        if (retain_kv_state || consume_kv_state) {
            context.prefillLayer0FullKv = full_kv_buffer;
            context.prefillLayer1FullKv = next_full_kv_buffer;
            if (continue_layer2_kv_state) {
                context.prefillLayer2FullKv = layer2_full_kv_buffer;
                context.prefillLayer2FullQNorm = layer2_full_q_norm_buffer;
                context.prefillLayer2InputHc = layer2_input_hc_buffer;
                context.prefillLayer2AttnSplit = layer2_attn_split_buffer;
                context.prefillLayer2Tokens = layer2_tokens_buffer;
            }
            if (continue_layer2_compressors) {
                context.prefillLayer2AttnCompressed =
                    layer2_attn_compressed_buffer;
                context.prefillLayer2AttnStateKv = layer2_attn_state_kv_buffer;
                context.prefillLayer2AttnStateScore =
                    layer2_attn_state_score_buffer;
                context.prefillLayer2IndexerCompressed =
                    layer2_indexer_compressed_buffer;
                context.prefillLayer2IndexerStateKv =
                    layer2_indexer_state_kv_buffer;
                context.prefillLayer2IndexerStateScore =
                    layer2_indexer_state_score_buffer;
            }
            context.prefillKvRows = position_start + rows;
        }
        memcpy(hc_collapsed, collapsed_buffer.contents, attn_bytes);
        memcpy(attn_norm, attn_buffer.contents, attn_bytes);
        memcpy(q_lora, q_buffer.contents, q_rank_bytes);
        memcpy(q_lora_norm, q_norm_buffer.contents, q_rank_bytes);
        memcpy(kv_raw, kv_raw_buffer.contents, kv_bytes);
        memcpy(kv_norm, kv_norm_snapshot_buffer.contents, kv_bytes);
        memcpy(q_raw, q_raw_buffer.contents, q_bytes);
        memcpy(q_cur, q_cur_buffer.contents, q_bytes);
        memcpy(kv_rope, kv_rope_buffer.contents, kv_bytes);
        memcpy(kv_cur, kv_norm_buffer.contents, kv_bytes);
        memcpy(raw_cache, raw_cache_buffer.contents, raw_cache_bytes);
        memcpy(full_kv, full_kv_buffer.contents, full_kv_bytes);
        memcpy(attention_output, attention_output_buffer.contents, q_bytes);
        memcpy(attention_back, attention_back_buffer.contents, q_bytes);
        memcpy(attention_low, attention_low_buffer.contents, attention_low_bytes);
        memcpy(attention_out, attention_out_buffer.contents, attention_out_bytes);
        memcpy(after_attention_hc, after_attention_hc_buffer.contents, hc_bytes);
        memcpy(ffn_cur, ffn_cur_buffer.contents, attn_bytes);
        memcpy(ffn_norm, ffn_norm_buffer.contents, attn_bytes);
        memcpy(router_logits, router_logits_buffer.contents, router_bytes);
        memcpy(router_probs, router_probs_buffer.contents, router_bytes);
        memcpy(router_selected, router_selected_buffer.contents, selected_bytes);
        memcpy(router_weights, router_weights_buffer.contents, selected_bytes);
        const __fp16 *routed_mid_half = routed_mid_buffer.contents;
        for (NSUInteger index = 0; index < rows*n_used*ffn_mid; index++) {
            routed_mid[index] = (float)routed_mid_half[index];
        }
        memcpy(routed_out, routed_out_buffer.contents, attn_bytes);
        memcpy(shared_out, shared_out_buffer.contents, attn_bytes);
        memcpy(after_ffn_hc, after_ffn_hc_buffer.contents, hc_bytes);
        if (continue_layer1) {
            memcpy(next_hc_collapsed, next_cur_buffer.contents, attn_bytes);
            memcpy(next_attn_norm, next_norm_buffer.contents, attn_bytes);
            memcpy(next_q_lora, next_q_lora_buffer.contents, q_rank_bytes);
        }
        if (complete_layer1) {
            memcpy(next_outputs->q_lora_norm, next_q_norm_buffer.contents,
                q_rank_bytes);
            memcpy(next_outputs->kv_norm, next_kv_norm_snapshot_buffer.contents,
                kv_bytes);
            memcpy(next_outputs->q_cur, next_q_cur_buffer.contents, q_bytes);
            memcpy(next_outputs->kv_rope, next_kv_rope_buffer.contents, kv_bytes);
            memcpy(next_outputs->kv_cur, next_kv_norm_buffer.contents, kv_bytes);
            memcpy(next_outputs->attention_output,
                next_attention_output_buffer.contents, q_bytes);
            memcpy(next_outputs->attention_back,
                next_attention_back_buffer.contents, q_bytes);
            memcpy(next_outputs->attention_low,
                next_attention_low_buffer.contents, attention_low_bytes);
            memcpy(next_outputs->attention_out,
                next_attention_out_buffer.contents, attention_out_bytes);
            memcpy(next_outputs->after_attention_hc,
                next_after_attention_hc_buffer.contents, hc_bytes);
            memcpy(next_outputs->ffn_cur, next_ffn_cur_buffer.contents, attn_bytes);
            memcpy(next_outputs->ffn_norm, next_ffn_norm_buffer.contents, attn_bytes);
            memcpy(next_outputs->router_logits,
                next_router_logits_buffer.contents, router_bytes);
            memcpy(next_outputs->router_probs,
                next_router_probs_buffer.contents, router_bytes);
            memcpy(next_outputs->router_selected,
                next_router_selected_buffer.contents, selected_bytes);
            memcpy(next_outputs->router_weights,
                next_router_weights_buffer.contents, selected_bytes);
            const __fp16 *next_routed_mid_half = next_routed_mid_buffer.contents;
            for (NSUInteger index = 0; index < rows*n_used*ffn_mid; index++) {
                next_outputs->routed_mid[index] = (float)next_routed_mid_half[index];
            }
            memcpy(next_outputs->routed_out, next_routed_out_buffer.contents,
                attn_bytes);
            memcpy(next_outputs->shared_out, next_shared_out_buffer.contents,
                attn_bytes);
            memcpy(next_outputs->after_ffn_hc,
                next_after_ffn_hc_buffer.contents, hc_bytes);
        }
        if (continue_layer2) {
            memcpy(layer2_kv_norm_output,
                continue_layer2_kv_state ? layer2_kv_norm_snapshot_buffer.contents :
                    layer2_kv_norm_buffer.contents,
                kv_bytes);
        }
        if (continue_layer2_kv_state) {
            memcpy(layer2_kv_rope_output, layer2_kv_rope_buffer.contents, kv_bytes);
            memcpy(layer2_kv_cur_output, layer2_kv_norm_buffer.contents, kv_bytes);
        }
        if (continue_layer2_compressors) {
            memcpy(layer2_attn_compressed_output,
                (const uint8_t *)layer2_attn_compressed_buffer.contents +
                    attn_compressed_prefix_bytes,
                compressor_tile_rows*attn_compressor_head*sizeof(float));
            memcpy(layer2_indexer_compressed_output,
                (const uint8_t *)layer2_indexer_compressed_buffer.contents +
                    indexer_compressed_prefix_bytes,
                compressor_tile_rows*indexer_compressor_head*sizeof(float));
            memcpy(layer2_attn_state_kv_output,
                layer2_attn_state_kv_buffer.contents, attn_state_bytes);
            memcpy(layer2_attn_state_score_output,
                layer2_attn_state_score_buffer.contents, attn_state_bytes);
            memcpy(layer2_indexer_state_kv_output,
                layer2_indexer_state_kv_buffer.contents, indexer_state_bytes);
            memcpy(layer2_indexer_state_score_output,
                layer2_indexer_state_score_buffer.contents, indexer_state_bytes);
        }

        uint32_t pointer_matches = 0;
        for (uint32_t index = 0; index < model_range_count; index++) {
            pointer_matches += matches[index] ? 1u : 0u;
        }
        result->rows = rows;
        result->input_elements_per_row = n_embd;
        result->q_lora_elements_per_row = q_rank;
        result->kv_elements_per_row = kv_dim;
        result->q_elements_per_row = q_dim;
        result->dispatches = continue_layer2_compressors ?
            (position_start == 2016u ? 122u : 118u) :
            (continue_layer2_kv_state ? 92u : (continue_layer2 ? 90u :
            (complete_layer1 ? 84u : (continue_layer1 ? 47u : 43u))));
        result->wrapped_model_ranges = model_range_count;
        result->pointer_matches = pointer_matches;
        result->position_start = position_start;
        result->raw_cache_rows = raw_cache_rows;
        result->raw_cache_target_row = raw_cache_target_row;
        result->raw_cache_guard_rows = raw_cache_target_row;
        result->kv_state_mode = kv_state_mode;
        result->wall_ms = wall_end-wall_start;
        result->gpu_ms = gpu_elapsed_ms(command);
        return 1;
    }
}

int rust_star_metal_run_prefill_layer2_attention(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    const rust_star_metal_prefill_layer2_attention_weights *weights,
    float *attention_output,
    float *after_attention_hc,
    float *after_ffn_hc,
    float *ffn_cur_final_tile,
    float *ffn_norm_final_tile,
    int32_t *router_selected_final_tile,
    float *router_weights_final_tile,
    float *routed_out_final_tile,
    float *shared_out_final_tile,
    float *layer3_hc_attn_pre,
    float *layer3_attn_norm,
    float *layer3_q_lora,
    rust_star_metal_prefill_layer2_attention_result *result,
    char *error,
    size_t error_bytes)
{
    if (!opaque_context || !model_mapping || !weights || !attention_output ||
        !after_attention_hc || !after_ffn_hc || !ffn_cur_final_tile ||
        !ffn_norm_final_tile || !router_selected_final_tile ||
        !router_weights_final_tile || !routed_out_final_tile ||
        !shared_out_final_tile || !layer3_hc_attn_pre || !layer3_attn_norm ||
        !layer3_q_lora || !result) {
        return fail_with_message(error, error_bytes,
            @"prefill layer-2 attention received a null input");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        enum {
            rows = 2048, q_rank = 1024, q_dim = 32768,
            n_head = 64, head_dim = 512,
            raw_rows = 2048, compressed_rows = 512,
            key_rows = raw_rows + compressed_rows,
            attention_window = 128, compressor_ratio = 4,
            output_rank = 1024, n_embd = 4096,
        };
        if (context.prefillKvRows != rows ||
            !context.prefillLayer2FullQNorm || !context.prefillLayer2FullKv ||
            !context.prefillLayer2AttnCompressed || !context.prefillLayer2InputHc ||
            !context.prefillLayer2AttnSplit) {
            return fail_with_message(error, error_bytes,
                @"prefill layer-2 attention requires a complete retained 2K boundary");
        }
        if (weights->q_b_bytes !=
                (uint64_t)q_dim*((uint64_t)q_rank/32u)*34u ||
            weights->attn_sinks_bytes != n_head*sizeof(float) ||
            weights->attn_output_a_bytes !=
                8ull*output_rank*((uint64_t)n_embd/32u)*34u ||
            weights->attn_output_b_bytes !=
                (uint64_t)n_embd*((uint64_t)(8u*output_rank)/32u)*34u ||
            weights->ffn.hc_fn_bytes != 16384ull*24ull*sizeof(uint16_t) ||
            weights->ffn.hc_scale_bytes != 3u*sizeof(float) ||
            weights->ffn.hc_base_bytes != 24u*sizeof(float) ||
            weights->ffn.norm_bytes != n_embd*sizeof(float) ||
            weights->ffn.router_gate_bytes !=
                (uint64_t)n_embd*256u*sizeof(uint16_t) ||
            weights->ffn.router_hash_bytes != 129280ull*6ull*sizeof(int32_t) ||
            weights->ffn.routed_gate_bytes != 256ull*2048ull*1056ull ||
            weights->ffn.routed_up_bytes != 256ull*2048ull*1056ull ||
            weights->ffn.routed_down_bytes != 256ull*n_embd*672ull ||
            weights->ffn.shared_gate_bytes != 2048ull*(n_embd/32u)*34u ||
            weights->ffn.shared_up_bytes != 2048ull*(n_embd/32u)*34u ||
            weights->ffn.shared_down_bytes != n_embd*(2048u/32u)*34u ||
            weights->layer3_ingress.hc_fn_bytes != 16384ull*24ull*sizeof(uint16_t) ||
            weights->layer3_ingress.hc_scale_bytes != 3u*sizeof(float) ||
            weights->layer3_ingress.hc_base_bytes != 24u*sizeof(float) ||
            weights->layer3_ingress.norm_bytes != n_embd*sizeof(float) ||
            weights->layer3_ingress.q_a_bytes !=
                1024ull*((uint64_t)n_embd/32u)*34u) {
            return fail_with_message(error, error_bytes,
                @"prefill layer-2 attention tensor shapes are invalid");
        }
        if (!ensure_attention_ingress_pipelines(context, error, error_bytes) ||
            !ensure_moe_output_pipelines(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));

        uint64_t offsets[21] = {
            weights->q_b_offset, weights->attn_sinks_offset,
            weights->attn_output_a_offset, weights->attn_output_b_offset,
            weights->ffn.hc_fn_offset, weights->ffn.hc_scale_offset,
            weights->ffn.hc_base_offset, weights->ffn.norm_offset,
            weights->ffn.router_gate_offset, weights->ffn.router_hash_offset,
            weights->ffn.routed_gate_offset, weights->ffn.routed_up_offset,
            weights->ffn.routed_down_offset, weights->ffn.shared_gate_offset,
            weights->ffn.shared_up_offset, weights->ffn.shared_down_offset,
            weights->layer3_ingress.hc_fn_offset,
            weights->layer3_ingress.hc_scale_offset,
            weights->layer3_ingress.hc_base_offset,
            weights->layer3_ingress.norm_offset,
            weights->layer3_ingress.q_a_offset,
        };
        uint64_t sizes[21] = {
            weights->q_b_bytes, weights->attn_sinks_bytes,
            weights->attn_output_a_bytes, weights->attn_output_b_bytes,
            weights->ffn.hc_fn_bytes, weights->ffn.hc_scale_bytes,
            weights->ffn.hc_base_bytes, weights->ffn.norm_bytes,
            weights->ffn.router_gate_bytes, weights->ffn.router_hash_bytes,
            weights->ffn.routed_gate_bytes, weights->ffn.routed_up_bytes,
            weights->ffn.routed_down_bytes, weights->ffn.shared_gate_bytes,
            weights->ffn.shared_up_bytes, weights->ffn.shared_down_bytes,
            weights->layer3_ingress.hc_fn_bytes,
            weights->layer3_ingress.hc_scale_bytes,
            weights->layer3_ingress.hc_base_bytes,
            weights->layer3_ingress.norm_bytes,
            weights->layer3_ingress.q_a_bytes,
        };
        id<MTLBuffer> model_buffers[21] = { nil };
        NSUInteger inner[21] = { 0 };
        BOOL matches[21] = { NO };
        for (uint32_t index = 0; index < 21u; index++) {
            model_buffers[index] = wrap_model_range(
                context, model_mapping, model_bytes, offsets[index], sizes[index],
                &inner[index], &matches[index], error, error_bytes);
            if (!model_buffers[index]) return 0;
        }

        const NSUInteger q_bytes = (NSUInteger)rows*q_dim*sizeof(float);
        const NSUInteger staged_kv_bytes =
            (NSUInteger)key_rows*head_dim*sizeof(uint16_t);
        const NSUInteger mask_bytes =
            (NSUInteger)rows*key_rows*sizeof(uint16_t);
        const NSUInteger block_bytes = 40u*256u;
        const NSUInteger attention_low_bytes =
            (NSUInteger)rows*8u*output_rank*sizeof(float);
        const NSUInteger output_bytes = (NSUInteger)rows*n_embd*sizeof(float);
        const NSUInteger hc_bytes = (NSUInteger)rows*4u*n_embd*sizeof(float);
        const NSUInteger group_ids_bytes = (NSUInteger)rows*8u*sizeof(int32_t);
        const NSUInteger tpe_bytes = 8u*sizeof(int32_t);
        const NSUInteger hids_bytes = (NSUInteger)8u*rows*sizeof(int32_t);
        const NSUInteger work_offset = (tpe_bytes + hids_bytes + 7u) & ~7u;
        const NSUInteger work_cap =
            ((NSUInteger)8u*rows + 31u*8u + 31u)/32u;
        const NSUInteger group_map_bytes = work_offset + 8u + work_cap*8u;

#define RUST_STAR_NEW_L2_ATTN_BUFFER(name, bytes) \
        id<MTLBuffer> name = [context.device newBufferWithLength:(bytes) \
            options:MTLResourceStorageModeShared]
        RUST_STAR_NEW_L2_ATTN_BUFFER(q_buffer, q_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(staged_kv_buffer, staged_kv_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(mask_buffer, mask_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(block_buffer, block_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(pad_buffer, 1u);
        RUST_STAR_NEW_L2_ATTN_BUFFER(heads_buffer, q_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(attention_low_buffer, attention_low_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(output_buffer, output_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(after_attention_hc_buffer, hc_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(group_ids_buffer, group_ids_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(group_map_buffer, group_map_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(position_buffer, rows*sizeof(int32_t));
        const NSUInteger ffn_mix_bytes = (NSUInteger)rows*24u*sizeof(float);
        const NSUInteger ffn_router_bytes = (NSUInteger)rows*256u*sizeof(float);
        const NSUInteger ffn_selected_bytes = (NSUInteger)rows*6u*sizeof(int32_t);
        const NSUInteger ffn_mid_bytes = (NSUInteger)rows*2048u*sizeof(float);
        const NSUInteger routed_mid_bytes =
            (NSUInteger)rows*6u*2048u*sizeof(uint16_t);
        const NSUInteger routed_experts_bytes =
            (NSUInteger)rows*6u*n_embd*sizeof(float);
        const NSUInteger routed_map_tpe_bytes = 256u*sizeof(int32_t);
        const NSUInteger routed_map_ids_bytes = (NSUInteger)256u*rows*sizeof(int32_t);
        const NSUInteger routed_map_work_offset =
            (routed_map_tpe_bytes+routed_map_ids_bytes+7u)&~7u;
        const NSUInteger routed_map_work_cap =
            ((NSUInteger)rows*6u+31u*256u+31u)/32u;
        RUST_STAR_NEW_L2_ATTN_BUFFER(ffn_flat_hc_buffer, hc_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(ffn_mix_buffer, ffn_mix_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(ffn_split_buffer, ffn_mix_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(ffn_cur_buffer, output_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(ffn_norm_buffer, output_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(router_logits_buffer, ffn_router_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(router_probs_buffer, ffn_router_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(router_selected_buffer, ffn_selected_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(router_weights_buffer, ffn_selected_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(router_weight_sums_buffer, rows*sizeof(float));
        RUST_STAR_NEW_L2_ATTN_BUFFER(routed_map_buffer,
            routed_map_work_offset+8u+routed_map_work_cap*8u);
        RUST_STAR_NEW_L2_ATTN_BUFFER(routed_mid_buffer, routed_mid_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(routed_experts_buffer, routed_experts_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(routed_out_buffer, output_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(shared_gate_buffer, ffn_mid_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(shared_up_buffer, ffn_mid_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(shared_mid_buffer, ffn_mid_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(shared_out_buffer, output_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(after_ffn_hc_buffer, hc_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(layer3_flat_hc_buffer, hc_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(layer3_mix_buffer, ffn_mix_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(layer3_split_buffer, ffn_mix_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(layer3_cur_buffer, output_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(layer3_norm_buffer, output_bytes);
        RUST_STAR_NEW_L2_ATTN_BUFFER(layer3_q_lora_buffer,
            (NSUInteger)rows*1024u*sizeof(float));
#undef RUST_STAR_NEW_L2_ATTN_BUFFER
        if (!q_buffer || !staged_kv_buffer || !mask_buffer || !block_buffer ||
            !pad_buffer || !heads_buffer || !attention_low_buffer || !output_buffer ||
            !group_ids_buffer || !group_map_buffer || !position_buffer ||
            !after_attention_hc_buffer || !ffn_flat_hc_buffer ||
            !ffn_mix_buffer || !ffn_split_buffer || !ffn_cur_buffer ||
            !ffn_norm_buffer || !router_logits_buffer || !router_probs_buffer ||
            !router_selected_buffer || !router_weights_buffer ||
            !router_weight_sums_buffer || !routed_map_buffer ||
            !routed_mid_buffer || !routed_experts_buffer || !routed_out_buffer ||
            !shared_gate_buffer || !shared_up_buffer || !shared_mid_buffer ||
            !shared_out_buffer || !after_ffn_hc_buffer || !layer3_flat_hc_buffer ||
            !layer3_mix_buffer || !layer3_split_buffer || !layer3_cur_buffer ||
            !layer3_norm_buffer || !layer3_q_lora_buffer) {
            return fail_with_message(error, error_bytes,
                @"failed to allocate prefill layer-2 attention buffers");
        }

        uint16_t *mask = mask_buffer.contents;
        for (uint32_t query = 0; query < rows; query++) {
            uint16_t *mask_row = mask + (NSUInteger)query*key_rows;
            for (uint32_t key = 0; key < raw_rows; key++) {
                const BOOL visible = key <= query && query-key < attention_window;
                mask_row[key] = visible ? 0u : 0xfc00u;
            }
            const uint32_t visible_compressed = (query+1u)/compressor_ratio;
            for (uint32_t key = 0; key < compressed_rows; key++) {
                mask_row[raw_rows+key] =
                    key < visible_compressed ? 0u : 0xfc00u;
            }
        }
        int32_t *positions = position_buffer.contents;
        int32_t *group_ids = group_ids_buffer.contents;
        for (uint32_t row = 0; row < rows; row++) {
            positions[row] = (int32_t)row;
            for (uint32_t group = 0; group < 8u; group++) {
                group_ids[(NSUInteger)row*8u+group] = (int32_t)group;
            }
        }

        rust_star_q8_mm_args q_b_args = {
            .ne00=q_rank, .ne02=1,
            .nb01=(q_rank/32u)*34u,
            .nb02=weights->q_b_bytes, .nb03=weights->q_b_bytes,
            .ne12=1, .nb10=sizeof(float),
            .nb11=q_rank*sizeof(float),
            .nb12=(uint64_t)rows*q_rank*sizeof(float),
            .nb13=(uint64_t)rows*q_rank*sizeof(float),
            .ne0=q_dim, .ne1=rows, .r2=1, .r3=1,
        };
        const float freq_scale = 1.0f/16.0f;
        const float attn_factor =
            1.0f/(1.0f + 0.1f*logf(1.0f/freq_scale));
        rust_star_head_norm_rope_args q_rope_args = {
            .n_head=n_head, .head_dim=head_dim, .head_dim4=head_dim/4,
            .n_dims=64, .n_ctx_orig=65536, .pos0=0, .inverse=0,
            .eps=1.0e-6f, .freq_base=160000.0f, .freq_scale=freq_scale,
            .ext_factor=1.0f, .attn_factor=attn_factor,
            .beta_fast=32.0f, .beta_slow=1.0f,
        };
        rust_star_flash_blk_args block_args = {
            .ne01=rows, .ne30=key_rows, .ne31=rows, .ne32=1, .ne33=1,
            .nb31=key_rows*sizeof(uint16_t),
            .nb32=mask_bytes, .nb33=mask_bytes,
        };
        const uint64_t head_f32_bytes = head_dim*sizeof(float);
        const uint64_t head_f16_bytes = head_dim*sizeof(uint16_t);
        rust_star_flash_vec_args attention_args = {
            .ne01=rows, .ne02=n_head, .ne03=1,
            .nb01=n_head*head_f32_bytes, .nb02=head_f32_bytes,
            .nb03=q_bytes,
            .ne11=key_rows, .ne_12_2=1, .ne_12_3=1, .ns10=head_dim,
            .nb11=head_f16_bytes, .nb12=staged_kv_bytes,
            .nb13=staged_kv_bytes, .ns20=head_dim,
            .nb21=head_f16_bytes, .nb22=staged_kv_bytes,
            .nb23=staged_kv_bytes,
            .ne31=rows, .ne32=1, .ne33=1,
            .nb31=key_rows*sizeof(uint16_t),
            .nb32=mask_bytes, .nb33=mask_bytes,
            .ne1=n_head, .ne2=rows, .ne3=1,
            .scale=1.0f/sqrtf((float)head_dim),
            .max_bias=0.0f, .m0=0.0f, .m1=0.0f,
            .n_head_log2=0, .logit_softcap=0.0f,
        };
        rust_star_rope_tail_args inverse_args = {
            .ne00=head_dim, .ne01=n_head, .ne02=rows, .ne03=1,
            .nb00=sizeof(float), .nb01=head_f32_bytes,
            .nb02=n_head*head_f32_bytes, .nb03=q_bytes,
            .nb0=sizeof(float), .nb1=head_f32_bytes,
            .nb2=n_head*head_f32_bytes, .nb3=q_bytes,
            .n_dims=64, .mode=0, .n_ctx_orig=65536, .inverse=1,
            .freq_base=160000.0f, .freq_scale=freq_scale,
            .ext_factor=1.0f, .attn_factor=attn_factor,
            .beta_fast=32.0f, .beta_slow=1.0f, .src2=false,
        };
        rust_star_q8_mm_id_map_args map_args = {
            .ne02=8, .ne10=n_embd, .ne11=8,
            .nb11=n_embd*sizeof(float),
            .nb12=8u*n_embd*sizeof(float),
            .ne21=rows, .ne20=8, .nb21=8u*sizeof(int32_t),
        };
        rust_star_q8_mm_id_args low_args = {
            .ne00=n_embd, .ne02=8,
            .nb01=(n_embd/32u)*34u,
            .nb02=(uint64_t)output_rank*((n_embd/32u)*34u),
            .nb03=weights->attn_output_a_bytes,
            .ne11=8, .nb10=sizeof(float),
            .nb11=n_embd*sizeof(float),
            .nb12=8u*n_embd*sizeof(float),
            .nb13=(uint64_t)rows*8u*n_embd*sizeof(float),
            .ne20=8, .ne21=rows, .ne0=output_rank, .ne1=8,
            .r2=1, .r3=1,
        };
        rust_star_q8_mm_args output_args = {
            .ne00=8u*output_rank, .ne02=1,
            .nb01=((8u*output_rank)/32u)*34u,
            .nb02=weights->attn_output_b_bytes,
            .nb03=weights->attn_output_b_bytes,
            .ne12=1, .nb10=sizeof(float),
            .nb11=8u*output_rank*sizeof(float),
            .nb12=(uint64_t)rows*8u*output_rank*sizeof(float),
            .nb13=(uint64_t)rows*8u*output_rank*sizeof(float),
            .ne0=n_embd, .ne1=rows, .r2=1, .r3=1,
        };
        rust_star_hc_expand_args attention_hc_args = {
            .n_embd=n_embd, .n_hc=4, .n_tokens=rows,
            .nb_block0=sizeof(float), .nb_block1=n_embd*sizeof(float),
            .nb_add0=sizeof(float), .nb_add1=n_embd*sizeof(float),
            .nb_res0=sizeof(float), .nb_res1=n_embd*sizeof(float),
            .nb_res2=4u*n_embd*sizeof(float),
            .nb_post0=sizeof(float), .nb_post1=24u*sizeof(float),
            .nb_comb0=sizeof(float), .nb_comb1=4u*sizeof(float),
            .nb_comb2=24u*sizeof(float),
            .nb0=sizeof(float), .nb1=n_embd*sizeof(float),
            .nb2=4u*n_embd*sizeof(float), .has_add=0,
        };
        rust_star_norm_args ffn_hc_norm_args = {
            .ne00=4u*n_embd, .ne00_t=n_embd,
            .nb1=4u*n_embd*sizeof(float), .nb2=hc_bytes, .nb3=hc_bytes,
            .eps=1.0e-6f,
            .nef1={rows,1,1}, .nef2={1,1,1}, .nef3={1,1,1},
            .nbf1={4u*n_embd*sizeof(float),4u*n_embd*sizeof(float),4u*n_embd*sizeof(float)},
            .nbf2={hc_bytes,4u*n_embd*sizeof(float),4u*n_embd*sizeof(float)},
            .nbf3={hc_bytes,4u*n_embd*sizeof(float),4u*n_embd*sizeof(float)},
        };
        rust_star_q8_mm_args ffn_hc_mm_args = {
            .ne00=4u*n_embd, .ne02=1,
            .nb01=4u*n_embd*sizeof(uint16_t),
            .nb02=4ull*n_embd*24u*sizeof(uint16_t),
            .nb03=4ull*n_embd*24u*sizeof(uint16_t), .ne12=1,
            .nb10=sizeof(float), .nb11=4u*n_embd*sizeof(float),
            .nb12=hc_bytes, .nb13=hc_bytes,
            .ne0=24, .ne1=rows, .r2=1, .r3=1,
        };
        rust_star_hc_ingress_args ffn_hc_ingress_args = {
            .n_embd=n_embd, .n_hc=4, .sinkhorn_iters=20,
            .n_rows=rows, .mix_hc=24,
            .nb_mix1=24u*sizeof(float), .nb_split1=24u*sizeof(float),
            .nb_x0=sizeof(float), .nb_x1=n_embd*sizeof(float),
            .nb_x2=4u*n_embd*sizeof(float),
            .nb0=sizeof(float), .nb1=n_embd*sizeof(float),
            .nb_norm1=n_embd*sizeof(float), .eps=1.0e-6f, .norm_eps=1.0e-6f,
        };
        rust_star_q8_mm_args router_args = {
            .ne00=n_embd, .ne02=1,
            .nb01=n_embd*sizeof(uint16_t),
            .nb02=(uint64_t)n_embd*256u*sizeof(uint16_t),
            .nb03=(uint64_t)n_embd*256u*sizeof(uint16_t), .ne12=1,
            .nb10=sizeof(float), .nb11=n_embd*sizeof(float),
            .nb12=(uint64_t)rows*n_embd*sizeof(float),
            .nb13=(uint64_t)rows*n_embd*sizeof(float),
            .ne0=256, .ne1=rows, .r2=1, .r3=1,
        };
        rust_star_sum_rows_args router_sum_args = {
            .ne00=6, .ne01=rows, .ne02=1, .ne03=1,
            .nb00=sizeof(float), .nb01=6u*sizeof(float),
            .nb02=(uint64_t)rows*6u*sizeof(float),
            .nb03=(uint64_t)rows*6u*sizeof(float),
            .ne0=1, .ne1=rows, .ne2=1, .ne3=1,
            .nb0=sizeof(float), .nb1=sizeof(float),
            .nb2=(uint64_t)rows*sizeof(float),
            .nb3=(uint64_t)rows*sizeof(float),
        };
        rust_star_q8_mm_id_map_args routed_map_args = {
            .ne02=256, .ne10=n_embd, .ne11=1,
            .nb11=n_embd*sizeof(float), .nb12=n_embd*sizeof(float),
            .ne21=rows, .ne20=6, .nb21=6u*sizeof(int32_t),
        };
        rust_star_q8_mm_id_args routed_gate_args = {
            .ne00=n_embd, .ne02=256,
            .nb01=1056, .nb02=2048ull*1056ull,
            .nb03=256ull*2048ull*1056ull,
            .ne11=1, .nb10=sizeof(float), .nb11=n_embd*sizeof(float),
            .nb12=n_embd*sizeof(float),
            .nb13=(uint64_t)rows*n_embd*sizeof(float),
            .ne20=6, .ne21=rows, .ne0=2048, .ne1=6,
            .r2=1, .r3=1, .tp_rank=0, .tp_world=0, .tp_expert_base=0,
        };
        rust_star_moe_swiglu_weight_args routed_activation_args = {
            .width=2048, .rows=rows*6u,
            .gate_row_stride=2048u*sizeof(float),
            .up_row_stride=2048u*sizeof(float),
            .mid_row_stride=2048u*sizeof(uint16_t),
            .weight_stride=sizeof(float), .write_clamped=0, .clamp_value=10.0f,
        };
        rust_star_q8_mm_id_args routed_down_args = {
            .ne00=2048, .ne02=256,
            .nb01=672, .nb02=(uint64_t)n_embd*672u,
            .nb03=256ull*n_embd*672u,
            .ne11=6, .nb10=sizeof(uint16_t),
            .nb11=2048u*sizeof(uint16_t),
            .nb12=6u*2048u*sizeof(uint16_t),
            .nb13=(uint64_t)rows*6u*2048u*sizeof(uint16_t),
            .ne20=6, .ne21=rows, .ne0=n_embd, .ne1=6,
            .r2=1, .r3=1, .tp_rank=0, .tp_world=0, .tp_expert_base=0,
        };
        rust_star_moe_sum_args routed_sum_args = {
            .width=n_embd, .tokens=rows,
            .src_token_stride=6ull*n_embd*sizeof(float),
            .dst_token_stride=n_embd*sizeof(float),
        };
#define RUST_STAR_L2_FFN_Q8_ARGS(in_width, out_width, weight_bytes) \
        (rust_star_q8_mm_args){ \
            .ne00=(in_width), .ne02=1, \
            .nb01=(uint64_t)((in_width)/32u)*34u, \
            .nb02=(weight_bytes), .nb03=(weight_bytes), .ne12=1, \
            .nb10=sizeof(float), .nb11=(uint64_t)(in_width)*sizeof(float), \
            .nb12=(uint64_t)(in_width)*rows*sizeof(float), \
            .nb13=(uint64_t)(in_width)*rows*sizeof(float), \
            .ne0=(out_width), .ne1=rows, .r2=1, .r3=1 \
        }
        rust_star_q8_mm_args shared_gate_args = RUST_STAR_L2_FFN_Q8_ARGS(
            n_embd, 2048, weights->ffn.shared_gate_bytes);
        rust_star_q8_mm_args shared_up_args = RUST_STAR_L2_FFN_Q8_ARGS(
            n_embd, 2048, weights->ffn.shared_up_bytes);
        rust_star_q8_mm_args shared_down_args = RUST_STAR_L2_FFN_Q8_ARGS(
            2048, n_embd, weights->ffn.shared_down_bytes);
#undef RUST_STAR_L2_FFN_Q8_ARGS
        rust_star_glu_args shared_swiglu_args = {
            .ne00=(int32_t)(rows*2048u),
            .nb01=(uint64_t)rows*2048u*sizeof(float),
            .ne10=(int32_t)(rows*2048u),
            .nb11=(uint64_t)rows*2048u*sizeof(float),
            .ne0=(int32_t)(rows*2048u),
            .nb1=(uint64_t)rows*2048u*sizeof(float),
            .i00=0, .i10=0, .alpha=1.0f, .limit=10.0f,
        };
        rust_star_hc_expand_args ffn_hc_post_args = attention_hc_args;
        ffn_hc_post_args.has_add = 1;
        rust_star_q8_mm_args layer3_q_a_args = {
            .ne00=n_embd, .ne02=1,
            .nb01=(n_embd/32u)*34u,
            .nb02=weights->layer3_ingress.q_a_bytes,
            .nb03=weights->layer3_ingress.q_a_bytes,
            .ne12=1, .nb10=sizeof(float),
            .nb11=n_embd*sizeof(float),
            .nb12=(uint64_t)rows*n_embd*sizeof(float),
            .nb13=(uint64_t)rows*n_embd*sizeof(float),
            .ne0=1024, .ne1=rows, .r2=1, .r3=1,
        };

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail_with_message(error, error_bytes,
            @"failed to create prefill layer-2 attention command");

        [encoder setComputePipelineState:context.q8PrefillPipeline];
        [encoder setBytes:&q_b_args length:sizeof(q_b_args) atIndex:0];
        [encoder setBuffer:model_buffers[0] offset:inner[0] atIndex:1];
        [encoder setBuffer:context.prefillLayer2FullQNorm offset:0 atIndex:2];
        [encoder setBuffer:q_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:6144u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows/32u,q_dim/64u,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.headNormRopePipeline];
        [encoder setBytes:&q_rope_args length:sizeof(q_rope_args) atIndex:0];
        [encoder setBuffer:q_buffer offset:0 atIndex:1];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(n_head,rows,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        uint32_t raw_elements = raw_rows*head_dim;
        [encoder setComputePipelineState:context.cpyF32F16Pipeline];
        [encoder setBytes:&raw_elements length:sizeof(raw_elements) atIndex:0];
        [encoder setBuffer:context.prefillLayer2FullKv offset:0 atIndex:1];
        [encoder setBuffer:staged_kv_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((raw_elements+1023u)/1024u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        uint32_t compressed_elements = compressed_rows*head_dim;
        [encoder setBytes:&compressed_elements length:sizeof(compressed_elements) atIndex:0];
        [encoder setBuffer:context.prefillLayer2AttnCompressed offset:0 atIndex:1];
        [encoder setBuffer:staged_kv_buffer
                     offset:(NSUInteger)raw_rows*head_f16_bytes atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((compressed_elements+1023u)/1024u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [encoder setComputePipelineState:context.flashBlkPipeline];
        [encoder setBytes:&block_args length:sizeof(block_args) atIndex:0];
        [encoder setBuffer:mask_buffer offset:0 atIndex:1];
        [encoder setBuffer:block_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(40,256,1)
             threadsPerThreadgroup:MTLSizeMake(32,1,1)];

        [encoder setComputePipelineState:context.flashNonvecPipeline];
        [encoder setBytes:&attention_args length:sizeof(attention_args) atIndex:0];
        [encoder setBuffer:q_buffer offset:0 atIndex:1];
        [encoder setBuffer:staged_kv_buffer offset:0 atIndex:2];
        [encoder setBuffer:staged_kv_buffer offset:0 atIndex:3];
        [encoder setBuffer:mask_buffer offset:0 atIndex:4];
        [encoder setBuffer:model_buffers[1] offset:inner[1] atIndex:5];
        [encoder setBuffer:pad_buffer offset:0 atIndex:6];
        [encoder setBuffer:block_buffer offset:0 atIndex:7];
        [encoder setBuffer:heads_buffer offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:28672u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(256,n_head,1)
             threadsPerThreadgroup:MTLSizeMake(32,8,1)];

        [encoder setComputePipelineState:context.ropeTailPipeline];
        [encoder setBytes:&inverse_args length:sizeof(inverse_args) atIndex:0];
        [encoder setBuffer:heads_buffer offset:0 atIndex:1];
        [encoder setBuffer:position_buffer offset:0 atIndex:2];
        [encoder setBuffer:heads_buffer offset:0 atIndex:3];
        [encoder setBuffer:heads_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(n_head,rows,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [encoder setComputePipelineState:context.attentionOutputBatchMapPipeline];
        [encoder setBytes:&map_args length:sizeof(map_args) atIndex:0];
        [encoder setBuffer:group_ids_buffer offset:0 atIndex:1];
        [encoder setBuffer:group_map_buffer offset:0 atIndex:2];
        [encoder setBuffer:group_map_buffer offset:tpe_bytes atIndex:3];
        [encoder setBuffer:group_map_buffer offset:work_offset atIndex:4];
        [encoder setThreadgroupMemoryLength:128u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(8,1,1)];

        [encoder setComputePipelineState:context.attentionOutputBatchLowPipeline];
        [encoder setBytes:&low_args length:sizeof(low_args) atIndex:0];
        [encoder setBuffer:model_buffers[2] offset:inner[2] atIndex:1];
        [encoder setBuffer:heads_buffer offset:0 atIndex:2];
        [encoder setBuffer:group_map_buffer offset:0 atIndex:3];
        [encoder setBuffer:group_map_buffer offset:tpe_bytes atIndex:4];
        [encoder setBuffer:attention_low_buffer offset:0 atIndex:5];
        [encoder setBuffer:group_map_buffer offset:work_offset atIndex:6];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(work_cap,output_rank/64u,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.q8PrefillPipeline];
        [encoder setBytes:&output_args length:sizeof(output_args) atIndex:0];
        [encoder setBuffer:model_buffers[3] offset:inner[3] atIndex:1];
        [encoder setBuffer:attention_low_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:6144u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows/32u,n_embd/64u,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.hcExpand4Pipeline];
        [encoder setBytes:&attention_hc_args length:sizeof(attention_hc_args) atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.prefillLayer2InputHc offset:0 atIndex:2];
        [encoder setBuffer:context.prefillLayer2AttnSplit
                     offset:4u*sizeof(float) atIndex:3];
        [encoder setBuffer:context.prefillLayer2AttnSplit
                     offset:8u*sizeof(float) atIndex:4];
        [encoder setBuffer:output_buffer offset:0 atIndex:5];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((rows*n_embd+255u)/256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [encoder setComputePipelineState:context.rmsNormF32Pipeline];
        [encoder setBytes:&ffn_hc_norm_args length:sizeof(ffn_hc_norm_args) atIndex:0];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:1];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:3];
        [encoder setBuffer:ffn_flat_hc_buffer offset:0 atIndex:4];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16PrefillPipeline];
        [encoder setBytes:&ffn_hc_mm_args length:sizeof(ffn_hc_mm_args) atIndex:0];
        [encoder setBuffer:model_buffers[4] offset:inner[4] atIndex:1];
        [encoder setBuffer:ffn_flat_hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:ffn_mix_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows/32u,1,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.hcIngressPipeline];
        [encoder setBytes:&ffn_hc_ingress_args length:sizeof(ffn_hc_ingress_args) atIndex:0];
        [encoder setBuffer:ffn_mix_buffer offset:0 atIndex:1];
        [encoder setBuffer:model_buffers[5] offset:inner[5] atIndex:2];
        [encoder setBuffer:model_buffers[6] offset:inner[6] atIndex:3];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:4];
        [encoder setBuffer:ffn_split_buffer offset:0 atIndex:5];
        [encoder setBuffer:ffn_cur_buffer offset:0 atIndex:6];
        [encoder setBuffer:model_buffers[7] offset:inner[7] atIndex:7];
        [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:(n_embd+4u+32u)*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16PrefillPipeline];
        [encoder setBytes:&router_args length:sizeof(router_args) atIndex:0];
        [encoder setBuffer:model_buffers[8] offset:inner[8] atIndex:1];
        [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:2];
        [encoder setBuffer:router_logits_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows/32u,4,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.routerSoftplusBatchPipeline];
        [encoder setBuffer:router_logits_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_probs_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(rows*256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:context.routerSqrtBatchPipeline];
        [encoder setBuffer:router_probs_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_probs_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(rows*256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:context.routerHashRowsBatchPipeline];
        [encoder setBuffer:model_buffers[9] offset:inner[9] atIndex:0];
        [encoder setBuffer:context.prefillLayer2Tokens offset:0 atIndex:1];
        [encoder setBuffer:router_selected_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:context.routerGatherWeightsBatchPipeline];
        [encoder setBuffer:router_probs_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_selected_buffer offset:0 atIndex:1];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(6,rows,1)
             threadsPerThreadgroup:MTLSizeMake(6,1,1)];

        [encoder setComputePipelineState:context.compressorSumRowsPipeline];
        [encoder setBytes:&router_sum_args length:sizeof(router_sum_args) atIndex:0];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:1];
        [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:2];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(6,1,1)];
        [encoder setComputePipelineState:context.routerClampSumsBatchPipeline];
        [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:context.routerDivideBatchPipeline];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_weight_sums_buffer offset:0 atIndex:1];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(6,rows,1)
             threadsPerThreadgroup:MTLSizeMake(6,1,1)];
        [encoder setComputePipelineState:context.routerScaleBatchPipeline];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:0];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(rows*6u,1,1)
             threadsPerThreadgroup:MTLSizeMake(192,1,1)];

        [encoder setComputePipelineState:context.routedBatchMapPipeline];
        [encoder setBytes:&routed_map_args length:sizeof(routed_map_args) atIndex:0];
        [encoder setBuffer:router_selected_buffer offset:0 atIndex:1];
        [encoder setBuffer:routed_map_buffer offset:0 atIndex:2];
        [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:3];
        [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:4];
        [encoder setThreadgroupMemoryLength:256u*6u*sizeof(uint16_t) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [encoder setComputePipelineState:context.routedBatchPairSwigluPipeline];
        [encoder setBytes:&routed_gate_args length:sizeof(routed_gate_args) atIndex:0];
        [encoder setBytes:&routed_activation_args length:sizeof(routed_activation_args) atIndex:1];
        [encoder setBuffer:model_buffers[10] offset:inner[10] atIndex:2];
        [encoder setBuffer:model_buffers[11] offset:inner[11] atIndex:3];
        [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:4];
        [encoder setBuffer:routed_map_buffer offset:0 atIndex:5];
        [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:6];
        [encoder setBuffer:routed_mid_buffer offset:0 atIndex:7];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:8];
        [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:9];
        [encoder setThreadgroupMemoryLength:16384u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(routed_map_work_cap,32,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.routedBatchDownPipeline];
        [encoder setBytes:&routed_down_args length:sizeof(routed_down_args) atIndex:0];
        [encoder setBuffer:model_buffers[12] offset:inner[12] atIndex:1];
        [encoder setBuffer:routed_mid_buffer offset:0 atIndex:2];
        [encoder setBuffer:routed_map_buffer offset:0 atIndex:3];
        [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:4];
        [encoder setBuffer:routed_experts_buffer offset:0 atIndex:5];
        [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:6];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(routed_map_work_cap,64,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [encoder setComputePipelineState:context.routedBatchSumPipeline];
        [encoder setBytes:&routed_sum_args length:sizeof(routed_sum_args) atIndex:0];
        [encoder setBuffer:routed_experts_buffer offset:0 atIndex:1];
        [encoder setBuffer:routed_out_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

#define RUST_STAR_ENCODE_L2_SHARED(args, index, input, output, width) do { \
        [encoder setComputePipelineState:context.q8PrefillPipeline]; \
        [encoder setBytes:&(args) length:sizeof(args) atIndex:0]; \
        [encoder setBuffer:model_buffers[(index)] offset:inner[(index)] atIndex:1]; \
        [encoder setBuffer:(input) offset:0 atIndex:2]; \
        [encoder setBuffer:(output) offset:0 atIndex:3]; \
        [encoder setThreadgroupMemoryLength:6144u atIndex:0]; \
        [encoder dispatchThreadgroups:MTLSizeMake(rows/32u,(width)/64u,1) \
             threadsPerThreadgroup:MTLSizeMake(128,1,1)]; \
    } while (0)
        RUST_STAR_ENCODE_L2_SHARED(shared_gate_args, 13, ffn_norm_buffer,
            shared_gate_buffer, 2048);
        RUST_STAR_ENCODE_L2_SHARED(shared_up_args, 14, ffn_norm_buffer,
            shared_up_buffer, 2048);
        [encoder setComputePipelineState:context.sharedSwigluBatchPipeline];
        [encoder setBytes:&shared_swiglu_args length:sizeof(shared_swiglu_args) atIndex:0];
        [encoder setBuffer:shared_gate_buffer offset:0 atIndex:1];
        [encoder setBuffer:shared_up_buffer offset:0 atIndex:2];
        [encoder setBuffer:shared_mid_buffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((rows*2048u+255u)/256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        RUST_STAR_ENCODE_L2_SHARED(shared_down_args, 15, shared_mid_buffer,
            shared_out_buffer, n_embd);
#undef RUST_STAR_ENCODE_L2_SHARED

        [encoder setComputePipelineState:context.hcExpand4Pipeline];
        [encoder setBytes:&ffn_hc_post_args length:sizeof(ffn_hc_post_args) atIndex:0];
        [encoder setBuffer:routed_out_buffer offset:0 atIndex:1];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:ffn_split_buffer offset:4u*sizeof(float) atIndex:3];
        [encoder setBuffer:ffn_split_buffer offset:8u*sizeof(float) atIndex:4];
        [encoder setBuffer:shared_out_buffer offset:0 atIndex:5];
        [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((rows*n_embd+255u)/256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [encoder setComputePipelineState:context.rmsNormF32Pipeline];
        [encoder setBytes:&ffn_hc_norm_args length:sizeof(ffn_hc_norm_args) atIndex:0];
        [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:1];
        [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:3];
        [encoder setBuffer:layer3_flat_hc_buffer offset:0 atIndex:4];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16PrefillPipeline];
        [encoder setBytes:&ffn_hc_mm_args length:sizeof(ffn_hc_mm_args) atIndex:0];
        [encoder setBuffer:model_buffers[16] offset:inner[16] atIndex:1];
        [encoder setBuffer:layer3_flat_hc_buffer offset:0 atIndex:2];
        [encoder setBuffer:layer3_mix_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows/32u,1,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.hcIngressPipeline];
        [encoder setBytes:&ffn_hc_ingress_args length:sizeof(ffn_hc_ingress_args) atIndex:0];
        [encoder setBuffer:layer3_mix_buffer offset:0 atIndex:1];
        [encoder setBuffer:model_buffers[17] offset:inner[17] atIndex:2];
        [encoder setBuffer:model_buffers[18] offset:inner[18] atIndex:3];
        [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:4];
        [encoder setBuffer:layer3_split_buffer offset:0 atIndex:5];
        [encoder setBuffer:layer3_cur_buffer offset:0 atIndex:6];
        [encoder setBuffer:model_buffers[19] offset:inner[19] atIndex:7];
        [encoder setBuffer:layer3_norm_buffer offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:(n_embd+4u+32u)*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.q8PrefillPipeline];
        [encoder setBytes:&layer3_q_a_args length:sizeof(layer3_q_a_args) atIndex:0];
        [encoder setBuffer:model_buffers[20] offset:inner[20] atIndex:1];
        [encoder setBuffer:layer3_norm_buffer offset:0 atIndex:2];
        [encoder setBuffer:layer3_q_lora_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:6144u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows/32u,1024u/64u,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [encoder endEncoding];

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        memcpy(attention_output, output_buffer.contents, output_bytes);
        memcpy(after_attention_hc, after_attention_hc_buffer.contents, hc_bytes);
        memcpy(after_ffn_hc, after_ffn_hc_buffer.contents, hc_bytes);
        memcpy(layer3_hc_attn_pre, layer3_cur_buffer.contents, output_bytes);
        memcpy(layer3_attn_norm, layer3_norm_buffer.contents, output_bytes);
        memcpy(layer3_q_lora, layer3_q_lora_buffer.contents,
               (NSUInteger)rows*1024u*sizeof(float));
        const NSUInteger final_tile_offset = (NSUInteger)(rows-32u)*n_embd*sizeof(float);
        memcpy(ffn_cur_final_tile,
               (const uint8_t *)ffn_cur_buffer.contents+final_tile_offset,
               32u*n_embd*sizeof(float));
        memcpy(ffn_norm_final_tile,
               (const uint8_t *)ffn_norm_buffer.contents+final_tile_offset,
               32u*n_embd*sizeof(float));
        const NSUInteger final_selected_offset = (NSUInteger)(rows-32u)*6u*sizeof(int32_t);
        memcpy(router_selected_final_tile,
               (const uint8_t *)router_selected_buffer.contents+final_selected_offset,
               32u*6u*sizeof(int32_t));
        memcpy(router_weights_final_tile,
               (const uint8_t *)router_weights_buffer.contents+final_selected_offset,
               32u*6u*sizeof(float));
        memcpy(routed_out_final_tile,
               (const uint8_t *)routed_out_buffer.contents+final_tile_offset,
               32u*n_embd*sizeof(float));
        memcpy(shared_out_final_tile,
               (const uint8_t *)shared_out_buffer.contents+final_tile_offset,
               32u*n_embd*sizeof(float));

        uint32_t pointer_matches = 0;
        for (uint32_t index = 0; index < 21u; index++) {
            pointer_matches += matches[index] ? 1u : 0u;
        }
        result->rows = rows;
        result->raw_kv_rows = raw_rows;
        result->compressed_kv_rows = compressed_rows;
        result->dispatches = 36u;
        result->wrapped_model_ranges = 21u;
        result->pointer_matches = pointer_matches;
        result->wall_ms = wall_end-wall_start;
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
    size_t error_bytes,
    const rust_star_metal_layer0_extension *layer0)
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
    const BOOL full_layer = layer0 != NULL;
    rust_star_metal_layer0_extension standalone_layer = {0};
    if (!layer0) layer0 = &standalone_layer;
    const uint32_t position = full_layer ? layer0->position : 1u;
    const uint32_t initial_state_mode = full_layer
        ? layer0->initial_state_mode : RUST_STAR_INITIAL_STATE_CAPTURED;
    const BOOL cold_initial_state = initial_state_mode == RUST_STAR_INITIAL_STATE_COLD;
    const uint32_t context_capacity = full_layer ? layer0->context_capacity : 3u;
    const uint32_t cache_capacity_rows = full_layer
        ? (context_capacity < 128u ? context_capacity : 128u) : 3u;
    const BOOL compressed_layer = full_layer && layer0->layer_index >= 2u;
    const uint32_t compressor_ratio = (layer0->layer_index % 2u) == 0u ? 4u : 128u;
    const uint32_t compressed_cache_capacity_rows = compressed_layer
        ? context_capacity / compressor_ratio + 2u : 2u;
    /* Synchronized multi-layer controls intentionally reuse unscoped
     * attention scratch. Reserve the ratio-4 maximum for every full layer so
     * its shape does not change when execution crosses into layer 2. */
    const uint32_t attention_capacity_rows = full_layer
        ? cache_capacity_rows + context_capacity / 4u + 2u : 3u;
    const uint32_t visible_cache_rows = position + 1u < cache_capacity_rows
        ? position + 1u : cache_capacity_rows;
    const uint32_t raw_cache_start = (position + 1u - visible_cache_rows) %
        cache_capacity_rows;
    const uint32_t compressor_width = compressor_ratio == 4u ? 1024u : 512u;
    const BOOL indexer_layer = compressed_layer && compressor_ratio == 4u;
    const BOOL compressor_emit = compressed_layer &&
        ((position + 1u) % compressor_ratio) == 0u;
    const uint32_t compressed_cache_rows = compressed_layer
        ? (position + 1u) / compressor_ratio : 0u;
    const uint32_t attention_cache_rows = visible_cache_rows + compressed_cache_rows;
    const BOOL continuing_layer = full_layer && layer0->reuse_previous_hc != 0;
    const uint32_t command_mode = full_layer ? layer0->command_mode : RUST_STAR_COMMAND_SYNCHRONIZED;
    const BOOL chained_submission = command_mode == RUST_STAR_COMMAND_CHAINED_ENQUEUE ||
        command_mode == RUST_STAR_COMMAND_CHAINED_FINAL;
    const BOOL chained_collect = command_mode == RUST_STAR_COMMAND_CHAINED_COLLECT;
    const BOOL chained_timing = command_mode == RUST_STAR_COMMAND_CHAINED_TIMING;
    const BOOL chained_replay = chained_collect || chained_timing;
    const BOOL layer_scoped_buffers = chained_submission || chained_replay;
    const uint32_t warmup_iterations = full_layer ? layer0->warmup_iterations : 0;
    const uint32_t measured_iterations = full_layer ? layer0->measured_iterations : 1;
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
    if (full_layer && (!attention_output ||
        !layer0->ffn_mixes || !layer0->ffn_split || !layer0->ffn_norm ||
        !layer0->router_logits || !layer0->router_probs || !layer0->selected ||
        !layer0->router_weights || !layer0->routed_mid || !layer0->routed_out ||
        !layer0->shared_out || !layer0->after_ffn_hc)) {
        return fail_with_message(error, error_bytes,
            @"full layer outputs require the complete attention path and output set");
    }
    if (full_layer && (layer0->layer_index > 42 ||
        (layer0->layer_index == 0 && continuing_layer) ||
        (layer0->layer_index > 0 && !continuing_layer))) {
        return fail_with_message(error, error_bytes,
            @"full layer execution must run layer 0 before continuing into later layers");
    }
    if (full_layer && (context_capacity == 0u || position >= context_capacity ||
        (!cold_initial_state && position < 1u))) {
        return fail_with_message(error, error_bytes,
            @"the retained decoder frontier received an invalid position/state mode");
    }
    if (full_layer && initial_state_mode > RUST_STAR_INITIAL_STATE_COLD) {
        return fail_with_message(error, error_bytes,
            @"the retained decoder frontier received an invalid initial-state mode");
    }
    if (full_layer && (visible_cache_rows > cache_capacity_rows ||
        compressed_cache_rows > compressed_cache_capacity_rows ||
        attention_cache_rows > attention_capacity_rows)) {
        return fail_with_message(error, error_bytes,
            @"the retained decoder frontier exceeds its cache capacities");
    }
    if (indexer_layer && compressed_cache_rows > 512u) {
        return fail_with_message(error, error_bytes,
            @"ratio-4 attention beyond 512 compressed rows requires sparse indexer selection");
    }
    if (full_layer && position > 1u && command_mode == RUST_STAR_COMMAND_SYNCHRONIZED) {
        return fail_with_message(error, error_bytes,
            @"position-advancing execution requires the layer-scoped chained scheduler");
    }
    if (command_mode > RUST_STAR_COMMAND_CHAINED_TIMING || (!full_layer && command_mode != 0)) {
        return fail_with_message(error, error_bytes, @"invalid full-layer command mode");
    }
    if (full_layer && command_mode == RUST_STAR_COMMAND_SYNCHRONIZED &&
        layer0->chain_final_layer != 0) {
        return fail_with_message(error, error_bytes,
            @"synchronized layer execution must not declare a command-chain tail");
    }
    if ((chained_submission || chained_replay) &&
        (layer0->chain_final_layer < 2 || layer0->chain_final_layer > 42 ||
         layer0->layer_index > layer0->chain_final_layer)) {
        return fail_with_message(error, error_bytes,
            @"chained layer execution has an invalid command-chain tail");
    }
    if (chained_submission && (warmup_iterations != 0 || measured_iterations != 1)) {
        return fail_with_message(error, error_bytes,
            @"chained layer submission requires exactly one measured iteration");
    }
    if (continuing_layer && (warmup_iterations != 0 || measured_iterations != 1)) {
        return fail_with_message(error, error_bytes,
            @"continued layer execution currently requires exactly one measured iteration");
    }
    if (full_layer && (measured_iterations == 0 || warmup_iterations > 1000 ||
        measured_iterations > 1000 || warmup_iterations > 1000-measured_iterations)) {
        return fail_with_message(error, error_bytes,
            @"full layer iteration counts must be positive and total at most 1000");
    }
    if (full_layer && measured_iterations > 1 &&
        (!layer0->wall_ms_samples || !layer0->gpu_ms_samples ||
         !layer0->repeat_bitwise_matches)) {
        return fail_with_message(error, error_bytes,
            @"repeated full layer execution requires timing and repeat-validation outputs");
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
    if (full_layer &&
        (layer0->hc_ffn_fn_bytes != 16384ull*24ull*sizeof(uint16_t) ||
         layer0->hc_ffn_scale_bytes != 3u*sizeof(float) ||
         layer0->hc_ffn_base_bytes != 24u*sizeof(float) ||
         layer0->ffn_norm_bytes != 4096u*sizeof(float) ||
         layer0->router_gate_bytes != 4096ull*256ull*sizeof(uint16_t) ||
         ((layer0->layer_index < 3 &&
           layer0->router_aux_bytes != 6ull*129280ull*sizeof(int32_t)) ||
          (layer0->layer_index >= 3 &&
           layer0->router_aux_bytes != 256u*sizeof(float))) ||
         layer0->routed_gate_bytes != 256ull*2048ull*1056ull ||
         layer0->routed_up_bytes != 256ull*2048ull*1056ull ||
         layer0->routed_down_bytes != 256ull*4096ull*672ull ||
         layer0->shared_gate_bytes != 2048ull*4352ull ||
         layer0->shared_up_bytes != 2048ull*4352ull ||
         layer0->shared_down_bytes != 4096ull*2176ull)) {
        return fail_with_message(error, error_bytes, @"full layer tensor shapes are invalid");
    }
    if (compressed_layer &&
        ((!cold_initial_state && !layer0->compressor_prime_attn_norm) ||
         !layer0->compressed_kv_row ||
         layer0->attn_compressor_ape_bytes != (uint64_t)compressor_width*compressor_ratio*sizeof(uint16_t) ||
         layer0->attn_compressor_kv_bytes != (uint64_t)4096u*compressor_width*sizeof(uint16_t) ||
         layer0->attn_compressor_gate_bytes != (uint64_t)4096u*compressor_width*sizeof(uint16_t) ||
         layer0->attn_compressor_norm_bytes != 512u*sizeof(float))) {
        return fail_with_message(error, error_bytes, @"attention compressor tensor shapes are invalid");
    }
    if (!compressed_layer &&
        (layer0->attn_compressor_ape_bytes || layer0->attn_compressor_kv_bytes ||
         layer0->attn_compressor_gate_bytes || layer0->attn_compressor_norm_bytes)) {
        return fail_with_message(error, error_bytes, @"raw-attention layer unexpectedly supplied compressor tensors");
    }
    if (indexer_layer &&
        (!layer0->compressed_indexer_row ||
         layer0->indexer_compressor_ape_bytes != 256u*4u*sizeof(uint16_t) ||
         layer0->indexer_compressor_kv_bytes != 4096ull*256ull*sizeof(uint16_t) ||
         layer0->indexer_compressor_gate_bytes != 4096ull*256ull*sizeof(uint16_t) ||
         layer0->indexer_compressor_norm_bytes != 128u*sizeof(float))) {
        return fail_with_message(error, error_bytes, @"indexer compressor tensor shapes are invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_get_rows_f16_pipeline(context, error, error_bytes) ||
            !ensure_attention_ingress_pipelines(context, error, error_bytes) ||
            !ensure_q8_projection_pipeline(context, error, error_bytes) ||
            (attention_output && !ensure_attention_output_pipelines(context, error, error_bytes)) ||
            (full_layer && !ensure_moe_output_pipelines(context, error, error_bytes))) return 0;
        memset(result, 0, sizeof(*result));

        NSNumber *chained_key = @(layer0->layer_index);
        if (chained_submission && layer0->layer_index == 0) {
            [context.chainedCommands removeAllObjects];
            [context.chainedWallStarts removeAllObjects];
            context.chainedReady = NO;
            context.chainedWallEnd = 0.0;
            context.chainedFinalLayer = layer0->chain_final_layer;
        } else if (chained_submission && !context.chainedCommands[@(layer0->layer_index-1)]) {
            return fail_with_message(error, error_bytes,
                @"chained layer submission is missing its preceding command buffer");
        }
        if ((chained_submission || chained_replay) &&
            context.chainedFinalLayer != layer0->chain_final_layer) {
            return fail_with_message(error, error_bytes,
                @"chained layer execution changed its declared command-chain tail");
        }
        if (command_mode == RUST_STAR_COMMAND_CHAINED_FINAL &&
            layer0->layer_index != layer0->chain_final_layer) {
            return fail_with_message(error, error_bytes,
                @"only the declared tail layer may finalize the command chain");
        }
        if (command_mode == RUST_STAR_COMMAND_CHAINED_ENQUEUE &&
            layer0->layer_index == layer0->chain_final_layer) {
            return fail_with_message(error, error_bytes,
                @"the declared tail layer must finalize the command chain");
        }
        if (chained_replay && (!context.chainedReady || !context.chainedCommands[chained_key])) {
            return fail_with_message(error, error_bytes,
                @"chained layer collection requires a completed command chain");
        }
        if (chained_timing && layer0->layer_index != 0) {
            return fail_with_message(error, error_bytes,
                @"chained timing must be collected from layer 0");
        }

        NSUInteger embedding_inner = 0, hc_fn_inner = 0, scale_inner = 0;
        NSUInteger base_inner = 0, norm_inner = 0, q_inner = 0;
        NSUInteger q_norm_inner = 0, kv_inner = 0, kv_norm_inner = 0, q_b_inner = 0;
        NSUInteger sinks_inner = 0;
        NSUInteger output_a_inner = 0, output_b_inner = 0;
        BOOL matches[33] = {NO};
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
        NSUInteger layer_inner[12] = {0};
        id<MTLBuffer> ffn_hc_fn = nil, ffn_hc_scale = nil, ffn_hc_base = nil;
        id<MTLBuffer> ffn_norm_weight = nil, router_gate = nil, router_aux = nil;
        id<MTLBuffer> routed_gate_weight = nil, routed_up_weight = nil, routed_down_weight = nil;
        id<MTLBuffer> shared_gate_weight = nil, shared_up_weight = nil, shared_down_weight = nil;
        NSUInteger compressor_inner[4] = {0};
        NSUInteger indexer_inner[4] = {0};
        id<MTLBuffer> compressor_ape = nil, compressor_kv_weight = nil;
        id<MTLBuffer> compressor_gate_weight = nil, compressor_norm_weight = nil;
        id<MTLBuffer> indexer_ape = nil, indexer_kv_weight = nil;
        id<MTLBuffer> indexer_gate_weight = nil, indexer_norm_weight = nil;
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
        if (full_layer) {
            const uint64_t layer_offsets[12] = {
                layer0->hc_ffn_fn_offset, layer0->hc_ffn_scale_offset,
                layer0->hc_ffn_base_offset, layer0->ffn_norm_offset,
                layer0->router_gate_offset, layer0->router_aux_offset,
                layer0->routed_gate_offset, layer0->routed_up_offset,
                layer0->routed_down_offset, layer0->shared_gate_offset,
                layer0->shared_up_offset, layer0->shared_down_offset,
            };
            const uint64_t layer_bytes[12] = {
                layer0->hc_ffn_fn_bytes, layer0->hc_ffn_scale_bytes,
                layer0->hc_ffn_base_bytes, layer0->ffn_norm_bytes,
                layer0->router_gate_bytes, layer0->router_aux_bytes,
                layer0->routed_gate_bytes, layer0->routed_up_bytes,
                layer0->routed_down_bytes, layer0->shared_gate_bytes,
                layer0->shared_up_bytes, layer0->shared_down_bytes,
            };
            id<MTLBuffer> __strong *layer_buffers[12] = {
                &ffn_hc_fn, &ffn_hc_scale, &ffn_hc_base, &ffn_norm_weight,
                &router_gate, &router_aux, &routed_gate_weight, &routed_up_weight,
                &routed_down_weight, &shared_gate_weight, &shared_up_weight,
                &shared_down_weight,
            };
            for (uint32_t index = 0; index < 12; index++) {
                *layer_buffers[index] = wrap_model_range(
                    context, model_mapping, model_bytes,
                    layer_offsets[index], layer_bytes[index], &layer_inner[index],
                    &matches[13u + index], error, error_bytes);
                if (!*layer_buffers[index]) return 0;
            }
        }
        if (compressed_layer) {
            const uint64_t offsets[4] = {
                layer0->attn_compressor_ape_offset,
                layer0->attn_compressor_kv_offset,
                layer0->attn_compressor_gate_offset,
                layer0->attn_compressor_norm_offset,
            };
            const uint64_t bytes[4] = {
                layer0->attn_compressor_ape_bytes,
                layer0->attn_compressor_kv_bytes,
                layer0->attn_compressor_gate_bytes,
                layer0->attn_compressor_norm_bytes,
            };
            id<MTLBuffer> __strong *buffers[4] = {
                &compressor_ape, &compressor_kv_weight,
                &compressor_gate_weight, &compressor_norm_weight,
            };
            for (uint32_t index = 0; index < 4; index++) {
                *buffers[index] = wrap_model_range(context, model_mapping, model_bytes,
                    offsets[index], bytes[index], &compressor_inner[index],
                    &matches[25u+index], error, error_bytes);
                if (!*buffers[index]) return 0;
            }
        }
        if (indexer_layer) {
            const uint64_t offsets[4] = {
                layer0->indexer_compressor_ape_offset,
                layer0->indexer_compressor_kv_offset,
                layer0->indexer_compressor_gate_offset,
                layer0->indexer_compressor_norm_offset,
            };
            const uint64_t bytes[4] = {
                layer0->indexer_compressor_ape_bytes,
                layer0->indexer_compressor_kv_bytes,
                layer0->indexer_compressor_gate_bytes,
                layer0->indexer_compressor_norm_bytes,
            };
            id<MTLBuffer> __strong *buffers[4] = {
                &indexer_ape, &indexer_kv_weight,
                &indexer_gate_weight, &indexer_norm_weight,
            };
            for (uint32_t index = 0; index < 4; index++) {
                *buffers[index] = wrap_model_range(context, model_mapping, model_bytes,
                    offsets[index], bytes[index], &indexer_inner[index],
                    &matches[29u+index], error, error_bytes);
                if (!*buffers[index]) return 0;
            }
        }

        NSString *prior_hc_key = layer_scoped_buffers && layer0->layer_index > 0
            ? layer_buffer_key(@"layer_hc_state", YES, layer0->layer_index-1)
            : @"layer_hc_state";
        id<MTLBuffer> prior_hc = context.activationBufferCache[prior_hc_key];
        if (continuing_layer && !prior_hc) {
            return fail_with_message(error, error_bytes, @"continued layer execution has no retained HC state");
        }
        id<MTLBuffer> embedding = persistent_buffer(context, layer_buffer_key(@"embedding", layer_scoped_buffers, layer0->layer_index), n_embd*sizeof(float), error, error_bytes);
        id<MTLBuffer> cur_hc = persistent_buffer(context, layer_buffer_key(@"embedding_hc", layer_scoped_buffers, layer0->layer_index), hc_dim*sizeof(float), error, error_bytes);
        id<MTLBuffer> layer_input_hc = continuing_layer ? prior_hc : cur_hc;
        id<MTLBuffer> flat_hc = persistent_buffer(context, layer_buffer_key(@"attention_flat_hc", layer_scoped_buffers, layer0->layer_index), hc_dim*sizeof(float), error, error_bytes);
        id<MTLBuffer> mix_buffer = persistent_buffer(context, layer_buffer_key(@"attention_mix", layer_scoped_buffers, layer0->layer_index), mix_hc*sizeof(float), error, error_bytes);
        id<MTLBuffer> split_buffer = persistent_buffer(context, layer_buffer_key(@"attention_split", layer_scoped_buffers, layer0->layer_index), mix_hc*sizeof(float), error, error_bytes);
        id<MTLBuffer> collapsed_buffer = persistent_buffer(context, layer_buffer_key(@"attention_collapsed", layer_scoped_buffers, layer0->layer_index), n_embd*sizeof(float), error, error_bytes);
        id<MTLBuffer> norm_buffer = persistent_buffer(context, layer_buffer_key(@"attention_norm", layer_scoped_buffers, layer0->layer_index), n_embd*sizeof(float), error, error_bytes);
        id<MTLBuffer> q_buffer = persistent_buffer(context, layer_buffer_key(@"q_lora", layer_scoped_buffers, layer0->layer_index), q_elements*sizeof(float), error, error_bytes);
        id<MTLBuffer> q_norm_buffer = extended ? persistent_buffer(context, layer_buffer_key(@"q_lora_norm", layer_scoped_buffers, layer0->layer_index), q_elements*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> kv_raw_buffer = extended ? persistent_buffer(context, layer_buffer_key(@"kv_raw", layer_scoped_buffers, layer0->layer_index), kv_elements*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> kv_norm_buffer = extended ? persistent_buffer(context, layer_buffer_key(@"kv_norm", layer_scoped_buffers, layer0->layer_index), kv_elements*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> q_raw_buffer = extended ? persistent_buffer(context, layer_buffer_key(@"q_raw", layer_scoped_buffers, layer0->layer_index), q_raw_elements*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> q_cur_buffer = rope_and_store ? persistent_buffer(context, layer_buffer_key(@"q_cur", layer_scoped_buffers, layer0->layer_index), q_raw_elements*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> kv_rope_buffer = rope_and_store ? persistent_buffer(context, layer_buffer_key(@"kv_rope", layer_scoped_buffers, layer0->layer_index), kv_elements*sizeof(float), error, error_bytes) : nil;
        NSString *cache_key = full_layer ?
            [NSString stringWithFormat:@"kv_cache_layer_%u", layer0->layer_index] :
            @"kv_cache_probe";
        id<MTLBuffer> cache_buffer = rope_and_store ? persistent_buffer(context, cache_key, cache_capacity_rows*kv_elements*sizeof(float), error, error_bytes) : nil;
        const NSUInteger staged_kv_bytes = attention_capacity_rows*kv_elements*sizeof(uint16_t);
        const NSUInteger mask_storage_bytes = attention_capacity_rows*sizeof(uint16_t);
        const NSUInteger mask_row_bytes = attention_cache_rows*sizeof(uint16_t);
        /* The vector FlashAttention pad holds two 32-row tiles, not one tile
         * per cache row. Keeping this scratch shape fixed is what allows the
         * persistent context capacities to grow without quadratic waste. */
        const NSUInteger flash_pad_bytes = 2u*32u*kv_elements*sizeof(uint16_t) + 32u*sizeof(uint16_t);
        const NSUInteger flash_tmp_bytes = 64u*kv_elements*32u*sizeof(float) + 64u*64u*sizeof(float);
        id<MTLBuffer> staged_kv_buffer = attention_read ? persistent_buffer(context, layer_buffer_key(@"staged_kv", layer_scoped_buffers, layer0->layer_index), staged_kv_bytes, error, error_bytes) : nil;
        id<MTLBuffer> mask_buffer = attention_read ? persistent_buffer(context, layer_buffer_key(@"attention_mask", layer_scoped_buffers, layer0->layer_index), mask_storage_bytes, error, error_bytes) : nil;
        id<MTLBuffer> flash_pad_buffer = attention_read ? persistent_buffer(context, layer_buffer_key(@"flash_pad", layer_scoped_buffers, layer0->layer_index), flash_pad_bytes, error, error_bytes) : nil;
        id<MTLBuffer> flash_tmp_buffer = attention_read ? persistent_buffer(context, layer_buffer_key(@"flash_tmp", layer_scoped_buffers, layer0->layer_index), flash_tmp_bytes, error, error_bytes) : nil;
        id<MTLBuffer> attention_raw_buffer = attention_read ? persistent_buffer(context, layer_buffer_key(@"attention_raw", layer_scoped_buffers, layer0->layer_index), q_raw_elements*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> attention_back_buffer = attention_read ? persistent_buffer(context, layer_buffer_key(@"attention_back", layer_scoped_buffers, layer0->layer_index), q_raw_elements*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> attention_low_buffer = attention_output ? persistent_buffer(context, layer_buffer_key(@"attention_low", layer_scoped_buffers, layer0->layer_index), 8192u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> attention_out_buffer = attention_output ? persistent_buffer(context, layer_buffer_key(@"attention_out", layer_scoped_buffers, layer0->layer_index), n_embd*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> after_attention_hc_buffer = attention_output ? persistent_buffer(context, layer_buffer_key(@"after_attention_hc", layer_scoped_buffers, layer0->layer_index), hc_dim*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> ffn_flat_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"ffn_flat_hc", layer_scoped_buffers, layer0->layer_index), hc_dim*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> ffn_mix_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"ffn_mix", layer_scoped_buffers, layer0->layer_index), mix_hc*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> ffn_split_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"ffn_split", layer_scoped_buffers, layer0->layer_index), mix_hc*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> ffn_cur_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"ffn_cur", layer_scoped_buffers, layer0->layer_index), n_embd*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> ffn_norm_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"ffn_norm", layer_scoped_buffers, layer0->layer_index), n_embd*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> router_logits_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"router_logits", layer_scoped_buffers, layer0->layer_index), 256u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> router_probs_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"router_probs", layer_scoped_buffers, layer0->layer_index), 256u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> selected_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"selected_experts", layer_scoped_buffers, layer0->layer_index), 6u*sizeof(int32_t), error, error_bytes) : nil;
        id<MTLBuffer> route_weights_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"router_weights", layer_scoped_buffers, layer0->layer_index), 6u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> routed_gate_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"routed_gate", layer_scoped_buffers, layer0->layer_index), 6u*2048u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> routed_up_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"routed_up", layer_scoped_buffers, layer0->layer_index), 6u*2048u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> routed_mid_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"routed_mid", layer_scoped_buffers, layer0->layer_index), 6u*2048u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> routed_out_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"routed_out", layer_scoped_buffers, layer0->layer_index), n_embd*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> shared_gate_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"shared_gate", layer_scoped_buffers, layer0->layer_index), 2048u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> shared_up_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"shared_up", layer_scoped_buffers, layer0->layer_index), 2048u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> shared_mid_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"shared_mid", layer_scoped_buffers, layer0->layer_index), 2048u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> shared_out_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"shared_out", layer_scoped_buffers, layer0->layer_index), n_embd*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> after_ffn_hc_buffer = full_layer ? persistent_buffer(context, layer_buffer_key(@"layer_hc_state", layer_scoped_buffers, layer0->layer_index), hc_dim*sizeof(float), error, error_bytes) : nil;
        const uint32_t compressor_state_rows = compressor_ratio == 4u ? 8u : compressor_ratio;
        const uint32_t compressor_pool_rows = compressor_state_rows;
        id<MTLBuffer> compressor_prime_buffer = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressor_prime", YES, layer0->layer_index), n_embd*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressor_kv_cur = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressor_kv_cur", YES, layer0->layer_index), compressor_width*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressor_score_cur = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressor_score_cur", YES, layer0->layer_index), compressor_width*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressor_state_kv = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressor_state_kv", YES, layer0->layer_index), compressor_state_rows*compressor_width*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressor_state_score = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressor_state_score", YES, layer0->layer_index), compressor_state_rows*compressor_width*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressed_kv_buffer = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressed_kv_work", YES, layer0->layer_index), 512u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressed_kv_cache = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressed_kv_cache", YES, layer0->layer_index), compressed_cache_capacity_rows*512u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressor_packed_kv = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressor_packed_kv", YES, layer0->layer_index), compressor_pool_rows*512u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressor_packed_score = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressor_packed_score", YES, layer0->layer_index), compressor_pool_rows*512u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressor_softmax = compressed_layer ? persistent_buffer(context, layer_buffer_key(@"compressor_softmax", YES, layer0->layer_index), compressor_pool_rows*512u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> indexer_kv_cur = indexer_layer ? persistent_buffer(context, layer_buffer_key(@"indexer_kv_cur", YES, layer0->layer_index), 256u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> indexer_score_cur = indexer_layer ? persistent_buffer(context, layer_buffer_key(@"indexer_score_cur", YES, layer0->layer_index), 256u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> indexer_state_kv = indexer_layer ? persistent_buffer(context, layer_buffer_key(@"indexer_state_kv", YES, layer0->layer_index), 8u*256u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> indexer_state_score = indexer_layer ? persistent_buffer(context, layer_buffer_key(@"indexer_state_score", YES, layer0->layer_index), 8u*256u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressed_indexer_buffer = indexer_layer ? persistent_buffer(context, layer_buffer_key(@"compressed_indexer_work", YES, layer0->layer_index), 128u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> compressed_indexer_cache = indexer_layer ? persistent_buffer(context, layer_buffer_key(@"compressed_indexer_cache", YES, layer0->layer_index), compressed_cache_capacity_rows*128u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> indexer_packed_kv = indexer_layer ? persistent_buffer(context, layer_buffer_key(@"indexer_packed_kv", YES, layer0->layer_index), 8u*128u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> indexer_packed_score = indexer_layer ? persistent_buffer(context, layer_buffer_key(@"indexer_packed_score", YES, layer0->layer_index), 8u*128u*sizeof(float), error, error_bytes) : nil;
        id<MTLBuffer> indexer_softmax = indexer_layer ? persistent_buffer(context, layer_buffer_key(@"indexer_softmax", YES, layer0->layer_index), 8u*128u*sizeof(float), error, error_bytes) : nil;
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
            if (error && error_bytes != 0u && error[0] != '\0') return 0;
            return fail_with_message(error, error_bytes, @"failed to allocate attention-read buffers");
        }
        if (attention_output && (!attention_low_buffer || !attention_out_buffer || !after_attention_hc_buffer)) {
            return fail_with_message(error, error_bytes, @"failed to allocate attention-output buffers");
        }
        if (full_layer && (!ffn_flat_buffer || !ffn_mix_buffer || !ffn_split_buffer ||
            !ffn_cur_buffer || !ffn_norm_buffer || !router_logits_buffer ||
            !router_probs_buffer || !selected_buffer || !route_weights_buffer ||
            !routed_gate_buffer || !routed_up_buffer || !routed_mid_buffer ||
            !routed_out_buffer || !shared_gate_buffer || !shared_up_buffer ||
            !shared_mid_buffer || !shared_out_buffer || !after_ffn_hc_buffer)) {
            return fail_with_message(error, error_bytes, @"failed to allocate full layer buffers");
        }
        if (compressed_layer && (!compressor_prime_buffer || !compressor_kv_cur ||
            !compressor_score_cur || !compressor_state_kv || !compressor_state_score ||
            !compressed_kv_buffer || !compressed_kv_cache || !compressor_packed_kv ||
            !compressor_packed_score || !compressor_softmax)) {
            return fail_with_message(error, error_bytes, @"failed to allocate attention compressor state");
        }
        if (indexer_layer && (!indexer_kv_cur || !indexer_score_cur ||
            !indexer_state_kv || !indexer_state_score || !compressed_indexer_buffer ||
            !compressed_indexer_cache ||
            !indexer_packed_kv || !indexer_packed_score || !indexer_softmax)) {
            return fail_with_message(error, error_bytes, @"failed to allocate indexer compressor state");
        }
        const BOOL initialize_state = !chained_replay &&
            ((cold_initial_state && position == 0u) ||
             (!cold_initial_state && position == 1u));
        if (rope_and_store && initialize_state) {
            float *cache = cache_buffer.contents;
            for (uint32_t index = 0; index < cache_capacity_rows*kv_elements; index++) cache[index] = -12345.5f;
            if (!cold_initial_state && attention_read) {
                memcpy(cache, cache_row0, kv_elements*sizeof(float));
            }
        }
        if (attention_read && !chained_replay) memset(mask_buffer.contents, 0, mask_storage_bytes);
        if (compressed_layer && initialize_state) {
            if (!cold_initial_state) {
                memcpy(compressor_prime_buffer.contents, layer0->compressor_prime_attn_norm,
                    n_embd*sizeof(float));
            }
            memset(compressor_state_kv.contents, 0,
                compressor_state_rows*compressor_width*sizeof(float));
            float *scores = compressor_state_score.contents;
            for (uint32_t index = 0; index < compressor_state_rows*compressor_width; index++) {
                scores[index] = -INFINITY;
            }
            memset(compressed_kv_cache.contents, 0,
                compressed_cache_capacity_rows*512u*sizeof(float));
            if (indexer_layer) {
                memset(indexer_state_kv.contents, 0, 8u*256u*sizeof(float));
                scores = indexer_state_score.contents;
                for (uint32_t index = 0; index < 8u*256u; index++) scores[index] = -INFINITY;
                memset(compressed_indexer_cache.contents, 0,
                    compressed_cache_capacity_rows*128u*sizeof(float));
            }
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
        const bool compressed_attention = full_layer && layer0->layer_index >= 2;
        const float rope_freq_base = compressed_attention ? 160000.0f : 10000.0f;
        const float rope_freq_scale = compressed_attention ? (1.0f/16.0f) : 1.0f;
        const float rope_ext_factor = compressed_attention ? 1.0f : 0.0f;
        const float rope_attn_factor = compressed_attention
            ? 1.0f/(1.0f + 0.1f*logf(1.0f/rope_freq_scale))
            : 1.0f;
        rust_star_head_norm_rope_args q_rope_args = {
            .n_head=64, .head_dim=512, .head_dim4=128, .n_dims=64,
            .n_ctx_orig=compressed_attention ? 65536 : 0, .pos0=(int32_t)position, .inverse=0,
            .eps=1.0e-6f, .freq_base=rope_freq_base, .freq_scale=rope_freq_scale,
            .ext_factor=rope_ext_factor, .attn_factor=rope_attn_factor,
            .beta_fast=32.0f, .beta_slow=1.0f,
        };
        const uint64_t kv_row_f32_bytes = (uint64_t)kv_elements*sizeof(float);
        rust_star_rope_tail_args kv_rope_args = {
            .ne00=512, .ne01=1, .ne02=1, .ne03=1,
            .nb00=sizeof(float), .nb01=kv_row_f32_bytes,
            .nb02=kv_row_f32_bytes, .nb03=kv_row_f32_bytes,
            .nb0=sizeof(float), .nb1=kv_row_f32_bytes,
            .nb2=kv_row_f32_bytes, .nb3=kv_row_f32_bytes,
            .n_dims=64, .mode=0,
            .n_ctx_orig=compressed_attention ? 65536 : 0, .inverse=0,
            .freq_base=rope_freq_base, .freq_scale=rope_freq_scale,
            .ext_factor=rope_ext_factor, .attn_factor=rope_attn_factor,
            .beta_fast=32.0f, .beta_slow=1.0f,
            .src2=false,
        };
        rust_star_kv_store_args kv_store_args = {
            .head_dim=512, .n_rot=64,
            .raw_row=(int32_t)(position % cache_capacity_rows),
        };
        const uint64_t attention_head_bytes = (uint64_t)kv_elements*sizeof(float);
        const uint64_t staged_row_bytes = (uint64_t)kv_elements*sizeof(uint16_t);
        rust_star_flash_pad_args flash_pad_args = {
            .ne11=(int32_t)attention_cache_rows, .ne_12_2=1, .ne_12_3=1,
            .nb11=staged_row_bytes, .nb12=(uint64_t)attention_cache_rows*staged_row_bytes,
            .nb13=(uint64_t)attention_cache_rows*staged_row_bytes,
            .nb21=staged_row_bytes, .nb22=(uint64_t)attention_cache_rows*staged_row_bytes,
            .nb23=(uint64_t)attention_cache_rows*staged_row_bytes,
            .ne31=1, .ne32=1, .ne33=1,
            .nb31=mask_row_bytes, .nb32=mask_row_bytes, .nb33=mask_row_bytes,
        };
        rust_star_flash_vec_args flash_vec_args = {
            .ne01=1, .ne02=64, .ne03=1,
            .nb01=64u*attention_head_bytes, .nb02=attention_head_bytes,
            .nb03=64u*attention_head_bytes,
            .ne11=(int32_t)attention_cache_rows, .ne_12_2=1, .ne_12_3=1, .ns10=512,
            .nb11=staged_row_bytes,
            .nb12=(uint64_t)attention_cache_rows*staged_row_bytes,
            .nb13=(uint64_t)attention_cache_rows*staged_row_bytes,
            .ns20=512,
            .nb21=staged_row_bytes,
            .nb22=(uint64_t)attention_cache_rows*staged_row_bytes,
            .nb23=(uint64_t)attention_cache_rows*staged_row_bytes,
            .ne31=1, .ne32=1, .ne33=1,
            .nb31=mask_row_bytes, .nb32=mask_row_bytes, .nb33=mask_row_bytes,
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
        rust_star_norm_args ffn_norm_args = norm;
        rust_star_q8_mv_args ffn_hc_mv = {
            .ne00=16384, .ne01=24, .ne02=1,
            .nb00=sizeof(uint16_t), .nb01=16384ull*sizeof(uint16_t),
            .nb02=16384ull*24ull*sizeof(uint16_t), .nb03=16384ull*24ull*sizeof(uint16_t),
            .ne10=16384, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=16384ull*sizeof(float),
            .nb12=16384ull*sizeof(float), .nb13=16384ull*sizeof(float),
            .ne0=24, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_hc_ingress_args ffn_hc_args = {
            .n_embd=4096, .n_hc=4, .sinkhorn_iters=20,
            .n_rows=1, .mix_hc=24,
            .nb_mix1=24u*sizeof(float), .nb_split1=24u*sizeof(float),
            .nb_x0=sizeof(float), .nb_x1=4096u*sizeof(float),
            .nb_x2=16384u*sizeof(float), .nb0=sizeof(float),
            .nb1=4096u*sizeof(float), .nb_norm1=4096u*sizeof(float),
            .eps=1.0e-6f, .norm_eps=1.0e-6f,
        };
        rust_star_q8_mv_args router_gate_mv = {
            .ne00=4096, .ne01=256, .ne02=1,
            .nb00=sizeof(uint16_t), .nb01=4096ull*sizeof(uint16_t),
            .nb02=4096ull*256ull*sizeof(uint16_t), .nb03=4096ull*256ull*sizeof(uint16_t),
            .ne10=4096, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=4096ull*sizeof(float),
            .nb12=4096ull*sizeof(float), .nb13=4096ull*sizeof(float),
            .ne0=256, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_unary_args router_unary_args = {
            .ne00=64, .ne01=1, .ne02=1, .ne03=1,
            .nb00=sizeof(float), .nb01=256u*sizeof(float),
            .nb02=256u*sizeof(float), .nb03=256u*sizeof(float),
            .ne0=64, .ne1=1, .ne2=1, .ne3=1,
            .nb0=sizeof(float), .nb1=256u*sizeof(float),
            .nb2=256u*sizeof(float), .nb3=256u*sizeof(float),
        };
        const BOOL hash_router = layer0->layer_index < 3;
        rust_star_router_select_one_args layer_router_args = {
            .has_bias=hash_router ? 0u : 1u,
            .hash_mode=hash_router ? 1u : 0u,
            .use_token_buffer=0,
            .token=token,
            .hash_rows=hash_router ? 129280u : 0u,
        };
        rust_star_q8_mv_id_args routed_gate_args = {
            .nei0=6, .nei1=1, .nbi1=6u*sizeof(int32_t),
            .ne00=4096, .ne01=2048, .ne02=256,
            .nb00=66, .nb01=1056, .nb02=2048ull*1056ull,
            .ne10=4096, .ne11=1, .ne12=1, .ne13=1,
            .nb10=sizeof(float), .nb11=4096u*sizeof(float),
            .nb12=4096u*sizeof(float), .ne0=2048, .ne1=6,
            .nb1=2048u*sizeof(float), .nr0=4,
            .tp_rank=0, .tp_world=0, .tp_addend=0, .tp_expert_base=0,
        };
        rust_star_moe_swiglu_weight_args routed_activation_args = {
            .width=2048, .rows=6,
            .gate_row_stride=2048u*sizeof(float), .up_row_stride=2048u*sizeof(float),
            .mid_row_stride=2048u*sizeof(float), .weight_stride=sizeof(float),
            .write_clamped=0, .clamp_value=10.0f,
        };
        rust_star_q8_mv_id_args routed_down_args = {
            .nei0=6, .nei1=1, .nbi1=6u*sizeof(int32_t),
            .ne00=2048, .ne01=4096, .ne02=256,
            .nb00=84, .nb01=672, .nb02=4096ull*672ull,
            .ne10=2048, .ne11=6, .ne12=1, .ne13=1,
            .nb10=sizeof(float), .nb11=2048u*sizeof(float),
            .nb12=6u*2048u*sizeof(float), .ne0=4096, .ne1=6,
            .nb1=4096u*sizeof(float), .nr0=4,
            .tp_rank=0, .tp_world=0, .tp_addend=0, .tp_expert_base=0,
        };
        rust_star_q8_mv_args shared_gate_mv = {
            .ne00=4096, .ne01=2048, .ne02=1,
            .nb00=34, .nb01=4352, .nb02=2048ull*4352ull, .nb03=2048ull*4352ull,
            .ne10=4096, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=4096u*sizeof(float),
            .nb12=4096u*sizeof(float), .nb13=4096u*sizeof(float),
            .ne0=2048, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_q8_mv_args shared_down_mv = {
            .ne00=2048, .ne01=4096, .ne02=1,
            .nb00=34, .nb01=2176, .nb02=4096ull*2176ull, .nb03=4096ull*2176ull,
            .ne10=2048, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=2048u*sizeof(float),
            .nb12=2048u*sizeof(float), .nb13=2048u*sizeof(float),
            .ne0=4096, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_hc_expand_args ffn_hc_post_args = output_hc_args;
        ffn_hc_post_args.has_add = 1;

        const uint32_t total_iterations = chained_replay
            ? 0 : warmup_iterations + measured_iterations;
        double measured_wall_ms = 0.0;
        double measured_gpu_ms = 0.0;
        if (full_layer && layer0->repeat_bitwise_matches) {
            *layer0->repeat_bitwise_matches = 1;
        }
        id<MTLCommandBuffer> command = nil;
        if (chained_replay) {
            command = context.chainedCommands[chained_key];
            NSNumber *wall_start = context.chainedWallStarts[chained_key];
            if (!wall_start || command.status != MTLCommandBufferStatusCompleted) {
                return fail_with_message(error, error_bytes,
                    @"chained command buffer is not ready for collection");
            }
            measured_wall_ms = context.chainedWallEnd-wall_start.doubleValue;
            if (chained_timing) {
                for (uint32_t layer = 0; layer <= layer0->chain_final_layer; layer++) {
                    measured_gpu_ms += gpu_elapsed_ms(context.chainedCommands[@(layer)]);
                }
            } else {
                measured_gpu_ms = gpu_elapsed_ms(command);
            }
        }
        for (uint32_t execution = 0; execution < total_iterations; execution++) {
        if (execution > 0 && rope_and_store && position == 1u) {
            float *cache = cache_buffer.contents;
            for (uint32_t index = 0; index < cache_capacity_rows*kv_elements; index++) cache[index] = -12345.5f;
            if (attention_read) memcpy(cache, cache_row0, kv_elements*sizeof(float));
        }
        const double wall_start = monotonic_ms();
        command = [context.queue commandBuffer];
        if (!command) return fail_with_message(error, error_bytes, @"failed to create attention ingress command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return fail_with_message(error, error_bytes, @"failed to create attention ingress encoder");

        if (compressed_layer && !cold_initial_state && position == 1u) {
            if (!encode_compressor_step(context, encoder,
                    compressor_prime_buffer, 0,
                    compressor_kv_weight, compressor_inner[1],
                    compressor_gate_weight, compressor_inner[2],
                    compressor_ape, compressor_inner[0],
                    compressor_norm_weight, compressor_inner[3],
                    compressor_kv_cur, compressor_score_cur,
                    compressor_state_kv, compressor_state_score,
                    compressor_packed_kv, compressor_packed_score,
                    compressor_softmax, compressed_kv_buffer,
                    compressor_width, 512u, compressor_ratio, 0u, NO, NO)) {
                return fail_with_message(error, error_bytes, @"failed to encode attention compressor prime");
            }
            if (indexer_layer && !encode_compressor_step(context, encoder,
                    compressor_prime_buffer, 0,
                    indexer_kv_weight, indexer_inner[1],
                    indexer_gate_weight, indexer_inner[2],
                    indexer_ape, indexer_inner[0],
                    indexer_norm_weight, indexer_inner[3],
                    indexer_kv_cur, indexer_score_cur,
                    indexer_state_kv, indexer_state_score,
                    indexer_packed_kv, indexer_packed_score,
                    indexer_softmax, compressed_indexer_buffer,
                    256u, 128u, 4u, 0u, NO, YES)) {
                return fail_with_message(error, error_bytes, @"failed to encode indexer compressor prime");
            }
        }

        if (!continuing_layer) {
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
        }

        [encoder setComputePipelineState:context.rmsNormF32Pipeline];
        [encoder setBytes:&norm length:sizeof(norm) atIndex:0];
        [encoder setBuffer:layer_input_hc offset:0 atIndex:1];
        [encoder setBuffer:layer_input_hc offset:0 atIndex:2];
        [encoder setBuffer:layer_input_hc offset:0 atIndex:3];
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
        [encoder setBuffer:layer_input_hc offset:0 atIndex:4];
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
            int32_t rope_position = (int32_t)position;
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

            if (compressed_layer) {
                if (!encode_compressor_step(context, encoder, norm_buffer, 0,
                        compressor_kv_weight, compressor_inner[1],
                        compressor_gate_weight, compressor_inner[2],
                        compressor_ape, compressor_inner[0],
                        compressor_norm_weight, compressor_inner[3],
                        compressor_kv_cur, compressor_score_cur,
                        compressor_state_kv, compressor_state_score,
                        compressor_packed_kv, compressor_packed_score,
                        compressor_softmax, compressed_kv_buffer,
                        compressor_width, 512u, compressor_ratio, position,
                        compressor_emit, NO)) {
                    return fail_with_message(error, error_bytes, @"failed to encode attention compressor update");
                }
                if (indexer_layer && !encode_compressor_step(context, encoder, norm_buffer, 0,
                        indexer_kv_weight, indexer_inner[1],
                        indexer_gate_weight, indexer_inner[2],
                        indexer_ape, indexer_inner[0],
                        indexer_norm_weight, indexer_inner[3],
                        indexer_kv_cur, indexer_score_cur,
                        indexer_state_kv, indexer_state_score,
                        indexer_packed_kv, indexer_packed_score,
                        indexer_softmax, compressed_indexer_buffer,
                        256u, 128u, 4u, position, compressor_emit, YES)) {
                    return fail_with_message(error, error_bytes, @"failed to encode indexer compressor update");
                }
            }

            if (attention_read) {
                const uint32_t tail_rows = cache_capacity_rows - raw_cache_start <
                    visible_cache_rows
                    ? cache_capacity_rows - raw_cache_start
                    : visible_cache_rows;
                const uint32_t head_rows = visible_cache_rows - tail_rows;
                uint32_t staged_elements = tail_rows*kv_elements;
                [encoder setComputePipelineState:context.cpyF32F16Pipeline];
                [encoder setBytes:&staged_elements length:sizeof(staged_elements) atIndex:0];
                [encoder setBuffer:cache_buffer
                             offset:raw_cache_start*kv_elements*sizeof(float)
                            atIndex:1];
                [encoder setBuffer:staged_kv_buffer offset:0 atIndex:2];
                const NSUInteger staged_groups = (staged_elements + 1023u) / 1024u;
                [encoder dispatchThreadgroups:MTLSizeMake(staged_groups,1,1)
                     threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                if (head_rows != 0u) {
                    staged_elements = head_rows*kv_elements;
                    [encoder setBytes:&staged_elements length:sizeof(staged_elements) atIndex:0];
                    [encoder setBuffer:cache_buffer offset:0 atIndex:1];
                    [encoder setBuffer:staged_kv_buffer
                                 offset:tail_rows*kv_elements*sizeof(uint16_t)
                                atIndex:2];
                    const NSUInteger head_groups = (staged_elements + 1023u) / 1024u;
                    [encoder dispatchThreadgroups:MTLSizeMake(head_groups,1,1)
                         threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                }
                const uint32_t prior_compressed_rows = compressed_cache_rows -
                    (compressor_emit ? 1u : 0u);
                if (prior_compressed_rows != 0u) {
                    uint32_t compressed_elements = prior_compressed_rows*kv_elements;
                    [encoder setComputePipelineState:context.cpyF32F16Pipeline];
                    [encoder setBytes:&compressed_elements length:sizeof(compressed_elements) atIndex:0];
                    [encoder setBuffer:compressed_kv_cache offset:0 atIndex:1];
                    [encoder setBuffer:staged_kv_buffer
                                 offset:visible_cache_rows*kv_elements*sizeof(uint16_t)
                                atIndex:2];
                    const NSUInteger compressed_groups = (compressed_elements + 1023u) / 1024u;
                    [encoder dispatchThreadgroups:MTLSizeMake(compressed_groups,1,1)
                         threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                }
                if (compressor_emit) {
                    uint32_t compressed_elements = kv_elements;
                    [encoder setComputePipelineState:context.cpyF32F16Pipeline];
                    [encoder setBytes:&compressed_elements length:sizeof(compressed_elements) atIndex:0];
                    [encoder setBuffer:compressed_kv_buffer offset:0 atIndex:1];
                    [encoder setBuffer:staged_kv_buffer
                                 offset:(visible_cache_rows + prior_compressed_rows)*
                                    kv_elements*sizeof(uint16_t)
                                atIndex:2];
                    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                         threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                }

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

            if (compressor_emit) {
                const NSUInteger compressed_row = compressed_cache_rows - 1u;
                id<MTLBlitCommandEncoder> compressed_copy = [command blitCommandEncoder];
                if (!compressed_copy) return fail_with_message(error, error_bytes,
                    @"failed to create compressed-cache commit encoder");
                [compressed_copy copyFromBuffer:compressed_kv_buffer sourceOffset:0
                                       toBuffer:compressed_kv_cache
                              destinationOffset:compressed_row*kv_elements*sizeof(float)
                                          size:kv_elements*sizeof(float)];
                if (indexer_layer) {
                    [compressed_copy copyFromBuffer:compressed_indexer_buffer sourceOffset:0
                                           toBuffer:compressed_indexer_cache
                                  destinationOffset:compressed_row*128u*sizeof(float)
                                              size:128u*sizeof(float)];
                }
                [compressed_copy endEncoding];
            }

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
                int32_t inverse_position = (int32_t)position;
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
                    [encoder setBuffer:layer_input_hc offset:0 atIndex:5];
                    [encoder setBuffer:split_buffer offset:4u*sizeof(float) atIndex:6];
                    [encoder setBuffer:split_buffer offset:8u*sizeof(float) atIndex:7];
                    [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:8];
                    [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(2048,1,1)
                         threadsPerThreadgroup:MTLSizeMake(32,4,1)];
                }
                if (full_layer) {
                    [encoder setComputePipelineState:context.rmsNormF32Pipeline];
                    [encoder setBytes:&ffn_norm_args length:sizeof(ffn_norm_args) atIndex:0];
                    [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:1];
                    [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:2];
                    [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:3];
                    [encoder setBuffer:ffn_flat_buffer offset:0 atIndex:4];
                    [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                         threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

                    [encoder setComputePipelineState:context.f16ProjectionPipeline];
                    [encoder setBytes:&ffn_hc_mv length:sizeof(ffn_hc_mv) atIndex:0];
                    [encoder setBuffer:ffn_hc_fn offset:layer_inner[0] atIndex:1];
                    [encoder setBuffer:ffn_flat_buffer offset:0 atIndex:2];
                    [encoder setBuffer:ffn_mix_buffer offset:0 atIndex:3];
                    [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(12,1,1)
                         threadsPerThreadgroup:MTLSizeMake(32,8,1)];

                    [encoder setComputePipelineState:context.hcIngressPipeline];
                    [encoder setBytes:&ffn_hc_args length:sizeof(ffn_hc_args) atIndex:0];
                    [encoder setBuffer:ffn_mix_buffer offset:0 atIndex:1];
                    [encoder setBuffer:ffn_hc_scale offset:layer_inner[1] atIndex:2];
                    [encoder setBuffer:ffn_hc_base offset:layer_inner[2] atIndex:3];
                    [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:4];
                    [encoder setBuffer:ffn_split_buffer offset:0 atIndex:5];
                    [encoder setBuffer:ffn_cur_buffer offset:0 atIndex:6];
                    [encoder setBuffer:ffn_norm_weight offset:layer_inner[3] atIndex:7];
                    [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:8];
                    [encoder setThreadgroupMemoryLength:(4096u+4u+32u)*sizeof(float) atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                         threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

                    [encoder setComputePipelineState:context.f16ProjectionPipeline];
                    [encoder setBytes:&router_gate_mv length:sizeof(router_gate_mv) atIndex:0];
                    [encoder setBuffer:router_gate offset:layer_inner[4] atIndex:1];
                    [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:2];
                    [encoder setBuffer:router_logits_buffer offset:0 atIndex:3];
                    [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(128,1,1)
                         threadsPerThreadgroup:MTLSizeMake(32,8,1)];

                    [encoder setComputePipelineState:context.routerProbabilityPipeline];
                    [encoder setBytes:&router_unary_args length:sizeof(router_unary_args) atIndex:0];
                    [encoder setBuffer:router_logits_buffer offset:0 atIndex:1];
                    [encoder setBuffer:router_probs_buffer offset:0 atIndex:2];
                    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                         threadsPerThreadgroup:MTLSizeMake(64,1,1)];

                    [encoder setComputePipelineState:context.routerFinalizePipeline];
                    [encoder setBytes:&layer_router_args length:sizeof(layer_router_args) atIndex:0];
                    [encoder setBuffer:router_probs_buffer offset:0 atIndex:1];
                    float zero_bias = 0.0f;
                    int32_t zero_id = 0;
                    if (hash_router) {
                        [encoder setBytes:&zero_bias length:sizeof(zero_bias) atIndex:2];
                        [encoder setBuffer:router_aux offset:layer_inner[5] atIndex:3];
                    } else {
                        [encoder setBuffer:router_aux offset:layer_inner[5] atIndex:2];
                        [encoder setBytes:&zero_id length:sizeof(zero_id) atIndex:3];
                    }
                    [encoder setBytes:&zero_id length:sizeof(zero_id) atIndex:4];
                    [encoder setBuffer:selected_buffer offset:0 atIndex:5];
                    [encoder setThreadgroupMemoryLength:256u*(sizeof(float)+sizeof(int32_t)) atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                         threadsPerThreadgroup:MTLSizeMake(256,1,1)];

                    [encoder setComputePipelineState:context.routerWeightsPipeline];
                    [encoder setBuffer:router_probs_buffer offset:0 atIndex:0];
                    [encoder setBuffer:selected_buffer offset:0 atIndex:1];
                    [encoder setBuffer:route_weights_buffer offset:0 atIndex:2];
                    [encoder dispatchThreads:MTLSizeMake(6,1,1)
                         threadsPerThreadgroup:MTLSizeMake(6,1,1)];
                }
                [encoder endEncoding];

                if (full_layer) {
                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:context.routedPairSwigluPipeline];
                    [encoder setBytes:&routed_gate_args length:sizeof(routed_gate_args) atIndex:0];
                    [encoder setBytes:&routed_activation_args length:sizeof(routed_activation_args) atIndex:1];
                    [encoder setBuffer:routed_gate_weight offset:layer_inner[6] atIndex:2];
                    [encoder setBuffer:routed_up_weight offset:layer_inner[7] atIndex:3];
                    [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:4];
                    [encoder setBuffer:routed_gate_buffer offset:0 atIndex:5];
                    [encoder setBuffer:routed_up_buffer offset:0 atIndex:6];
                    [encoder setBuffer:routed_mid_buffer offset:0 atIndex:7];
                    [encoder setBuffer:selected_buffer offset:0 atIndex:8];
                    [encoder setBuffer:route_weights_buffer offset:0 atIndex:9];
                    [encoder setThreadgroupMemoryLength:2176u atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(256,1,6)
                         threadsPerThreadgroup:MTLSizeMake(32,2,1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:context.routedDownSumPipeline];
                    [encoder setBytes:&routed_down_args length:sizeof(routed_down_args) atIndex:0];
                    [encoder setBuffer:routed_down_weight offset:layer_inner[8] atIndex:1];
                    [encoder setBuffer:routed_mid_buffer offset:0 atIndex:2];
                    [encoder setBuffer:routed_out_buffer offset:0 atIndex:3];
                    [encoder setBuffer:selected_buffer offset:0 atIndex:4];
                    [encoder setBuffer:routed_out_buffer offset:0 atIndex:5];
                    [encoder dispatchThreadgroups:MTLSizeMake(512,1,1)
                         threadsPerThreadgroup:MTLSizeMake(32,2,1)];
                    [encoder endEncoding];

                    const float layer_clamp = 10.0f;
                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:context.sharedGateUpPipeline];
                    [encoder setBytes:&shared_gate_mv length:sizeof(shared_gate_mv) atIndex:0];
                    [encoder setBuffer:shared_gate_weight offset:layer_inner[9] atIndex:1];
                    [encoder setBuffer:shared_up_weight offset:layer_inner[10] atIndex:2];
                    [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:3];
                    [encoder setBuffer:shared_gate_buffer offset:0 atIndex:4];
                    [encoder setBuffer:shared_up_buffer offset:0 atIndex:5];
                    [encoder setBuffer:shared_mid_buffer offset:0 atIndex:6];
                    [encoder setBytes:&layer_clamp length:sizeof(layer_clamp) atIndex:7];
                    [encoder setThreadgroupMemoryLength:512u atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(1024,1,1)
                         threadsPerThreadgroup:MTLSizeMake(32,4,1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:context.sharedDownHcPipeline];
                    [encoder setBytes:&shared_down_mv length:sizeof(shared_down_mv) atIndex:0];
                    [encoder setBytes:&ffn_hc_post_args length:sizeof(ffn_hc_post_args) atIndex:1];
                    [encoder setBuffer:shared_down_weight offset:layer_inner[11] atIndex:2];
                    [encoder setBuffer:shared_mid_buffer offset:0 atIndex:3];
                    [encoder setBuffer:shared_out_buffer offset:0 atIndex:4];
                    [encoder setBuffer:routed_out_buffer offset:0 atIndex:5];
                    [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:6];
                    [encoder setBuffer:ffn_split_buffer offset:4u*sizeof(float) atIndex:7];
                    [encoder setBuffer:ffn_split_buffer offset:8u*sizeof(float) atIndex:8];
                    [encoder setBuffer:after_ffn_hc_buffer offset:0 atIndex:9];
                    [encoder setThreadgroupMemoryLength:256u atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake(2048,1,1)
                         threadsPerThreadgroup:MTLSizeMake(32,4,1)];
                    [encoder endEncoding];
                }
            }
        }

        [command commit];
        if (chained_submission) {
            context.chainedCommands[chained_key] = command;
            context.chainedWallStarts[chained_key] = @(wall_start);
            if (command_mode == RUST_STAR_COMMAND_CHAINED_FINAL) {
                if (!command_succeeded(command, error, error_bytes)) return 0;
                context.chainedWallEnd = monotonic_ms();
                for (uint32_t layer = 0; layer <= layer0->chain_final_layer; layer++) {
                    id<MTLCommandBuffer> chained = context.chainedCommands[@(layer)];
                    if (!chained || chained.status == MTLCommandBufferStatusError) {
                        return fail_with_message(error, error_bytes,
                            chained.error.localizedDescription ?: @"a chained command buffer failed");
                    }
                }
                context.chainedReady = YES;
            }
            continue;
        }
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        if (execution >= warmup_iterations) {
            const uint32_t measured = execution-warmup_iterations;
            const double iteration_wall_ms = wall_end-wall_start;
            const double iteration_gpu_ms = gpu_elapsed_ms(command);
            measured_wall_ms += iteration_wall_ms;
            measured_gpu_ms += iteration_gpu_ms;
            if (full_layer && layer0->wall_ms_samples) {
                layer0->wall_ms_samples[measured] = iteration_wall_ms;
                layer0->gpu_ms_samples[measured] = iteration_gpu_ms;
            }
            if (full_layer && measured == 0 && layer0->repeat_bitwise_matches) {
                memcpy(after_attention_hc, after_attention_hc_buffer.contents, hc_dim*sizeof(float));
                memcpy(layer0->ffn_mixes, ffn_mix_buffer.contents, mix_hc*sizeof(float));
                memcpy(layer0->ffn_split, ffn_split_buffer.contents, mix_hc*sizeof(float));
                memcpy(layer0->ffn_norm, ffn_norm_buffer.contents, n_embd*sizeof(float));
                memcpy(layer0->router_logits, router_logits_buffer.contents, 256u*sizeof(float));
                memcpy(layer0->router_probs, router_probs_buffer.contents, 256u*sizeof(float));
                memcpy(layer0->selected, selected_buffer.contents, 6u*sizeof(int32_t));
                memcpy(layer0->router_weights, route_weights_buffer.contents, 6u*sizeof(float));
                memcpy(layer0->routed_mid, routed_mid_buffer.contents, 6u*2048u*sizeof(float));
                memcpy(layer0->routed_out, routed_out_buffer.contents, n_embd*sizeof(float));
                memcpy(layer0->shared_out, shared_out_buffer.contents, n_embd*sizeof(float));
                memcpy(layer0->after_ffn_hc, after_ffn_hc_buffer.contents, hc_dim*sizeof(float));
            } else if (full_layer && layer0->repeat_bitwise_matches &&
                (memcmp(after_attention_hc, after_attention_hc_buffer.contents, hc_dim*sizeof(float)) != 0 ||
                 memcmp(layer0->ffn_mixes, ffn_mix_buffer.contents, mix_hc*sizeof(float)) != 0 ||
                 memcmp(layer0->ffn_split, ffn_split_buffer.contents, mix_hc*sizeof(float)) != 0 ||
                 memcmp(layer0->ffn_norm, ffn_norm_buffer.contents, n_embd*sizeof(float)) != 0 ||
                 memcmp(layer0->router_logits, router_logits_buffer.contents, 256u*sizeof(float)) != 0 ||
                 memcmp(layer0->router_probs, router_probs_buffer.contents, 256u*sizeof(float)) != 0 ||
                 memcmp(layer0->selected, selected_buffer.contents, 6u*sizeof(int32_t)) != 0 ||
                 memcmp(layer0->router_weights, route_weights_buffer.contents, 6u*sizeof(float)) != 0 ||
                 memcmp(layer0->routed_mid, routed_mid_buffer.contents, 6u*2048u*sizeof(float)) != 0 ||
                 memcmp(layer0->routed_out, routed_out_buffer.contents, n_embd*sizeof(float)) != 0 ||
                 memcmp(layer0->shared_out, shared_out_buffer.contents, n_embd*sizeof(float)) != 0 ||
                 memcmp(layer0->after_ffn_hc, after_ffn_hc_buffer.contents, hc_dim*sizeof(float)) != 0)) {
                *layer0->repeat_bitwise_matches = 0;
            }
        }
        }
        if (chained_submission) {
            result->model_bytes = model_bytes;
            result->max_buffer_length = context.device.maxBufferLength;
            result->wrapped_model_ranges = 25u + (compressed_layer ? 4u : 0u) +
                (indexer_layer ? 4u : 0u);
            for (uint32_t index = 0; index < result->wrapped_model_ranges; index++) {
                if (matches[index]) result->pointer_matches++;
            }
            return 1;
        }
        if (chained_timing) {
            result->model_bytes = model_bytes;
            result->max_buffer_length = context.device.maxBufferLength;
            result->wrapped_model_ranges = 25u + (compressed_layer ? 4u : 0u) +
                (indexer_layer ? 4u : 0u);
            for (uint32_t index = 0; index < result->wrapped_model_ranges; index++) {
                if (matches[index]) result->pointer_matches++;
            }
            result->wall_ms = measured_wall_ms;
            result->gpu_ms = measured_gpu_ms;
            return 1;
        }
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
            memcpy(cache_rows, cache_buffer.contents, cache_capacity_rows*kv_elements*sizeof(float));
            if (compressed_layer) {
                memcpy(layer0->compressed_kv_row, compressed_kv_buffer.contents,
                    kv_elements*sizeof(float));
            }
            if (indexer_layer) {
                memcpy(layer0->compressed_indexer_row, compressed_indexer_buffer.contents,
                    128u*sizeof(float));
            }
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
        if (full_layer) {
            memcpy(layer0->ffn_mixes, ffn_mix_buffer.contents, mix_hc*sizeof(float));
            memcpy(layer0->ffn_split, ffn_split_buffer.contents, mix_hc*sizeof(float));
            memcpy(layer0->ffn_norm, ffn_norm_buffer.contents, n_embd*sizeof(float));
            memcpy(layer0->router_logits, router_logits_buffer.contents, 256u*sizeof(float));
            memcpy(layer0->router_probs, router_probs_buffer.contents, 256u*sizeof(float));
            memcpy(layer0->selected, selected_buffer.contents, 6u*sizeof(int32_t));
            memcpy(layer0->router_weights, route_weights_buffer.contents, 6u*sizeof(float));
            memcpy(layer0->routed_mid, routed_mid_buffer.contents, 6u*2048u*sizeof(float));
            memcpy(layer0->routed_out, routed_out_buffer.contents, n_embd*sizeof(float));
            memcpy(layer0->shared_out, shared_out_buffer.contents, n_embd*sizeof(float));
            memcpy(layer0->after_ffn_hc, after_ffn_hc_buffer.contents, hc_dim*sizeof(float));
        }
        result->model_bytes = model_bytes;
        result->max_buffer_length = context.device.maxBufferLength;
        result->wrapped_model_ranges = full_layer
            ? 25u + (compressed_layer ? 4u : 0u) + (indexer_layer ? 4u : 0u) :
            (attention_output ? 13 : (attention_read ? 11 : (extended ? 10 : 6)));
        for (uint32_t index = 0; index < result->wrapped_model_ranges; index++) if (matches[index]) result->pointer_matches++;
        result->wall_ms = measured_wall_ms/measured_iterations;
        result->gpu_ms = measured_gpu_ms/measured_iterations;
        return 1;
    }
}

int rust_star_metal_run_output_head(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t hc_fn_offset,
    uint64_t hc_fn_bytes,
    uint64_t hc_scale_offset,
    uint64_t hc_scale_bytes,
    uint64_t hc_base_offset,
    uint64_t hc_base_bytes,
    uint64_t output_norm_offset,
    uint64_t output_norm_bytes,
    uint64_t output_offset,
    uint64_t output_bytes,
    float *hc_pre,
    float *hc_weights,
    float *hc,
    float *norm,
    float *logits,
    uint32_t collect_intermediates,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes)
{
    const uint32_t n_embd = 4096u;
    const uint32_t n_hc = 4u;
    const uint32_t hc_dim = n_embd*n_hc;
    const uint32_t n_vocab = 129280u;
    const uint64_t output_row_bytes = (uint64_t)(n_embd/32u)*34u;
    if (!opaque_context || !model_mapping || !logits || !result ||
        collect_intermediates > 1u ||
        (collect_intermediates && (!hc_pre || !hc_weights || !hc || !norm))) {
        return fail_with_message(error, error_bytes, @"output head received invalid inputs");
    }
    if (hc_fn_bytes != (uint64_t)hc_dim*n_hc*sizeof(uint16_t) ||
        hc_scale_bytes != sizeof(float) ||
        hc_base_bytes != n_hc*sizeof(float) ||
        output_norm_bytes != n_embd*sizeof(float) ||
        output_bytes != (uint64_t)n_vocab*output_row_bytes) {
        return fail_with_message(error, error_bytes, @"output head tensor shapes are invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_attention_ingress_pipelines(context, error, error_bytes) ||
            !ensure_q8_projection_pipeline(context, error, error_bytes) ||
            !ensure_moe_output_pipelines(context, error, error_bytes)) return 0;
        if (!context.chainedReady || context.chainedFinalLayer != 42u) {
            return fail_with_message(error, error_bytes,
                @"output head requires a completed layers-0-through-42 chain");
        }
        id<MTLBuffer> input_hc =
            context.activationBufferCache[layer_buffer_key(@"layer_hc_state", YES, 42u)];
        if (!input_hc || input_hc.length != hc_dim*sizeof(float)) {
            return fail_with_message(error, error_bytes,
                @"output head could not find the retained layer-42 HC state");
        }

        NSUInteger inner[5] = {0};
        BOOL matches[5] = {NO};
        id<MTLBuffer> hc_fn_weight = wrap_model_range(context, model_mapping, model_bytes,
            hc_fn_offset, hc_fn_bytes, &inner[0], &matches[0], error, error_bytes);
        id<MTLBuffer> hc_scale_weight = wrap_model_range(context, model_mapping, model_bytes,
            hc_scale_offset, hc_scale_bytes, &inner[1], &matches[1], error, error_bytes);
        id<MTLBuffer> hc_base_weight = wrap_model_range(context, model_mapping, model_bytes,
            hc_base_offset, hc_base_bytes, &inner[2], &matches[2], error, error_bytes);
        id<MTLBuffer> norm_weight = wrap_model_range(context, model_mapping, model_bytes,
            output_norm_offset, output_norm_bytes, &inner[3], &matches[3], error, error_bytes);
        id<MTLBuffer> output_weight = wrap_model_range(context, model_mapping, model_bytes,
            output_offset, output_bytes, &inner[4], &matches[4], error, error_bytes);
        if (!hc_fn_weight || !hc_scale_weight || !hc_base_weight ||
            !norm_weight || !output_weight) return 0;

        id<MTLBuffer> flat_buffer = persistent_buffer(context, @"output_flat_hc",
            hc_dim*sizeof(float), error, error_bytes);
        id<MTLBuffer> pre_buffer = persistent_buffer(context, @"output_hc_pre",
            n_hc*sizeof(float), error, error_bytes);
        id<MTLBuffer> weights_buffer = persistent_buffer(context, @"output_hc_weights",
            n_hc*sizeof(float), error, error_bytes);
        id<MTLBuffer> hc_buffer = persistent_buffer(context, @"output_hc",
            n_embd*sizeof(float), error, error_bytes);
        id<MTLBuffer> norm_buffer = persistent_buffer(context, @"output_norm",
            n_embd*sizeof(float), error, error_bytes);
        id<MTLBuffer> logits_buffer = persistent_buffer(context, @"output_logits",
            n_vocab*sizeof(float), error, error_bytes);
        if (!flat_buffer || !pre_buffer || !weights_buffer || !hc_buffer ||
            !norm_buffer || !logits_buffer) return 0;

        const uint64_t hc_bytes = (uint64_t)hc_dim*sizeof(float);
        rust_star_norm_args plain_norm = {
            .ne00=(int32_t)hc_dim, .ne00_t=(int32_t)(hc_dim/4u),
            .nb1=hc_bytes, .nb2=hc_bytes, .nb3=hc_bytes, .eps=1.0e-6f,
            .nef1={1,1,1}, .nef2={1,1,1}, .nef3={1,1,1},
            .nbf1={hc_bytes,hc_bytes,hc_bytes},
            .nbf2={hc_bytes,hc_bytes,hc_bytes},
            .nbf3={hc_bytes,hc_bytes,hc_bytes},
        };
        rust_star_q8_mv_args hc_projection = {
            .ne00=(int32_t)hc_dim, .ne01=(int32_t)n_hc, .ne02=1,
            .nb00=sizeof(uint16_t), .nb01=(uint64_t)hc_dim*sizeof(uint16_t),
            .nb02=hc_fn_bytes, .nb03=hc_fn_bytes,
            .ne10=(int32_t)hc_dim, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=hc_bytes, .nb12=hc_bytes, .nb13=hc_bytes,
            .ne0=(int32_t)n_hc, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_output_hc_weights_args weights_args = {
            .post_scale=1.0f, .eps=1.0e-6f,
        };
        rust_star_hc_weighted_sum_norm_args sum_norm = {
            .n_embd=n_embd, .n_hc=n_hc, .n_tokens=1,
            .nb_x0=sizeof(float), .nb_x1=(uint64_t)n_embd*sizeof(float),
            .nb_x2=(uint64_t)hc_dim*sizeof(float),
            .nb_w0=sizeof(float), .nb_w1=n_hc*sizeof(float),
            .nb0=sizeof(float), .nb1=(uint64_t)n_embd*sizeof(float),
            .nb_norm1=(uint64_t)n_embd*sizeof(float), .norm_eps=1.0e-6f,
        };
        rust_star_q8_mv_args vocab_projection = {
            .ne00=(int32_t)n_embd, .ne01=(int32_t)n_vocab, .ne02=1,
            .nb00=34, .nb01=output_row_bytes,
            .nb02=output_bytes, .nb03=output_bytes,
            .ne10=(int32_t)n_embd, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=(uint64_t)n_embd*sizeof(float),
            .nb12=(uint64_t)n_embd*sizeof(float), .nb13=(uint64_t)n_embd*sizeof(float),
            .ne0=(int32_t)n_vocab, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        if (!command) {
            return fail_with_message(error, error_bytes,
                @"failed to create output-head command buffer");
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            return fail_with_message(error, error_bytes,
                @"failed to create output-head command encoder");
        }

        [encoder setComputePipelineState:context.rmsNormF32Pipeline];
        [encoder setBytes:&plain_norm length:sizeof(plain_norm) atIndex:0];
        [encoder setBuffer:input_hc offset:0 atIndex:1];
        [encoder setBuffer:input_hc offset:0 atIndex:2];
        [encoder setBuffer:input_hc offset:0 atIndex:3];
        [encoder setBuffer:flat_buffer offset:0 atIndex:4];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16ProjectionPipeline];
        [encoder setBytes:&hc_projection length:sizeof(hc_projection) atIndex:0];
        [encoder setBuffer:hc_fn_weight offset:inner[0] atIndex:1];
        [encoder setBuffer:flat_buffer offset:0 atIndex:2];
        [encoder setBuffer:pre_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(2,1,1)
             threadsPerThreadgroup:MTLSizeMake(32,8,1)];

        [encoder setComputePipelineState:context.outputHcWeightsPipeline];
        [encoder setBytes:&weights_args length:sizeof(weights_args) atIndex:0];
        [encoder setBuffer:pre_buffer offset:0 atIndex:1];
        [encoder setBuffer:hc_scale_weight offset:inner[1] atIndex:2];
        [encoder setBuffer:hc_base_weight offset:inner[2] atIndex:3];
        [encoder setBuffer:weights_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(2,1,1)];

        [encoder setComputePipelineState:context.outputHcSumNormPipeline];
        [encoder setBytes:&sum_norm length:sizeof(sum_norm) atIndex:0];
        [encoder setBuffer:input_hc offset:0 atIndex:1];
        [encoder setBuffer:weights_buffer offset:0 atIndex:2];
        [encoder setBuffer:hc_buffer offset:0 atIndex:3];
        [encoder setBuffer:norm_weight offset:inner[3] atIndex:4];
        [encoder setBuffer:norm_buffer offset:0 atIndex:5];
        [encoder setThreadgroupMemoryLength:(n_embd+32u)*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.q8OutputProjectionPipeline];
        [encoder setBytes:&vocab_projection length:sizeof(vocab_projection) atIndex:0];
        [encoder setBuffer:output_weight offset:inner[4] atIndex:1];
        [encoder setBuffer:norm_buffer offset:0 atIndex:2];
        [encoder setBuffer:logits_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((n_vocab+1u)/2u,1,1)
             threadsPerThreadgroup:MTLSizeMake(32,8,1)];
        [encoder endEncoding];

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();

        if (collect_intermediates) {
            memcpy(hc_pre, pre_buffer.contents, n_hc*sizeof(float));
            memcpy(hc_weights, weights_buffer.contents, n_hc*sizeof(float));
            memcpy(hc, hc_buffer.contents, n_embd*sizeof(float));
            memcpy(norm, norm_buffer.contents, n_embd*sizeof(float));
        }
        memcpy(logits, logits_buffer.contents, n_vocab*sizeof(float));
        memset(result, 0, sizeof(*result));
        result->model_bytes = model_bytes;
        result->max_buffer_length = context.device.maxBufferLength;
        result->wrapped_model_ranges = 5;
        for (uint32_t i = 0; i < 5; i++) if (matches[i]) result->pointer_matches++;
        result->wall_ms = wall_end-wall_start;
        result->gpu_ms = gpu_elapsed_ms(command);
        return 1;
    }
}

int rust_star_metal_copy_compressed_kv_row(
    void *opaque_context,
    uint32_t layer_index,
    uint32_t row_index,
    float *output,
    uint64_t output_elements,
    char *error,
    size_t error_bytes)
{
    const uint32_t row_elements = 512u;
    if (!opaque_context || !output || output_elements != row_elements ||
        layer_index < 2u || layer_index > 42u) {
        return fail_with_message(error, error_bytes,
            @"compressed-cache readback received invalid inputs");
    }
    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        id<MTLBuffer> buffer = context.activationBufferCache[
            layer_buffer_key(@"compressed_kv_cache", YES, layer_index)];
        const NSUInteger row_bytes = (NSUInteger)row_elements*sizeof(float);
        if (!buffer || buffer.length % row_bytes != 0u ||
            row_index >= buffer.length/row_bytes) {
            return fail_with_message(error, error_bytes,
                @"compressed-cache readback could not find the retained layer cache");
        }
        const uint8_t *source = (const uint8_t *)buffer.contents +
            (NSUInteger)row_index*row_elements*sizeof(float);
        memcpy(output, source, row_elements*sizeof(float));
        return 1;
    }
}

int rust_star_metal_run_ffn_router(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t hc_fn_offset,
    uint64_t hc_fn_bytes,
    uint64_t hc_scale_offset,
    uint64_t hc_scale_bytes,
    uint64_t hc_base_offset,
    uint64_t hc_base_bytes,
    uint64_t ffn_norm_offset,
    uint64_t ffn_norm_bytes,
    uint64_t gate_offset,
    uint64_t gate_bytes,
    uint64_t bias_offset,
    uint64_t bias_bytes,
    uint64_t hash_offset,
    uint64_t hash_bytes,
    const float *after_attention_hc,
    float *mixes,
    float *split,
    float *ffn_cur,
    float *ffn_norm,
    float *logits,
    float *probs,
    int32_t *selected,
    float *weights,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes)
{
    const uint32_t n_embd = 4096, n_hc = 4, hc_dim = 16384;
    const uint32_t mix_hc = 24, n_expert = 256, n_used = 6;
    if (!opaque_context || !model_mapping || !after_attention_hc || !mixes ||
        !split || !ffn_cur || !ffn_norm || !logits || !probs || !selected ||
        !weights || !result) {
        return fail_with_message(error, error_bytes, @"FFN router probe received a null input");
    }
    if (hc_fn_bytes != (uint64_t)hc_dim*mix_hc*sizeof(uint16_t) ||
        hc_scale_bytes != 3u*sizeof(float) || hc_base_bytes != mix_hc*sizeof(float) ||
        ffn_norm_bytes != n_embd*sizeof(float) ||
        gate_bytes != (uint64_t)n_embd*n_expert*sizeof(uint16_t) ||
        (bias_bytes != 0 && bias_bytes != n_expert*sizeof(float)) ||
        (hash_bytes != 0 && hash_bytes != 6ull*129280ull*sizeof(int32_t))) {
        return fail_with_message(error, error_bytes, @"FFN router tensor shapes are invalid");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_attention_ingress_pipelines(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));

        NSUInteger inner[7] = {0};
        BOOL matches[7] = {NO};
        id<MTLBuffer> hc_fn = wrap_model_range(context, model_mapping, model_bytes,
            hc_fn_offset, hc_fn_bytes, &inner[0], &matches[0], error, error_bytes);
        id<MTLBuffer> hc_scale = wrap_model_range(context, model_mapping, model_bytes,
            hc_scale_offset, hc_scale_bytes, &inner[1], &matches[1], error, error_bytes);
        id<MTLBuffer> hc_base = wrap_model_range(context, model_mapping, model_bytes,
            hc_base_offset, hc_base_bytes, &inner[2], &matches[2], error, error_bytes);
        id<MTLBuffer> norm_weight = wrap_model_range(context, model_mapping, model_bytes,
            ffn_norm_offset, ffn_norm_bytes, &inner[3], &matches[3], error, error_bytes);
        id<MTLBuffer> gate = wrap_model_range(context, model_mapping, model_bytes,
            gate_offset, gate_bytes, &inner[4], &matches[4], error, error_bytes);
        id<MTLBuffer> bias = bias_bytes ? wrap_model_range(context, model_mapping, model_bytes,
            bias_offset, bias_bytes, &inner[5], &matches[5], error, error_bytes) : nil;
        id<MTLBuffer> hash = hash_bytes ? wrap_model_range(context, model_mapping, model_bytes,
            hash_offset, hash_bytes, &inner[6], &matches[6], error, error_bytes) : nil;
        if (!hc_fn || !hc_scale || !hc_base || !norm_weight || !gate ||
            (bias_bytes && !bias) || (hash_bytes && !hash)) return 0;

        id<MTLBuffer> after = [context.device newBufferWithBytes:after_attention_hc
            length:(NSUInteger)hc_dim*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> flat = [context.device newBufferWithLength:(NSUInteger)hc_dim*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> mix = [context.device newBufferWithLength:mix_hc*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> split_buffer = [context.device newBufferWithLength:mix_hc*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cur = [context.device newBufferWithLength:n_embd*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> norm = [context.device newBufferWithLength:n_embd*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> logits_buffer = [context.device newBufferWithLength:n_expert*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> probs_buffer = [context.device newBufferWithLength:n_expert*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> selected_buffer = [context.device newBufferWithLength:n_used*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> weights_buffer = [context.device newBufferWithLength:n_used*sizeof(float) options:MTLResourceStorageModeShared];
        if (!after || !flat || !mix || !split_buffer || !cur || !norm ||
            !logits_buffer || !probs_buffer || !selected_buffer || !weights_buffer) {
            return fail_with_message(error, error_bytes, @"failed to allocate FFN router buffers");
        }

        const uint64_t hc_row_bytes = (uint64_t)hc_dim*sizeof(float);
        rust_star_norm_args norm_args = {
            .ne00=(int32_t)hc_dim, .ne00_t=(int32_t)(hc_dim/4u),
            .nb1=hc_row_bytes, .nb2=hc_row_bytes, .nb3=hc_row_bytes, .eps=1.0e-6f,
            .nef1={1,1,1}, .nef2={1,1,1}, .nef3={1,1,1},
            .nbf1={hc_row_bytes,hc_row_bytes,hc_row_bytes},
            .nbf2={hc_row_bytes,hc_row_bytes,hc_row_bytes},
            .nbf3={hc_row_bytes,hc_row_bytes,hc_row_bytes},
        };
        const uint64_t hc_weight_row_bytes = (uint64_t)hc_dim*sizeof(uint16_t);
        rust_star_q8_mv_args hc_mv = {
            .ne00=(int32_t)hc_dim, .ne01=(int32_t)mix_hc, .ne02=1,
            .nb00=sizeof(uint16_t), .nb01=hc_weight_row_bytes,
            .nb02=hc_fn_bytes, .nb03=hc_fn_bytes,
            .ne10=(int32_t)hc_dim, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=hc_row_bytes, .nb12=hc_row_bytes, .nb13=hc_row_bytes,
            .ne0=(int32_t)mix_hc, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_hc_ingress_args hc_args = {
            .n_embd=n_embd, .n_hc=(int32_t)n_hc, .sinkhorn_iters=20,
            .n_rows=1, .mix_hc=mix_hc,
            .nb_mix1=mix_hc*sizeof(float), .nb_split1=mix_hc*sizeof(float),
            .nb_x0=sizeof(float), .nb_x1=n_embd*sizeof(float), .nb_x2=hc_row_bytes,
            .nb0=sizeof(float), .nb1=n_embd*sizeof(float), .nb_norm1=n_embd*sizeof(float),
            .eps=1.0e-6f, .norm_eps=1.0e-6f,
        };
        const uint64_t gate_row_bytes = (uint64_t)n_embd*sizeof(uint16_t);
        rust_star_q8_mv_args gate_mv = {
            .ne00=(int32_t)n_embd, .ne01=(int32_t)n_expert, .ne02=1,
            .nb00=sizeof(uint16_t), .nb01=gate_row_bytes,
            .nb02=gate_bytes, .nb03=gate_bytes,
            .ne10=(int32_t)n_embd, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=n_embd*sizeof(float),
            .nb12=n_embd*sizeof(float), .nb13=n_embd*sizeof(float),
            .ne0=(int32_t)n_expert, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        const uint64_t expert_row_bytes = (uint64_t)n_expert*sizeof(float);
        rust_star_unary_args unary_args = {
            .ne00=64, .ne01=1, .ne02=1, .ne03=1,
            .nb00=sizeof(float), .nb01=expert_row_bytes,
            .nb02=expert_row_bytes, .nb03=expert_row_bytes,
            .ne0=64, .ne1=1, .ne2=1, .ne3=1,
            .nb0=sizeof(float), .nb1=expert_row_bytes,
            .nb2=expert_row_bytes, .nb3=expert_row_bytes,
        };
        rust_star_router_select_one_args router_args = {
            .has_bias=bias_bytes ? 1u : 0u, .hash_mode=hash_bytes ? 1u : 0u,
            .use_token_buffer=0, .token=201, .hash_rows=hash_bytes ? 129280u : 0u,
        };

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail_with_message(error, error_bytes, @"failed to create FFN router command encoder");

        [encoder setComputePipelineState:context.rmsNormF32Pipeline];
        [encoder setBytes:&norm_args length:sizeof(norm_args) atIndex:0];
        [encoder setBuffer:after offset:0 atIndex:1];
        [encoder setBuffer:after offset:0 atIndex:2];
        [encoder setBuffer:after offset:0 atIndex:3];
        [encoder setBuffer:flat offset:0 atIndex:4];
        [encoder setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16ProjectionPipeline];
        [encoder setBytes:&hc_mv length:sizeof(hc_mv) atIndex:0];
        [encoder setBuffer:hc_fn offset:inner[0] atIndex:1];
        [encoder setBuffer:flat offset:0 atIndex:2];
        [encoder setBuffer:mix offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(12,1,1) threadsPerThreadgroup:MTLSizeMake(32,8,1)];

        [encoder setComputePipelineState:context.hcIngressPipeline];
        [encoder setBytes:&hc_args length:sizeof(hc_args) atIndex:0];
        [encoder setBuffer:mix offset:0 atIndex:1];
        [encoder setBuffer:hc_scale offset:inner[1] atIndex:2];
        [encoder setBuffer:hc_base offset:inner[2] atIndex:3];
        [encoder setBuffer:after offset:0 atIndex:4];
        [encoder setBuffer:split_buffer offset:0 atIndex:5];
        [encoder setBuffer:cur offset:0 atIndex:6];
        [encoder setBuffer:norm_weight offset:inner[3] atIndex:7];
        [encoder setBuffer:norm offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:(n_embd+4u+32u)*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];

        [encoder setComputePipelineState:context.f16ProjectionPipeline];
        [encoder setBytes:&gate_mv length:sizeof(gate_mv) atIndex:0];
        [encoder setBuffer:gate offset:inner[4] atIndex:1];
        [encoder setBuffer:norm offset:0 atIndex:2];
        [encoder setBuffer:logits_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:32u*2u*sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(128,1,1) threadsPerThreadgroup:MTLSizeMake(32,8,1)];

        [encoder setComputePipelineState:context.routerProbabilityPipeline];
        [encoder setBytes:&unary_args length:sizeof(unary_args) atIndex:0];
        [encoder setBuffer:logits_buffer offset:0 atIndex:1];
        [encoder setBuffer:probs_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];

        [encoder setComputePipelineState:context.routerFinalizePipeline];
        [encoder setBytes:&router_args length:sizeof(router_args) atIndex:0];
        [encoder setBuffer:probs_buffer offset:0 atIndex:1];
        int32_t zero = 0;
        float zero_f32 = 0.0f;
        if (bias) [encoder setBuffer:bias offset:inner[5] atIndex:2];
        else [encoder setBytes:&zero_f32 length:sizeof(zero_f32) atIndex:2];
        if (hash) [encoder setBuffer:hash offset:inner[6] atIndex:3];
        else [encoder setBytes:&zero length:sizeof(zero) atIndex:3];
        [encoder setBytes:&zero length:sizeof(zero) atIndex:4];
        [encoder setBuffer:selected_buffer offset:0 atIndex:5];
        [encoder setThreadgroupMemoryLength:256u*(sizeof(float)+sizeof(int32_t)) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [encoder setComputePipelineState:context.routerWeightsPipeline];
        [encoder setBuffer:probs_buffer offset:0 atIndex:0];
        [encoder setBuffer:selected_buffer offset:0 atIndex:1];
        [encoder setBuffer:weights_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(6,1,1)];
        [encoder endEncoding];

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        memcpy(mixes, mix.contents, mix_hc*sizeof(float));
        memcpy(split, split_buffer.contents, mix_hc*sizeof(float));
        memcpy(ffn_cur, cur.contents, n_embd*sizeof(float));
        memcpy(ffn_norm, norm.contents, n_embd*sizeof(float));
        memcpy(logits, logits_buffer.contents, n_expert*sizeof(float));
        memcpy(probs, probs_buffer.contents, n_expert*sizeof(float));
        memcpy(selected, selected_buffer.contents, n_used*sizeof(int32_t));
        memcpy(weights, weights_buffer.contents, n_used*sizeof(float));
        result->model_bytes = model_bytes;
        result->max_buffer_length = context.device.maxBufferLength;
        result->wrapped_model_ranges = 5u + (bias_bytes ? 1u : 0u) + (hash_bytes ? 1u : 0u);
        for (uint32_t i = 0; i < 7; i++) if (matches[i]) result->pointer_matches++;
        result->wall_ms = wall_end-wall_start;
        result->gpu_ms = gpu_elapsed_ms(command);
        return 1;
    }
}

int rust_star_metal_run_moe_output(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t routed_gate_offset,
    uint64_t routed_gate_bytes,
    uint64_t routed_up_offset,
    uint64_t routed_up_bytes,
    uint64_t routed_down_offset,
    uint64_t routed_down_bytes,
    uint64_t shared_gate_offset,
    uint64_t shared_gate_bytes,
    uint64_t shared_up_offset,
    uint64_t shared_up_bytes,
    uint64_t shared_down_offset,
    uint64_t shared_down_bytes,
    const float *ffn_norm,
    const int32_t *selected,
    const float *weights,
    const float *after_attention_hc,
    const float *split,
    float *routed_mid,
    float *routed_out,
    float *shared_out,
    float *after_ffn_hc,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes)
{
    const uint32_t n_embd = 4096, n_mid = 2048, n_expert = 256, n_used = 6, n_hc = 4;
    const uint64_t routed_gate_row = 1056, routed_down_row = 672;
    const uint64_t routed_gate_expert = n_mid*routed_gate_row;
    const uint64_t routed_down_expert = n_embd*routed_down_row;
    const uint64_t shared_gate_row = 4352, shared_down_row = 2176;
    if (!opaque_context || !model_mapping || !ffn_norm || !selected || !weights ||
        !after_attention_hc || !split || !routed_mid || !routed_out || !shared_out ||
        !after_ffn_hc || !result) {
        return fail_with_message(error, error_bytes, @"MoE output received invalid inputs");
    }
    if (routed_gate_bytes != n_expert*routed_gate_expert ||
        routed_up_bytes != n_expert*routed_gate_expert ||
        routed_down_bytes != n_expert*routed_down_expert ||
        shared_gate_bytes != n_mid*shared_gate_row ||
        shared_up_bytes != n_mid*shared_gate_row ||
        shared_down_bytes != n_embd*shared_down_row) {
        return fail_with_message(error, error_bytes, @"MoE output tensor shapes are invalid");
    }
    for (uint32_t i = 0; i < n_used; i++) {
        if (selected[i] < 0 || selected[i] >= (int32_t)n_expert || !isfinite(weights[i])) {
            return fail_with_message(error, error_bytes, @"MoE output router inputs are invalid");
        }
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!ensure_moe_output_pipelines(context, error, error_bytes)) return 0;
        memset(result, 0, sizeof(*result));

        NSUInteger inner[6] = {0};
        BOOL matches[6] = {NO};
        id<MTLBuffer> routed_gate = wrap_model_range(context, model_mapping, model_bytes,
            routed_gate_offset, routed_gate_bytes, &inner[0], &matches[0], error, error_bytes);
        id<MTLBuffer> routed_up = wrap_model_range(context, model_mapping, model_bytes,
            routed_up_offset, routed_up_bytes, &inner[1], &matches[1], error, error_bytes);
        id<MTLBuffer> routed_down = wrap_model_range(context, model_mapping, model_bytes,
            routed_down_offset, routed_down_bytes, &inner[2], &matches[2], error, error_bytes);
        id<MTLBuffer> shared_gate = wrap_model_range(context, model_mapping, model_bytes,
            shared_gate_offset, shared_gate_bytes, &inner[3], &matches[3], error, error_bytes);
        id<MTLBuffer> shared_up = wrap_model_range(context, model_mapping, model_bytes,
            shared_up_offset, shared_up_bytes, &inner[4], &matches[4], error, error_bytes);
        id<MTLBuffer> shared_down = wrap_model_range(context, model_mapping, model_bytes,
            shared_down_offset, shared_down_bytes, &inner[5], &matches[5], error, error_bytes);
        if (!routed_gate || !routed_up || !routed_down ||
            !shared_gate || !shared_up || !shared_down) return 0;

        const NSUInteger embd_bytes = n_embd*sizeof(float);
        const NSUInteger mid_bytes = n_mid*sizeof(float);
        const NSUInteger routed_mid_bytes = n_used*mid_bytes;
        const NSUInteger hc_bytes = n_hc*embd_bytes;
        id<MTLBuffer> norm = [context.device newBufferWithLength:embd_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> ids = [context.device newBufferWithLength:n_used*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> route_weights = [context.device newBufferWithLength:n_used*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> residual = [context.device newBufferWithLength:hc_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> split_buffer = [context.device newBufferWithLength:24u*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> routed_gate_out = [context.device newBufferWithLength:routed_mid_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> routed_up_out = [context.device newBufferWithLength:routed_mid_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> routed_mid_buffer = [context.device newBufferWithLength:routed_mid_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> routed_out_buffer = [context.device newBufferWithLength:embd_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> shared_gate_out = [context.device newBufferWithLength:mid_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> shared_up_out = [context.device newBufferWithLength:mid_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> shared_mid_buffer = [context.device newBufferWithLength:mid_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> shared_out_buffer = [context.device newBufferWithLength:embd_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> after_hc_buffer = [context.device newBufferWithLength:hc_bytes options:MTLResourceStorageModeShared];
        if (!norm || !ids || !route_weights || !residual || !split_buffer ||
            !routed_gate_out || !routed_up_out || !routed_mid_buffer || !routed_out_buffer ||
            !shared_gate_out || !shared_up_out || !shared_mid_buffer ||
            !shared_out_buffer || !after_hc_buffer) {
            return fail_with_message(error, error_bytes, @"failed to allocate MoE output buffers");
        }
        memcpy(norm.contents, ffn_norm, embd_bytes);
        memcpy(ids.contents, selected, n_used*sizeof(int32_t));
        memcpy(route_weights.contents, weights, n_used*sizeof(float));
        memcpy(residual.contents, after_attention_hc, hc_bytes);
        memcpy(split_buffer.contents, split, 24u*sizeof(float));

        rust_star_q8_mv_id_args gate_args = {
            .nei0=6, .nei1=1, .nbi1=6u*sizeof(int32_t),
            .ne00=4096, .ne01=2048, .ne02=256,
            .nb00=66, .nb01=routed_gate_row, .nb02=routed_gate_expert,
            .ne10=4096, .ne11=1, .ne12=1, .ne13=1,
            .nb10=sizeof(float), .nb11=embd_bytes, .nb12=embd_bytes,
            .ne0=2048, .ne1=6, .nb1=mid_bytes, .nr0=4,
            .tp_rank=0, .tp_world=0, .tp_addend=0, .tp_expert_base=0,
        };
        rust_star_moe_swiglu_weight_args activation = {
            .width=2048, .rows=6,
            .gate_row_stride=mid_bytes, .up_row_stride=mid_bytes,
            .mid_row_stride=mid_bytes, .weight_stride=sizeof(float),
            .write_clamped=0, .clamp_value=10.0f,
        };
        rust_star_q8_mv_id_args down_args = {
            .nei0=6, .nei1=1, .nbi1=6u*sizeof(int32_t),
            .ne00=2048, .ne01=4096, .ne02=256,
            .nb00=84, .nb01=routed_down_row, .nb02=routed_down_expert,
            .ne10=2048, .ne11=6, .ne12=1, .ne13=1,
            .nb10=sizeof(float), .nb11=mid_bytes, .nb12=routed_mid_bytes,
            .ne0=4096, .ne1=6, .nb1=embd_bytes, .nr0=4,
            .tp_rank=0, .tp_world=0, .tp_addend=0, .tp_expert_base=0,
        };
        rust_star_q8_mv_args shared_gate_args = {
            .ne00=4096, .ne01=2048, .ne02=1,
            .nb00=34, .nb01=shared_gate_row,
            .nb02=shared_gate_bytes, .nb03=shared_gate_bytes,
            .ne10=4096, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=embd_bytes, .nb12=embd_bytes, .nb13=embd_bytes,
            .ne0=2048, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_q8_mv_args shared_down_args = {
            .ne00=2048, .ne01=4096, .ne02=1,
            .nb00=34, .nb01=shared_down_row,
            .nb02=shared_down_bytes, .nb03=shared_down_bytes,
            .ne10=2048, .ne11=1, .ne12=1,
            .nb10=sizeof(float), .nb11=mid_bytes, .nb12=mid_bytes, .nb13=mid_bytes,
            .ne0=4096, .ne1=1, .nr0=2, .r2=1, .r3=1,
        };
        rust_star_hc_expand_args hc = {
            .n_embd=4096, .n_hc=4, .n_tokens=1,
            .nb_block0=sizeof(float), .nb_block1=embd_bytes,
            .nb_add0=sizeof(float), .nb_add1=embd_bytes,
            .nb_res0=sizeof(float), .nb_res1=embd_bytes, .nb_res2=hc_bytes,
            .nb_post0=sizeof(float), .nb_post1=24u*sizeof(float),
            .nb_comb0=sizeof(float), .nb_comb1=4u*sizeof(float), .nb_comb2=24u*sizeof(float),
            .nb0=sizeof(float), .nb1=embd_bytes, .nb2=hc_bytes, .has_add=1,
        };

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        if (!command) return fail_with_message(error, error_bytes, @"failed to create MoE output command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:context.routedPairSwigluPipeline];
        [encoder setBytes:&gate_args length:sizeof(gate_args) atIndex:0];
        [encoder setBytes:&activation length:sizeof(activation) atIndex:1];
        [encoder setBuffer:routed_gate offset:inner[0] atIndex:2];
        [encoder setBuffer:routed_up offset:inner[1] atIndex:3];
        [encoder setBuffer:norm offset:0 atIndex:4];
        [encoder setBuffer:routed_gate_out offset:0 atIndex:5];
        [encoder setBuffer:routed_up_out offset:0 atIndex:6];
        [encoder setBuffer:routed_mid_buffer offset:0 atIndex:7];
        [encoder setBuffer:ids offset:0 atIndex:8];
        [encoder setBuffer:route_weights offset:0 atIndex:9];
        [encoder setThreadgroupMemoryLength:2176u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(256, 1, 6)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:context.routedDownSumPipeline];
        [encoder setBytes:&down_args length:sizeof(down_args) atIndex:0];
        [encoder setBuffer:routed_down offset:inner[2] atIndex:1];
        [encoder setBuffer:routed_mid_buffer offset:0 atIndex:2];
        [encoder setBuffer:routed_out_buffer offset:0 atIndex:3];
        [encoder setBuffer:ids offset:0 atIndex:4];
        [encoder setBuffer:routed_out_buffer offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(512, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        [encoder endEncoding];

        const float clamp = 10.0f;
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:context.sharedGateUpPipeline];
        [encoder setBytes:&shared_gate_args length:sizeof(shared_gate_args) atIndex:0];
        [encoder setBuffer:shared_gate offset:inner[3] atIndex:1];
        [encoder setBuffer:shared_up offset:inner[4] atIndex:2];
        [encoder setBuffer:norm offset:0 atIndex:3];
        [encoder setBuffer:shared_gate_out offset:0 atIndex:4];
        [encoder setBuffer:shared_up_out offset:0 atIndex:5];
        [encoder setBuffer:shared_mid_buffer offset:0 atIndex:6];
        [encoder setBytes:&clamp length:sizeof(clamp) atIndex:7];
        [encoder setThreadgroupMemoryLength:512u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1024, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:context.sharedDownHcPipeline];
        [encoder setBytes:&shared_down_args length:sizeof(shared_down_args) atIndex:0];
        [encoder setBytes:&hc length:sizeof(hc) atIndex:1];
        [encoder setBuffer:shared_down offset:inner[5] atIndex:2];
        [encoder setBuffer:shared_mid_buffer offset:0 atIndex:3];
        [encoder setBuffer:shared_out_buffer offset:0 atIndex:4];
        [encoder setBuffer:routed_out_buffer offset:0 atIndex:5];
        [encoder setBuffer:residual offset:0 atIndex:6];
        [encoder setBuffer:split_buffer offset:4u*sizeof(float) atIndex:7];
        [encoder setBuffer:split_buffer offset:8u*sizeof(float) atIndex:8];
        [encoder setBuffer:after_hc_buffer offset:0 atIndex:9];
        [encoder setThreadgroupMemoryLength:256u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(2048, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        [encoder endEncoding];

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        const double wall_end = monotonic_ms();
        memcpy(routed_mid, routed_mid_buffer.contents, routed_mid_bytes);
        memcpy(routed_out, routed_out_buffer.contents, embd_bytes);
        memcpy(shared_out, shared_out_buffer.contents, embd_bytes);
        memcpy(after_ffn_hc, after_hc_buffer.contents, hc_bytes);
        result->model_bytes = model_bytes;
        result->max_buffer_length = context.device.maxBufferLength;
        result->wrapped_model_ranges = 6;
        for (uint32_t i = 0; i < 6; i++) if (matches[i]) result->pointer_matches++;
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
