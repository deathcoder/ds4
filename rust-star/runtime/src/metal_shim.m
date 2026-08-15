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
@property(nonatomic, strong) id<MTLComputePipelineState> routerProbabilityPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerFinalizePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routerWeightsPipeline;
@property(nonatomic, strong) id<MTLLibrary> attentionOutputLibrary;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputLowPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputHcPipeline;
@property(nonatomic, strong) id<MTLLibrary> moeOutputLibrary;
@property(nonatomic, strong) id<MTLComputePipelineState> routedPairSwigluPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> routedDownSumPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sharedGateUpPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sharedDownHcPipeline;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLBuffer>> *modelViewCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLBuffer>> *activationBufferCache;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLCommandBuffer>> *chainedCommands;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSNumber *> *chainedWallStarts;
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
    id<MTLFunction> routerProbability = [library newFunctionWithName:@"kernel_dsv4_softplus_sqrt_f32_4"];
    id<MTLFunction> routerFinalize = [library newFunctionWithName:@"kernel_dsv4_router_finalize_one"];
    id<MTLFunction> routerWeights = [library newFunctionWithName:@"kernel_dsv4_router_weights_one"];
    int16_t simdgroups = 8;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&simdgroups type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> projection = [library newFunctionWithName:@"kernel_mul_mv_f16_f32_4"
                                               constantValues:constants
                                                        error:&compile_error];
    if (!repeat || !norm || !hc || !qkvNorm || !headNormRope || !ropeTail ||
        !kvStore || !cpyF32F16 || !flashPad || !flashVec || !flashReduce ||
        !projection || !routerProbability || !routerFinalize || !routerWeights) {
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
    context.routerProbabilityPipeline = [context.device newComputePipelineStateWithFunction:routerProbability error:&compile_error];
    context.routerFinalizePipeline = [context.device newComputePipelineStateWithFunction:routerFinalize error:&compile_error];
    context.routerWeightsPipeline = [context.device newComputePipelineStateWithFunction:routerWeights error:&compile_error];
    if (!context.repeatF32Pipeline || !context.rmsNormF32Pipeline ||
        !context.f16ProjectionPipeline || !context.hcIngressPipeline ||
        !context.qkvNormPipeline || !context.headNormRopePipeline ||
        !context.ropeTailPipeline || !context.kvStorePipeline ||
        !context.cpyF32F16Pipeline || !context.flashPadPipeline ||
        !context.flashVecPipeline || !context.flashReducePipeline ||
        !context.routerProbabilityPipeline || !context.routerFinalizePipeline ||
        !context.routerWeightsPipeline) {
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
    if (!routed_pair || !routed_down || !shared_gate_up || !shared_down_hc) {
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
    if (!context.routedPairSwigluPipeline || !context.routedDownSumPipeline ||
        !context.sharedGateUpPipeline || !context.sharedDownHcPipeline) {
        return fail_with_message(error, error_bytes, compile_error.localizedDescription);
    }
    context.moeOutputLibrary = library;
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

typedef struct rust_star_moe_swiglu_weight_args {
    uint32_t width, rows;
    uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride;
    uint32_t write_clamped;
    float clamp_value;
} rust_star_moe_swiglu_weight_args;

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
            fail_with_message(error, error_bytes, @"persistent Metal buffer size changed");
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
    const uint32_t visible_cache_rows = position + 1u;
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
    if (full_layer && (layer0->layer_index > 3 ||
        (layer0->layer_index == 0 && continuing_layer) ||
        (layer0->layer_index > 0 && !continuing_layer))) {
        return fail_with_message(error, error_bytes,
            @"full layer execution must run layer 0 before continuing into later layers");
    }
    if (full_layer && (position < 1u || position > 2u)) {
        return fail_with_message(error, error_bytes,
            @"the exact four-layer slice currently supports decode positions 1 and 2");
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
        (layer0->chain_final_layer < 2 || layer0->chain_final_layer > 3 ||
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
        BOOL matches[25] = {NO};
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
        id<MTLBuffer> cache_buffer = rope_and_store ? persistent_buffer(context, cache_key, 3u*kv_elements*sizeof(float), error, error_bytes) : nil;
        const NSUInteger staged_kv_bytes = 3u*kv_elements*sizeof(uint16_t);
        const NSUInteger mask_storage_bytes = 3u*sizeof(uint16_t);
        const NSUInteger mask_row_bytes = visible_cache_rows*sizeof(uint16_t);
        const NSUInteger flash_pad_bytes = 3u*32u*kv_elements*sizeof(uint16_t) + 32u*sizeof(uint16_t);
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
        if (full_layer && (!ffn_flat_buffer || !ffn_mix_buffer || !ffn_split_buffer ||
            !ffn_cur_buffer || !ffn_norm_buffer || !router_logits_buffer ||
            !router_probs_buffer || !selected_buffer || !route_weights_buffer ||
            !routed_gate_buffer || !routed_up_buffer || !routed_mid_buffer ||
            !routed_out_buffer || !shared_gate_buffer || !shared_up_buffer ||
            !shared_mid_buffer || !shared_out_buffer || !after_ffn_hc_buffer)) {
            return fail_with_message(error, error_bytes, @"failed to allocate full layer buffers");
        }
        if (rope_and_store && !chained_replay && position == 1u) {
            float *cache = cache_buffer.contents;
            for (uint32_t index = 0; index < 3u*kv_elements; index++) cache[index] = -12345.5f;
            if (attention_read) memcpy(cache, cache_row0, kv_elements*sizeof(float));
        }
        if (attention_read && !chained_replay) memset(mask_buffer.contents, 0, mask_storage_bytes);

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
            .head_dim=512, .n_rot=64, .raw_row=(int32_t)position,
        };
        const uint64_t attention_head_bytes = (uint64_t)kv_elements*sizeof(float);
        const uint64_t staged_row_bytes = (uint64_t)kv_elements*sizeof(uint16_t);
        rust_star_flash_pad_args flash_pad_args = {
            .ne11=(int32_t)visible_cache_rows, .ne_12_2=1, .ne_12_3=1,
            .nb11=staged_row_bytes, .nb12=(uint64_t)visible_cache_rows*staged_row_bytes,
            .nb13=(uint64_t)visible_cache_rows*staged_row_bytes,
            .nb21=staged_row_bytes, .nb22=(uint64_t)visible_cache_rows*staged_row_bytes,
            .nb23=(uint64_t)visible_cache_rows*staged_row_bytes,
            .ne31=1, .ne32=1, .ne33=1,
            .nb31=mask_row_bytes, .nb32=mask_row_bytes, .nb33=mask_row_bytes,
        };
        rust_star_flash_vec_args flash_vec_args = {
            .ne01=1, .ne02=64, .ne03=1,
            .nb01=64u*attention_head_bytes, .nb02=attention_head_bytes,
            .nb03=64u*attention_head_bytes,
            .ne11=(int32_t)visible_cache_rows, .ne_12_2=1, .ne_12_3=1, .ns10=512,
            .nb11=staged_row_bytes,
            .nb12=(uint64_t)visible_cache_rows*staged_row_bytes,
            .nb13=(uint64_t)visible_cache_rows*staged_row_bytes,
            .ns20=512,
            .nb21=staged_row_bytes,
            .nb22=(uint64_t)visible_cache_rows*staged_row_bytes,
            .nb23=(uint64_t)visible_cache_rows*staged_row_bytes,
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
            for (uint32_t index = 0; index < 3u*kv_elements; index++) cache[index] = -12345.5f;
            if (attention_read) memcpy(cache, cache_row0, kv_elements*sizeof(float));
        }
        const double wall_start = monotonic_ms();
        command = [context.queue commandBuffer];
        if (!command) return fail_with_message(error, error_bytes, @"failed to create attention ingress command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return fail_with_message(error, error_bytes, @"failed to create attention ingress encoder");

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

            if (attention_read) {
                uint32_t staged_elements = visible_cache_rows*kv_elements;
                [encoder setComputePipelineState:context.cpyF32F16Pipeline];
                [encoder setBytes:&staged_elements length:sizeof(staged_elements) atIndex:0];
                [encoder setBuffer:cache_buffer offset:0 atIndex:1];
                [encoder setBuffer:staged_kv_buffer offset:0 atIndex:2];
                const NSUInteger staged_groups = (staged_elements + 1023u) / 1024u;
                [encoder dispatchThreadgroups:MTLSizeMake(staged_groups,1,1)
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
            result->wrapped_model_ranges = 25;
            for (uint32_t index = 0; index < 25; index++) {
                if (matches[index]) result->pointer_matches++;
            }
            return 1;
        }
        if (chained_timing) {
            result->model_bytes = model_bytes;
            result->max_buffer_length = context.device.maxBufferLength;
            result->wrapped_model_ranges = 25;
            for (uint32_t index = 0; index < 25; index++) {
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
        result->wrapped_model_ranges = full_layer ? 25 :
            (attention_output ? 13 : (attention_read ? 11 : (extended ? 10 : 6)));
        for (uint32_t index = 0; index < result->wrapped_model_ranges; index++) if (matches[index]) result->pointer_matches++;
        result->wall_ms = measured_wall_ms/measured_iterations;
        result->gpu_ms = measured_gpu_ms/measured_iterations;
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
