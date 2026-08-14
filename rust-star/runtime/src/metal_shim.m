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

@interface RustStarMetalContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> probePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> getRowsF16Pipeline;
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

void rust_star_metal_destroy(void *opaque_context) {
    if (!opaque_context) return;
    @autoreleasepool {
        RustStarMetalContext *context = CFBridgingRelease(opaque_context);
        (void)context;
    }
}
