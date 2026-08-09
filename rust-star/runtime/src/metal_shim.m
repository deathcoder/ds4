#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_shim.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

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

@interface RustStarMetalContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> probePipeline;
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

void rust_star_metal_destroy(void *opaque_context) {
    if (!opaque_context) return;
    @autoreleasepool {
        RustStarMetalContext *context = CFBridgingRelease(opaque_context);
        (void)context;
    }
}
