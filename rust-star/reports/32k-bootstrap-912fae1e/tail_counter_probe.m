/* Independent Metal counter diagnostic; no model or inference code is loaded.
 * Build: clang -fobjc-arc -O2 -framework Foundation -framework Metal FILE -o probe
 * Run: probe output.json
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2 || [[NSFileManager defaultManager] fileExistsAtPath:[NSString stringWithUTF8String:argv[1]]]) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return 2;
        id<MTLCounterSet> set = nil;
        for (id<MTLCounterSet> item in device.counterSets)
            if ([item.name isEqualToString:MTLCommonCounterSetTimestamp]) set = item;
        if (!set) return 2;
        NSError *error = nil;
        NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void step(device uint *a [[buffer(0)]], uint i [[thread_position_in_grid]]) {"
            "a[i] = a[i] * 1664525u + 1013904223u;}";
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"step"];
        id<MTLComputePipelineState> pipeline = function
            ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 2; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        MTLCounterSampleBufferDescriptor *descriptor = [MTLCounterSampleBufferDescriptor new];
        descriptor.counterSet = set; descriptor.storageMode = MTLStorageModeShared;
        descriptor.sampleCount = 18;
        id<MTLCounterSampleBuffer> samples = [device newCounterSampleBufferWithDescriptor:descriptor error:&error];
        if (!samples || !queue) return 2;
        NSMutableArray *records = [NSMutableArray array];
        NSUInteger error_values = 0, zero_spans = 0, backward_spans = 0;
        NSUInteger touching_boundaries = 0, overlapping_boundaries = 0, wrong_words = 0;
        for (NSNumber *width in @[@1, @256, @4096, @65536]) {
            NSUInteger count = width.unsignedIntegerValue;
            id<MTLBuffer> data = [device newBufferWithLength:count*sizeof(uint32_t) options:MTLResourceStorageModeShared];
            if (!data) return 2;
            memset(data.contents, 0, data.length);
            uint32_t expected = 0;
            for (NSUInteger repetition = 0; repetition < 64; repetition++) {
                @autoreleasepool {
                    id<MTLCommandBuffer> command = [queue commandBuffer];
                    for (NSUInteger stage = 0; stage < 9; stage++) {
                        MTLComputePassDescriptor *pass = [MTLComputePassDescriptor computePassDescriptor];
                        pass.sampleBufferAttachments[0].sampleBuffer = samples;
                        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2*stage;
                        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2*stage+1;
                        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDescriptor:pass];
                        if (!encoder) return 2;
                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:data offset:0 atIndex:0];
                        [encoder dispatchThreads:MTLSizeMake(count,1,1)
                            threadsPerThreadgroup:MTLSizeMake(MIN(count,MIN((NSUInteger)256,pipeline.maxTotalThreadsPerThreadgroup)),1,1)];
                        [encoder endEncoding];
                        expected = expected*1664525u+1013904223u;
                    }
                    [command commit]; [command waitUntilCompleted];
                    if (command.status != MTLCommandBufferStatusCompleted) return 2;
                    NSData *resolved = [samples resolveCounterRange:NSMakeRange(0,18)];
                    if (resolved.length != 18*sizeof(MTLCounterResultTimestamp)) return 2;
                    const MTLCounterResultTimestamp *values = resolved.bytes;
                    NSMutableArray *ticks = [NSMutableArray array];
                    for (NSUInteger i = 0; i < 18; i++) {
                        uint64_t value = values[i].timestamp;
                        [ticks addObject:@(value)];
                        if (value == MTLCounterErrorValue) error_values++;
                        else if (i && values[i-1].timestamp != MTLCounterErrorValue) {
                            uint64_t previous = values[i-1].timestamp;
                            if (i%2) { zero_spans += value==previous; backward_spans += value<previous; }
                            else { touching_boundaries += value==previous; overlapping_boundaries += value<previous; }
                        }
                    }
                    const uint32_t *actual = data.contents;
                    for (NSUInteger i = 0; i < count; i++) wrong_words += actual[i]!=expected;
                    [records addObject:@{@"elements":width,@"repetition":@(repetition),@"ticks":ticks,
                        @"command_gpu_ms":@((command.GPUEndTime-command.GPUStartTime)*1000.0)}];
                }
            }
        }
        NSDictionary *result = @{@"schema":@"rust-star-tail-counter-probe-v1",@"device":device.name,
            @"dependent_encoders_per_command":@9,@"sample_buffer_reused":@YES,
            @"error_values":@(error_values),@"zero_duration_spans":@(zero_spans),
            @"backward_spans":@(backward_spans),@"touching_boundaries":@(touching_boundaries),
            @"overlapping_boundaries":@(overlapping_boundaries),@"wrong_gpu_output_words":@(wrong_words),
            @"records":records};
        NSData *json = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error];
        if (!json || ![json writeToFile:[NSString stringWithUTF8String:argv[1]] options:NSDataWritingAtomic error:&error]) return 2;
        printf("commands=%lu error=%lu zero=%lu backward=%lu touching=%lu overlap=%lu wrong=%lu\n",
            records.count,error_values,zero_spans,backward_spans,touching_boundaries,overlapping_boundaries,wrong_words);
        return wrong_words ? 1 : 0;
    }
}
