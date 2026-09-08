#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// A sizing/allocation preflight, not model execution or a full working-set proof.
int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2 || [[NSFileManager defaultManager] fileExistsAtPath:@(argv[1])]) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!device || !queue) return 3;
        const NSUInteger context = 131072u, chunk = 4096u;
        NSMutableArray *specs = [NSMutableArray array];
        for (NSUInteger layer = 0; layer < 43u; layer++) {
            [specs addObject:@{@"role": [NSString stringWithFormat:@"layer%lu.FullKv", layer], @"bytes": @(context*512u*4u)}];
            if (layer < 2u) continue; // Bootstrap layers 0/1 retain only FullKv.
            NSUInteger ratio = layer%2u ? 128u : 4u;
            [specs addObject:@{@"role": [NSString stringWithFormat:@"layer%lu.AttnCompressed", layer], @"bytes": @((context/ratio+1u)*512u*4u)}];
            if (layer%2u == 0u) [specs addObject:@{@"role": [NSString stringWithFormat:@"layer%lu.IndexerCompressed", layer], @"bytes": @((context/4u+1u)*128u*4u)}];
        }
        // Representative simultaneous transient roles, deliberately explicit.
        // Other runtime scratch/model/driver allocations are outside this probe.
        NSDictionary *widths = @{@"layer1.InputHc": @16384, @"layer2.InputHc": @16384,
            @"layer2.FullQNorm": @1024, @"layer2.FullAttnNorm": @4096, @"layer2.AttnSplit": @24,
            @"layer2.ContinuationQ": @32768, @"odd.FullQ": @32768, @"odd.AttnSplit": @24,
            @"even.FullQ": @32768, @"even.FullQNorm": @1024, @"even.FullNorm": @4096,
            @"even.InputHc": @16384, @"even.AttnSplit": @24};
        for (NSString *role in [[widths allKeys] sortedArrayUsingSelector:@selector(compare:)])
            [specs addObject:@{@"role": role, @"bytes": @(chunk*[widths[role] unsignedIntegerValue]*4u)}];
        NSMutableArray *buffers = [NSMutableArray array];
        NSUInteger total = 0, largest = 0;
        for (NSDictionary *spec in specs) {
            NSUInteger bytes = [spec[@"bytes"] unsignedIntegerValue];
            if (bytes > device.maxBufferLength) return 4;
            id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!buffer) return 5;
            buffer.label = spec[@"role"]; [buffers addObject:buffer];
            total += bytes; largest = MAX(largest, bytes);
        }
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (!command || !blit) return 6;
        for (id<MTLBuffer> buffer in buffers) [blit fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:0xA5];
        [blit endEncoding]; [command commit]; [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return 7;
        NSUInteger checked = 0;
        for (id<MTLBuffer> buffer in buffers) {
            const uint8_t *bytes = buffer.contents;
            // Read every VM page plus the final byte; GPU filled the entire buffer.
            for (NSUInteger offset = 0; offset < buffer.length; offset += 16384u) {
                if (bytes[offset] != 0xA5) return 8;
                checked++;
            }
            if (bytes[buffer.length-1u] != 0xA5) return 9;
            checked++;
        }
        NSDictionary *result = @{@"schema": @"rust-star-128k-allocation-preflight-v1", @"passed": @YES,
            @"scope": @"Simultaneous modeled history and selected chunk transients; excludes model, other scratch and inference correctness",
            @"device": device.name, @"context_tokens": @(context), @"chunk_rows": @(chunk),
            @"physical_memory_bytes": @([NSProcessInfo processInfo].physicalMemory),
            @"max_buffer_length": @(device.maxBufferLength),
            @"recommended_max_working_set_size": @(device.recommendedMaxWorkingSetSize),
            @"current_allocated_size": @(device.currentAllocatedSize),
            @"total_requested_bytes": @(total), @"largest_requested_bytes": @(largest),
            @"checked_page_and_end_bytes": @(checked), @"buffers": specs};
        NSError *error = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
        if (!data || ![data writeToFile:@(argv[1]) options:NSDataWritingAtomic error:&error]) return 10;
        puts("128K modeled allocation/fill preflight passed; not an inference or residency proof");
    }
    return 0;
}
