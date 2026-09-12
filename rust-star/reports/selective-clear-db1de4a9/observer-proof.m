#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include "observer.inc"
int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 2;
        __weak id<MTLBuffer> weak;
        __attribute__((objc_precise_lifetime)) id<MTLBuffer> persistent =
            [device newBufferWithLength:8192 options:MTLResourceStorageModeShared];
        if (!persistent) return 3;
        rs_lifetime_register(persistent,@"persistent",YES,0);
        @autoreleasepool {
            __attribute__((objc_precise_lifetime)) id<MTLBuffer> scratch =
                [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
            if (!scratch) return 4;
            weak = scratch;
            rs_lifetime_register(scratch,@"scratch",NO,0);
            rs_lifetime_register(scratch,@"scratch",NO,0);
            rs_lifetime_snapshot(device,nil,@"proof_alive",32,0,8192);
            if (rs_lifetime_registered_count != 2 || !weak) return 5;
        }
        if (weak != nil) return 6;
        rs_lifetime_snapshot(device,nil,@"proof_scratch_dead",32,32,8192);
        if (rs_lifetime_records.count != 1 || rs_lifetime_dead_count != 1) return 7;
        persistent = nil;
        rs_lifetime_snapshot(device,nil,@"proof_all_dead",32,64,8192);
        if (rs_lifetime_records.count != 0 || rs_lifetime_dead_count != 2) return 8;
        puts("observer weak ownership and deduplication PASS");
    }
    return 0;
}
