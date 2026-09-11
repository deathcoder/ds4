#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(int argc, const char **argv) { @autoreleasepool {
    if (argc != 2) return 2;
    NSError *error = nil;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
        encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> library = source ? [device newLibraryWithSource:source
        options:[MTLCompileOptions new] error:&error] : nil;
    if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
    id<MTLFunction> function = [library newFunctionWithName:@"rust_star_routed_prefill_gate_up"];
    if (!function) return 3;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
    if (pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < 128) return 4;
    printf("{\"passed\":true,\"device\":\"%s\",\"kernel\":\"rust_star_routed_prefill_gate_up\","
           "\"thread_execution_width\":%lu,\"max_threads\":%lu,\"scope\":\"Metal library and pipeline compilation only\"}\n",
        device.name.UTF8String, pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup);
    return 0;
}}
