#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(int argc, const char **argv) { @autoreleasepool {
    if (argc != 2) return 2;
    NSError *error = nil;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> library = source ? [device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error] : nil;
    if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
    bool enabled = true, disabled = false;
    int32_t head_dim = 512, nsg = 8;
    for (unsigned padded=0; padded<2; padded++) {
        bool has_pad = padded != 0;
        MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
        [constants setConstantValue:&enabled type:MTLDataTypeBool atIndex:300];
        [constants setConstantValue:&enabled type:MTLDataTypeBool atIndex:301];
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:302];
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:303];
        [constants setConstantValue:&has_pad type:MTLDataTypeBool atIndex:304];
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:310];
        [constants setConstantValue:&head_dim type:MTLDataTypeInt atIndex:320];
        [constants setConstantValue:&head_dim type:MTLDataTypeInt atIndex:321];
        [constants setConstantValue:&nsg type:MTLDataTypeInt atIndex:322];
        id<MTLFunction> fn = [library newFunctionWithName:@"rust_star_compressed_prefill" constantValues:constants error:&error];
        id<MTLComputePipelineState> p = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p) { fprintf(stderr,"%s\n",error.description.UTF8String);return 3; }
        if (p.threadExecutionWidth != 32 || p.maxTotalThreadsPerThreadgroup < 256) return 4;
    }
    printf("{\"passed\":true,\"device\":\"%s\",\"kernel\":\"rust_star_compressed_prefill\",\"padded_and_unpadded\":true,\"nsg\":8,\"head_dim\":512,\"scope\":\"Production-assembled Metal library and both pipeline compilation checks\"}\n",device.name.UTF8String);
    return 0;
}}
