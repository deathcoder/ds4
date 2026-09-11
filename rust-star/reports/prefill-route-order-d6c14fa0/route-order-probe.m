#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <math.h>
#include <sys/resource.h>

// Host layouts match the production shim and imported Metal argument structs.
typedef struct {
    int32_t ne02, ne10, ne11;
    uint64_t nb11, nb12;
    int32_t ne21, ne20;
    uint64_t nb21;
} MapArgs;
typedef struct {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne11;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne20, ne21, ne0, ne1;
    int16_t r2, r3;
    int32_t tp_rank, tp_world, tp_expert_base;
} MatArgs;
typedef struct {
    uint32_t width, rows;
    uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride;
    uint32_t write_clamped;
    float clamp_value;
} ActArgs;
typedef struct { __fp16 d; uint16_t qs[32]; } IQ2Block;
_Static_assert(sizeof(MapArgs) == 48, "map ABI");
_Static_assert(sizeof(MatArgs) == 104, "matmul ABI");
_Static_assert(sizeof(ActArgs) == 48, "activation ABI");
_Static_assert(sizeof(IQ2Block) == 66, "IQ2_XXS block ABI");
enum { Width = 4096, Hidden = 2048, Experts = 256, Routes = 6, Guard = 4096 };

static void require(BOOL good, NSString *message) {
    if (!good) { fprintf(stderr, "%s\n", message.UTF8String); exit(1); }
}
static void emit(NSDictionary *row) {
    NSError *error = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:row options:NSJSONWritingSortedKeys error:&error];
    require(json != nil, error.description);
    fwrite(json.bytes, 1, json.length, stdout); putchar('\n'); fflush(stdout);
}
static uint32_t random_word(uint32_t *state) {
    uint32_t x = *state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return *state = x;
}
static id<MTLBuffer> buffer(id<MTLDevice> device, NSUInteger bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(b != nil, [NSString stringWithFormat:@"Cannot allocate %lu bytes", bytes]);
    return b;
}
static id<MTLLibrary> library(id<MTLDevice> device, const char *path) {
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path]
        encoding:NSUTF8StringEncoding error:&error];
    require(source != nil, error.description);
    // Match the runtime's default MTLCompileOptions; no alternate math mode.
    id<MTLLibrary> lib = [device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
    require(lib != nil, error.description); return lib;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> lib, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    require(fn != nil, [@"Missing function: " stringByAppendingString:name]);
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&error];
    require(p != nil, error.description); return p;
}
static double finish(id<MTLCommandBuffer> command) {
    [command commit]; [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, command.error.description);
    double ms = (command.GPUEndTime - command.GPUStartTime)*1000.0;
    require(isfinite(ms) && ms > 0, @"Invalid GPU command interval"); return ms;
}
static void fill_weights(id<MTLBuffer> b, uint32_t seed) {
    IQ2Block *blocks = b.contents;
    for (NSUInteger i = 0; i < b.length/sizeof(IQ2Block); i++) {
        blocks[i].d = (__fp16)((1 + random_word(&seed)%16)/8192.0f);
        for (int j = 0; j < 32; j += 2) {
            uint32_t v = random_word(&seed);
            blocks[i].qs[j] = (uint16_t)v; blocks[i].qs[j+1] = (uint16_t)(v >> 16);
        }
    }
}
static NSString *digest(id<MTLBuffer> b, NSUInteger bytes) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(b.contents, (CC_LONG)bytes, hash);
    NSMutableString *s = [NSMutableString string];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [s appendFormat:@"%02x", hash[i]];
    return s;
}
static void guards(id<MTLBuffer> b, NSUInteger bytes) {
    unsigned char *data = b.contents;
    for (NSUInteger i = bytes; i < b.length; i++) require(data[i] == 0xa5, @"Output guard overwritten");
}
static double run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> p, BOOL transposed,
                  MatArgs *args, ActArgs *act, NSArray<id<MTLBuffer>> *inputs,
                  id<MTLBuffer> output, NSUInteger tpeBytes, NSUInteger workOffset, NSUInteger workCap) {
    // Sentinel reset and comparisons are deliberately outside the GPU timing.
    memset(output.contents, 0xa5, output.length);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBytes:args length:sizeof(*args) atIndex:0];
    [e setBytes:act length:sizeof(*act) atIndex:1];
    [e setBuffer:inputs[0] offset:0 atIndex:2];
    [e setBuffer:inputs[1] offset:0 atIndex:3];
    [e setBuffer:inputs[2] offset:0 atIndex:4];
    [e setBuffer:inputs[3] offset:0 atIndex:5];
    [e setBuffer:inputs[3] offset:tpeBytes atIndex:6];
    [e setBuffer:output offset:0 atIndex:7];
    [e setBuffer:inputs[4] offset:0 atIndex:8];
    [e setBuffer:inputs[3] offset:workOffset atIndex:9];
    [e setThreadgroupMemoryLength:16384 atIndex:0];
    MTLSize grid = transposed ? MTLSizeMake(Hidden/64, workCap, 1) : MTLSizeMake(workCap, Hidden/64, 1);
    [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [e endEncoding]; return finish(cb);
}
static void check_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> mapper, NSArray<id<MTLComputePipelineState>> *variants,
                       id<MTLBuffer> gate, id<MTLBuffer> up, uint32_t rows, BOOL skewed, BOOL timed) {
    NSUInteger tpeBytes = Experts*sizeof(uint32_t), hidsBytes = Experts*rows*sizeof(int32_t);
    NSUInteger workOffset = (tpeBytes+hidsBytes+7)&~7ull;
    NSUInteger workCap = ((NSUInteger)rows*Routes + 31*Experts + 31)/32;
    NSUInteger outputBytes = (NSUInteger)rows*Routes*Hidden*sizeof(uint16_t);
    id<MTLBuffer> x = buffer(device, (NSUInteger)rows*Width*sizeof(float));
    id<MTLBuffer> ids = buffer(device, rows*Routes*sizeof(int32_t));
    id<MTLBuffer> weights = buffer(device, rows*Routes*sizeof(float));
    id<MTLBuffer> map = buffer(device, workOffset+8+workCap*8);
    id<MTLBuffer> expected = buffer(device, outputBytes+Guard), actual = buffer(device, outputBytes+Guard);
    memset(map.contents, 0, map.length);
    uint32_t seed = 0x67c10421 ^ rows ^ (skewed ? 0x3ca1085f : 0);
    float *values = x.contents, *routeWeights = weights.contents;
    int32_t *selected = ids.contents;
    for (NSUInteger i = 0; i < (NSUInteger)rows*Width; i++)
        values[i] = ((int32_t)(random_word(&seed)%20001)-10000)/8192.0f;
    uint32_t counts[Experts] = {0};
    for (uint32_t token = 0; token < rows; token++) {
        for (uint32_t slot = 0; slot < Routes; slot++) {
            uint32_t expert = skewed && token%4 != 0 ? (token*7+slot*3)%16 : (token*17+slot*43)%Experts;
            selected[token*Routes+slot] = expert; counts[expert]++;
            routeWeights[token*Routes+slot] = (1+random_word(&seed)%1024)/4096.0f;
        }
    }
    MapArgs ma = {.ne02=Experts,.ne10=Width,.ne11=1,.nb11=Width*4,.nb12=Width*4,
        .ne21=rows,.ne20=Routes,.nb21=Routes*4};
    id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:mapper]; [e setBytes:&ma length:sizeof(ma) atIndex:0];
    [e setBuffer:ids offset:0 atIndex:1]; [e setBuffer:map offset:0 atIndex:2];
    [e setBuffer:map offset:tpeBytes atIndex:3]; [e setBuffer:map offset:workOffset atIndex:4];
    [e setThreadgroupMemoryLength:Experts*Routes*sizeof(uint16_t) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(Experts,1,1)];
    [e endEncoding]; finish(cb);
    // Check the actual GPU work map against independent CPU enumeration.
    uint32_t *tpe = map.contents, *work = (uint32_t *)((char *)map.contents+workOffset), nwork = 0;
    int32_t *mapped = (int32_t *)((char *)map.contents+tpeBytes);
    uint32_t empty = 0, partial = 0;
    for (uint32_t expert = 0; expert < Experts; expert++) {
        require(tpe[expert] == counts[expert], @"Expert count mismatch");
        empty += counts[expert] == 0; partial += counts[expert]%32 != 0;
        uint32_t at = 0;
        for (uint32_t id = 0; id < rows*Routes; id++) if (selected[id] == (int32_t)expert)
            require(mapped[expert*rows+at++] == (int32_t)id, @"Expert route map mismatch");
        for (uint32_t start = 0; start < counts[expert]; start += 32) {
            require(work[2+2*nwork] == expert && work[3+2*nwork] == start, @"Work tile mismatch"); nwork++;
        }
    }
    require(work[0] == nwork && nwork <= workCap, @"Work count mismatch");
    MatArgs args = {.ne00=Width,.ne02=Experts,.nb01=1056,.nb02=Hidden*1056ull,.nb03=Experts*Hidden*1056ull,
        .ne11=1,.nb10=4,.nb11=Width*4,.nb12=Width*4,.nb13=(uint64_t)rows*Width*4,
        .ne20=Routes,.ne21=rows,.ne0=Hidden,.ne1=Routes,.r2=1,.r3=1};
    ActArgs act = {.width=Hidden,.rows=rows*Routes,.gate_row_stride=Hidden*4,.up_row_stride=Hidden*4,
        .mid_row_stride=Hidden*2,.weight_stride=4,.clamp_value=10};
    NSArray *inputs = @[gate, up, x, map, weights];
    double initial = run(queue, variants[0], NO, &args, &act, inputs, expected, tpeBytes, workOffset, workCap);
    guards(expected, outputBytes);
    uint16_t *bits = expected.contents; NSUInteger nonzero = 0, negative = 0;
    for (NSUInteger i = 0; i < outputBytes/2; i++) {
        require((bits[i]&0x7c00) != 0x7c00, @"Nonfinite baseline output");
        nonzero += (bits[i]&0x7fff) != 0; negative += (bits[i]&0x8000) != 0;
    }
    require(nonzero > outputBytes/20 && negative > 0 && negative < outputBytes/2, @"Degenerate synthetic outputs");
    emit(@{@"type":@"case",@"rows":@(rows),@"routing":skewed?@"skewed":@"balanced",@"timed":@(timed),
           @"words":@(outputBytes/2),@"sha256":digest(expected,outputBytes),@"active_tiles":@(nwork),
           @"work_capacity":@(workCap),@"empty_experts":@(empty),@"partial_expert_tiles":@(partial),
           @"nonzero_words":@(nonzero),@"negative_words":@(negative),@"initial_baseline_gpu_ms":@(initial)});
    int rounds = timed ? 9 : 1;
    for (int round = 0; round < rounds; round++) {
        int first = round%2;
        for (int turn = 0; turn < 2; turn++) { @autoreleasepool {
            int variant = first ^ turn;
            double ms = run(queue,variants[variant],variant == 1,&args,&act,inputs,actual,tpeBytes,workOffset,workCap);
            guards(actual, outputBytes);
            require(memcmp(actual.contents,expected.contents,outputBytes) == 0, @"Output bit mismatch");
            emit(@{@"type":@"sample",@"rows":@(rows),@"routing":skewed?@"skewed":@"balanced",
                   @"phase":timed?(round<2?@"warmup":@"timed"):@"control",@"round":@(timed?round-2:round),
                   @"order":first==0?@"AB":@"BA",@"variant":variant==0?@"A":@"B",@"gpu_ms":@(ms),
                   @"words_checked":@(outputBytes/2),@"exact":@YES,@"guards_intact":@YES});
        }}
    }
}
int main(int argc, const char **argv) { @autoreleasepool {
    require(argc == 3, @"usage: route-order-probe baseline.metal candidate.metal");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil && [device.name containsString:@"M1 Ultra"], @"This experiment targets M1 Ultra");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    emit(@{@"type":@"start",@"device":device.name,@"scope":@"Synthetic routed-prefill kernel comparison; not full-model C0 or engine throughput",
           @"os":[NSProcessInfo processInfo].operatingSystemVersionString,@"math_options":@"MTLCompileOptions defaults, matching native runtime"});
    id<MTLLibrary> base = library(device, argv[1]), candidate = library(device, argv[2]);
    NSString *name = @"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16";
    NSArray *variants = @[pipeline(device,base,name),pipeline(device,candidate,name)];
    id<MTLComputePipelineState> mapper = pipeline(device,base,@"kernel_mul_mm_id_map0_ne20_6");
    NSUInteger weightBytes = Experts*Hidden*1056ull;
    id<MTLBuffer> gate = buffer(device,weightBytes), up = buffer(device,weightBytes);
    fill_weights(gate,0x8dc23121); fill_weights(up,0xb524ffe3);
    emit(@{@"type":@"weights",@"bytes_each":@(weightBytes),@"gate_sha256":digest(gate,weightBytes),@"up_sha256":digest(up,weightBytes)});
    const uint32_t controls[] = {1,31,32,33,127};
    for (unsigned i = 0; i < sizeof(controls)/sizeof(*controls); i++) { @autoreleasepool {
        check_case(device,queue,mapper,variants,gate,up,controls[i],NO,NO);
    }}
    check_case(device,queue,mapper,variants,gate,up,2048,NO,YES);
    check_case(device,queue,mapper,variants,gate,up,4096,NO,YES);
    check_case(device,queue,mapper,variants,gate,up,4096,YES,YES);
    struct rusage usage; getrusage(RUSAGE_SELF,&usage);
    emit(@{@"type":@"complete",@"passed":@YES,@"max_rss_bytes":@(usage.ru_maxrss)});
    return 0;
}}
