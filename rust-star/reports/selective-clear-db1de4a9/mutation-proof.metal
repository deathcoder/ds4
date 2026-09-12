#include <metal_stdlib>
using namespace metal;
struct Args { ulong bytes; ulong offset; uint seed; uint kind; };
inline uint value(ulong i,uint seed,uint kind) {
    uint v=((uint)i*1664525u + seed*1013904223u) ^ ((uint)i>>7u);
    return kind==1u ? (seed & 255u)*0x01010101u : (kind==2u ? v^0x9e3779b9u : v);
}
kernel void write_pattern(device uchar *data [[buffer(0)]],
    constant Args &a [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    ulong base=(ulong)gid*16ul;
    if(base>=a.bytes)return;
    if(base+16ul<=a.bytes) {
        ulong i=base/4ul;
        *((device uint4 *)(data+a.offset+base))=uint4(value(i,a.seed,a.kind),
            value(i+1,a.seed,a.kind),value(i+2,a.seed,a.kind),value(i+3,a.seed,a.kind));
    } else {
        for(ulong j=base;j<a.bytes;j++)
            data[a.offset+j]=(uchar)(value(j/4ul,a.seed,a.kind)>>((j%4ul)*8ul));
    }
}
kernel void verify_pattern(device const uchar *data [[buffer(0)]],
    constant Args &a [[buffer(1)]], device atomic_uint *errors [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    ulong base=(ulong)gid*16ul;
    if(base>=a.bytes)return;
    uint bad=0;
    if(base+16ul<=a.bytes) {
        ulong i=base/4ul;uint4 got=*((device const uint4 *)(data+a.offset+base));
        uint4 expected=uint4(value(i,a.seed,a.kind),value(i+1,a.seed,a.kind),
            value(i+2,a.seed,a.kind),value(i+3,a.seed,a.kind));
        bad=(got.x!=expected.x)+(got.y!=expected.y)+(got.z!=expected.z)+(got.w!=expected.w);
    } else {
        for(ulong j=base;j<a.bytes;j++)
            bad+=(data[a.offset+j]!=(uchar)(value(j/4ul,a.seed,a.kind)>>((j%4ul)*8ul)));
    }
    if(bad)atomic_fetch_add_explicit(errors,bad,memory_order_relaxed);
}

kernel void mutate_existing(device uchar *data [[buffer(0)]],
    constant Args &a [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
    ulong base=(ulong)gid*16ul;
    if(base>=a.bytes)return;
    if(base+16ul<=a.bytes) {
        device uint4 *p=(device uint4 *)(data+a.offset+base);
        *p = *p ^ uint4(0x9e3779b9u);
    } else {
        for(ulong j=base;j<a.bytes;j++)data[a.offset+j]^=(uchar)(0x9e3779b9u>>((j%4ul)*8ul));
    }
}
