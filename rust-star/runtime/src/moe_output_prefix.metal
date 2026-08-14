#include <metal_stdlib>
using namespace metal;

// Exact standalone prefix used by DwarfStar before concatenating its Metal
// source files. Only the diagnostic page-touch kernel is omitted.
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
