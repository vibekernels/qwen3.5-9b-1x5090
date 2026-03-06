#include "../utils.h"
#include <cuda_bf16.h>
#include <cstdint>

// SwiGLU activation: SiLU(gate) * up
// SiLU(x) = x * sigmoid(x)
// gate and up are both [n_tokens, n_ff], output is [n_tokens, n_ff]

__global__ void swiglu_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int n_elements   // total elements = n_tokens * n_ff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);

    // SiLU(g) * u
    float silu_g = g / (1.0f + expf(-g));
    output[idx] = __float2bfloat16(silu_g * u);
}

void launch_swiglu(
    __nv_bfloat16* output,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int n_tokens,
    int n_ff,
    cudaStream_t stream
) {
    int n = n_tokens * n_ff;
    int threads = 256;
    int blocks = cdiv(n, threads);
    swiglu_kernel<<<blocks, threads, 0, stream>>>(output, gate, up, n);
}

// SwiGLU from packed input: packed = [n_tokens, 2*n_ff] where each row is [gate, up]
// output: [n_tokens, n_ff]
__global__ void swiglu_packed_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ packed,
    int n_ff,
    int n_elements   // n_tokens * n_ff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int token = idx / n_ff;
    int col = idx % n_ff;

    float g = __bfloat162float(packed[token * 2 * n_ff + col]);
    float u = __bfloat162float(packed[token * 2 * n_ff + n_ff + col]);

    float silu_g = g / (1.0f + expf(-g));
    output[idx] = __float2bfloat16(silu_g * u);
}

void launch_swiglu_packed(
    __nv_bfloat16* output,
    const __nv_bfloat16* packed,
    int n_tokens,
    int n_ff,
    cudaStream_t stream
) {
    int n = n_tokens * n_ff;
    int threads = 256;
    int blocks = cdiv(n, threads);
    swiglu_packed_kernel<<<blocks, threads, 0, stream>>>(output, packed, n_ff, n);
}
