#include "sampling.h"
#include "utils.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

__global__ void bf16_to_f32_kernel(float* __restrict__ output, const __nv_bfloat16* __restrict__ input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

void launch_bf16_to_f32(float* output, const __nv_bfloat16* input, int n, cudaStream_t stream) {
    bf16_to_f32_kernel<<<cdiv(n, 256), 256, 0, stream>>>(output, input, n);
}

// GPU argmax kernel - reduces vocab to find max token ID
__global__ void argmax_kernel(
    int* __restrict__ result,
    float* __restrict__ result_val,
    const float* __restrict__ logits,
    int n
) {
    extern __shared__ char smem_raw[];
    float* s_vals = (float*)smem_raw;
    int* s_idxs = (int*)(smem_raw + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float best_val = -1e30f;
    int best_idx = 0;
    for (int i = tid; i < n; i += stride) {
        float v = logits[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }

    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s && s_vals[tid + s] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + s];
            s_idxs[tid] = s_idxs[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = s_idxs[0];
        *result_val = s_vals[0];
    }
}

static int* g_argmax_result = nullptr;
static float* g_argmax_val = nullptr;

static int gpu_argmax(float* logits_device, int vocab_size) {
    if (!g_argmax_result) {
        g_argmax_result = cuda_alloc<int>(1);
        g_argmax_val = cuda_alloc<float>(1);
    }
    int threads = 1024;
    size_t smem = threads * (sizeof(float) + sizeof(int));
    argmax_kernel<<<1, threads, smem>>>(g_argmax_result, g_argmax_val, logits_device, vocab_size);
    int result;
    cuda_download(&result, g_argmax_result, 1);
    return result;
}

// Simple CPU-side sampling (good enough for single-token generation)
int sample_token(float* logits_device, int vocab_size, float temperature, int top_k, float top_p) {
    // Greedy (argmax) for temperature <= 0 — done on GPU
    if (temperature <= 0.0f) {
        return gpu_argmax(logits_device, vocab_size);
    }

    // Download logits to host for non-greedy sampling
    std::vector<float> logits(vocab_size);
    cuda_download(logits.data(), logits_device, vocab_size);

    // Apply temperature
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }

    // Find top-k
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    if (top_k > 0 && top_k < vocab_size) {
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
        indices.resize(top_k);
    }

    // Softmax over selected tokens
    float max_logit = logits[indices[0]];
    for (int idx : indices) max_logit = std::max(max_logit, logits[idx]);

    std::vector<float> probs(indices.size());
    float sum = 0.0f;
    for (size_t i = 0; i < indices.size(); i++) {
        probs[i] = expf(logits[indices[i]] - max_logit);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    // Top-p (nucleus) filtering
    if (top_p < 1.0f) {
        // Sort by probability descending
        std::vector<size_t> sorted_idx(probs.size());
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
            [&](size_t a, size_t b) { return probs[a] > probs[b]; });

        float cumsum = 0.0f;
        size_t cutoff = sorted_idx.size();
        for (size_t i = 0; i < sorted_idx.size(); i++) {
            cumsum += probs[sorted_idx[i]];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff
        for (size_t i = cutoff; i < sorted_idx.size(); i++) {
            probs[sorted_idx[i]] = 0.0f;
        }

        // Renormalize
        sum = 0.0f;
        for (auto p : probs) sum += p;
        for (auto& p : probs) p /= sum;
    }

    // Sample
    static std::mt19937 rng(42);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int sampled_idx = dist(rng);

    return indices[sampled_idx];
}
