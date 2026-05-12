/* Performance benchmark for ds4 ROCm/HIP backend.
 * Measures throughput (GB/s and TFLOPs) for Prefill and Decode.
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

extern "C" {
#include "../ds4_gpu.h"
}

static double get_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void benchmark_matmul_f16(uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, const char *label) {
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    
    // Allocate dummy weights in a mapped range
    uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    void *model_mem = malloc(weight_bytes + 4096);
    void *model_map = (void *)(((uintptr_t)model_mem + 4095) & ~4095ULL);
    memset(model_map, 0, weight_bytes);
    
    ds4_gpu_set_model_map(model_map, weight_bytes);
    
    // Warmup
    ds4_gpu_begin_commands();
    ds4_gpu_matmul_f16_tensor(out, model_map, weight_bytes, 0, in_dim, out_dim, x, n_tok);
    ds4_gpu_end_commands();
    ds4_gpu_synchronize();
    
    const int iters = 10;
    double start = get_ms();
    for (int i = 0; i < iters; i++) {
        ds4_gpu_begin_commands();
        ds4_gpu_matmul_f16_tensor(out, model_map, weight_bytes, 0, in_dim, out_dim, x, n_tok);
        ds4_gpu_end_commands();
        ds4_gpu_synchronize();
    }
    double end = get_ms();
    double avg_ms = (end - start) / iters;
    
    double bytes = (double)weight_bytes + (double)n_tok * in_dim * sizeof(float) + (double)n_tok * out_dim * sizeof(float);
    double gbs = (bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);
    double tflops = (2.0 * in_dim * out_dim * n_tok / 1e12) / (avg_ms / 1000.0);
    
    printf("%-20s: %8.2f ms | %8.2f GB/s | %8.4f TFLOPs (n_tok=%u, %ux%u)\n", 
           label, avg_ms, gbs, tflops, n_tok, out_dim, in_dim);
    
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(model_mem);
}

static void benchmark_attention_prefill(uint32_t n_tokens, uint32_t n_head, uint32_t head_dim, uint32_t window, const char *label) {
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)n_tokens * head_dim * sizeof(float));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
    
    uint64_t sinks_bytes = (uint64_t)n_head * sizeof(float);
    void *model_mem = malloc(sinks_bytes + 4096);
    void *model_map = (void *)(((uintptr_t)model_mem + 4095) & ~4095ULL);
    ds4_gpu_set_model_map(model_map, sinks_bytes);

    // Warmup
    ds4_gpu_begin_commands();
    ds4_gpu_attention_prefill_raw_heads_tensor(heads, model_map, sinks_bytes, 0, q, kv, n_tokens, window, n_head, head_dim);
    ds4_gpu_end_commands();
    ds4_gpu_synchronize();

    const int iters = 5;
    double start = get_ms();
    for (int i = 0; i < iters; i++) {
        ds4_gpu_begin_commands();
        ds4_gpu_attention_prefill_raw_heads_tensor(heads, model_map, sinks_bytes, 0, q, kv, n_tokens, window, n_head, head_dim);
        ds4_gpu_end_commands();
        ds4_gpu_synchronize();
    }
    double end = get_ms();
    double avg_ms = (end - start) / iters;
    
    printf("%-20s: %8.2f ms | (n_tok=%u, heads=%u, dim=%u, win=%u)\n", 
           label, avg_ms, n_tokens, n_head, head_dim, window);

    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(heads);
    free(model_mem);
}

int main(int argc, char **argv) {
    if (ds4_gpu_init() != 1) return 1;
    
    printf("ds4 ROCm Performance Benchmark (gfx1151 / Strix Halo)\n");
    printf("====================================================\n\n");
    
    // Decode test (1 token, memory bound)
    benchmark_matmul_f16(4096, 4096, 1, "F16 Decode (4k)");
    benchmark_matmul_f16(8192, 8192, 1, "F16 Decode (8k)");
    
    // Prefill test (Compute bound)
    benchmark_matmul_f16(4096, 4096, 512, "F16 Prefill (512)");
    
    // Q2-like Decode (84 bytes per block of 256)
    // We'll simulate this by doing a 4k x 4k matmul where weights are treated as Q2
    // Actually, I'll just add a simulated label for the F16 since we don't have a Q2 test yet
    // benchmark_matmul_q2(4096, 4096, 1, "Q2 Decode (4k)");
    
    // Attention Prefill (Short Context)
    benchmark_attention_prefill(1024, 32, 128, 1024, "Attn Short (1k)");
    
    // Attention Prefill (Long Context)
    benchmark_attention_prefill(32768, 32, 128, 32768, "Attn Long (32k)");
    
    ds4_gpu_cleanup();
    return 0;
}
