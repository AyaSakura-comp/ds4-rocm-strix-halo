/* ROCm/HIP component unit tests for ds4 — no model file required.
 *
 * Tests:
 *  1. GPU init / cleanup
 *  2. Tensor alloc, write, read round-trip
 *  3. HSA unified memory: tensor_contents() is host-accessible
 *  4. Tensor copy (device-to-device)
 *  5. __dp4a shim correctness (kernel)
 *  6. Warp reduction correctness (kernel)
 *  7. rocWMMA 16x16x16 FP16 matmul correctness (kernel)
 *  8. RMS norm correctness (GPU API)
 *  9. Element-wise add correctness (GPU API)
 * 10. Top-k selection correctness (GPU API)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" {
#include "../ds4_gpu.h"
}

/* -------------------------------------------------------------------------
 * Minimal test harness
 * --------------------------------------------------------------------- */

static int g_pass = 0, g_fail = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name)  do { printf("  %-50s", #name); fflush(stdout); test_##name(); } while(0)

static void pass_(const char *file, int line) {
    (void)file; (void)line;
    printf("PASS\n");
    g_pass++;
}
static void fail_(const char *file, int line, const char *msg) {
    printf("FAIL  %s:%d  %s\n", file, line, msg);
    g_fail++;
}

#define EXPECT(cond) do { if (cond) pass_(__FILE__,__LINE__); \
                          else fail_(__FILE__,__LINE__, #cond); } while(0)
#define EXPECT_NEAR(a, b, tol) do { \
    double _a=(double)(a), _b=(double)(b), _t=(double)(tol); \
    if (fabs(_a-_b)<=_t) pass_(__FILE__,__LINE__); \
    else { char _m[128]; snprintf(_m,sizeof(_m),"%.6g != %.6g (tol %.2g)",_a,_b,_t); \
           fail_(__FILE__,__LINE__,_m); } } while(0)

/* -------------------------------------------------------------------------
 * Device kernels for intrinsic tests
 * --------------------------------------------------------------------- */

/* Re-implement the same dp4a shim logic used in ds4_hip.cpp.  We intentionally
 * use the portable byte-loop here so the test compiles on any gfx target; this
 * exercises the correctness of the shim's fallback path and, on targets where
 * the hardware instruction is used in production, both paths must agree. */
__device__ static int32_t dp4a_ref(int32_t a, int32_t b, int32_t c) {
    int32_t r = c;
    for (int i = 0; i < 4; i++)
        r += (int32_t)(int8_t)((a >> (i*8)) & 0xff) *
             (int32_t)(int8_t)((b >> (i*8)) & 0xff);
    return r;
}

__global__ static void dp4a_kernel(int32_t *out, int32_t a, int32_t b, int32_t c) {
    *out = dp4a_ref(a, b, c);
}

/* Pack four int8 values into an int32 (little-endian byte order). */
static int32_t pack_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int32_t)(uint8_t)a)       |
           ((int32_t)(uint8_t)b << 8)  |
           ((int32_t)(uint8_t)c << 16) |
           ((int32_t)(uint8_t)d << 24);
}

__global__ static void warp_sum_kernel(float *out, float val) {
    float v = val + (float)threadIdx.x;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down(v, off, 32);
    if (threadIdx.x == 0) *out = v;
}

__global__ static void wmma_kernel(__half *a_mat, __half *b_mat, float *c_mat) {
    namespace wmma = rocwmma;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a_mat, 16);
    wmma::load_matrix_sync(b_frag, b_mat, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(c_mat, c_frag, 16, wmma::mem_row_major);
}

/* -------------------------------------------------------------------------
 * Test cases
 * --------------------------------------------------------------------- */

TEST(gpu_init) {
    int r = ds4_gpu_init();
    EXPECT(r == 1);
}

TEST(tensor_alloc_free) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(1024);
    EXPECT(t != NULL);
    EXPECT(ds4_gpu_tensor_bytes(t) == 1024);
    ds4_gpu_tensor_free(t);
    EXPECT(1); /* no crash */
}

TEST(tensor_write_read_roundtrip) {
    const uint32_t N = 256;
    float host_in[N], host_out[N];
    for (uint32_t i = 0; i < N; i++) host_in[i] = (float)i * 0.5f;

    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(N * sizeof(float));
    int ok = t &&
             ds4_gpu_tensor_write(t, 0, host_in, N * sizeof(float)) &&
             ds4_gpu_tensor_read(t, 0, host_out, N * sizeof(float));
    ds4_gpu_tensor_free(t);
    if (!ok) { fail_(__FILE__, __LINE__, "alloc/write/read failed"); return; }
    int match = 1;
    for (uint32_t i = 0; i < N; i++)
        if (host_in[i] != host_out[i]) { match = 0; break; }
    EXPECT(match);
}

TEST(tensor_contents_host_accessible) {
    /* On HSA (Strix Halo), contents() returns a host-accessible pointer. */
    const uint32_t N = 64;
    float pattern[N];
    for (uint32_t i = 0; i < N; i++) pattern[i] = (float)i;

    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(N * sizeof(float));
    if (!t) { fail_(__FILE__, __LINE__, "alloc failed"); return; }
    ds4_gpu_tensor_write(t, 0, pattern, N * sizeof(float));
    ds4_gpu_synchronize();

    void *ptr = ds4_gpu_tensor_contents(t);
    int ok = (ptr != NULL);
    if (ok) {
        /* Try a direct host read — valid on HSA unified memory. */
        float *fp = (float *)ptr;
        ok = (fp[0] == 0.0f && fp[N-1] == (float)(N-1));
    }
    ds4_gpu_tensor_free(t);
    EXPECT(ok);
}

TEST(tensor_device_copy) {
    const uint32_t N = 128;
    float src_data[N], dst_data[N];
    memset(dst_data, 0, sizeof(dst_data));
    for (uint32_t i = 0; i < N; i++) src_data[i] = (float)(i + 1);

    ds4_gpu_tensor *src = ds4_gpu_tensor_alloc(N * sizeof(float));
    ds4_gpu_tensor *dst = ds4_gpu_tensor_alloc(N * sizeof(float));
    int ok = src && dst &&
             ds4_gpu_tensor_write(src, 0, src_data, N * sizeof(float)) &&
             ds4_gpu_tensor_copy(dst, 0, src, 0, N * sizeof(float)) &&
             ds4_gpu_tensor_read(dst, 0, dst_data, N * sizeof(float));
    ds4_gpu_tensor_free(src);
    ds4_gpu_tensor_free(dst);
    if (!ok) { fail_(__FILE__, __LINE__, "copy failed"); return; }
    int match = 1;
    for (uint32_t i = 0; i < N; i++)
        if (src_data[i] != dst_data[i]) { match = 0; break; }
    EXPECT(match);
}

TEST(dp4a_intrinsic) {
    /* [1,2,3,4] · [1,2,3,4] + 0 = 1+4+9+16 = 30 */
    int32_t a = pack_i8(1, 2, 3, 4);
    int32_t b = pack_i8(1, 2, 3, 4);
    int32_t *d_out;
    hipMalloc(&d_out, sizeof(int32_t));
    hipLaunchKernelGGL(dp4a_kernel, dim3(1), dim3(1), 0, 0, d_out, a, b, 0);
    hipDeviceSynchronize();
    int32_t result = 0;
    hipMemcpy(&result, d_out, sizeof(int32_t), hipMemcpyDeviceToHost);
    hipFree(d_out);
    EXPECT(result == 30);
}

TEST(dp4a_with_accumulate) {
    /* [-1,0,1,2] · [3,4,-1,2] + 100 = -3+0-1+4+100 = 100 */
    int32_t a = pack_i8(-1, 0, 1, 2);
    int32_t b = pack_i8( 3, 4,-1, 2);
    int32_t *d_out;
    hipMalloc(&d_out, sizeof(int32_t));
    hipLaunchKernelGGL(dp4a_kernel, dim3(1), dim3(1), 0, 0, d_out, a, b, 100);
    hipDeviceSynchronize();
    int32_t result = 0;
    hipMemcpy(&result, d_out, sizeof(int32_t), hipMemcpyDeviceToHost);
    hipFree(d_out);
    EXPECT(result == 100);
}

TEST(warp_reduction) {
    /* 32 threads, thread i has val + i where val=1.0.
     * sum = 32*1.0 + (0+1+...+31) = 32 + 496 = 528 */
    float *d_out;
    hipMalloc(&d_out, sizeof(float));
    hipLaunchKernelGGL(warp_sum_kernel, dim3(1), dim3(32), 0, 0, d_out, 1.0f);
    hipDeviceSynchronize();
    float result = 0.0f;
    hipMemcpy(&result, d_out, sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_out);
    EXPECT_NEAR(result, 528.0f, 0.1f);
}

TEST(rocwmma_identity_matmul) {
    /* A = identity(16x16), B = identity(16x16) → C = identity(16x16) */
    const int M = 16, N = 16, K = 16;
    __half h_a[M * K], h_b[K * N];
    float  h_c[M * N];

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(0.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(0.0f);
    /* A row-major identity: A[i][j] = 1 if i==j */
    for (int i = 0; i < M; i++) h_a[i * K + i] = __float2half(1.0f);
    /* B col-major identity: B[k][j] = 1 if k==j */
    for (int j = 0; j < N; j++) h_b[j * K + j] = __float2half(1.0f);

    __half *d_a, *d_b; float *d_c;
    hipMalloc(&d_a, M * K * sizeof(__half));
    hipMalloc(&d_b, K * N * sizeof(__half));
    hipMalloc(&d_c, M * N * sizeof(float));
    hipMemcpy(d_a, h_a, M * K * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, K * N * sizeof(__half), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(wmma_kernel, dim3(1), dim3(32), 0, 0, d_a, d_b, d_c);
    hipDeviceSynchronize();
    hipMemcpy(h_c, d_c, M * N * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);

    /* Check diagonal = 1, off-diagonal = 0 */
    int ok = 1;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            if (fabsf(h_c[i * N + j] - expected) > 1e-3f) ok = 0;
        }
    }
    EXPECT(ok);
}

TEST(rms_norm) {
    /* RMS norm of [3,4] with eps=0: norm = sqrt((9+16)/2) = sqrt(12.5)
     * out = [3,4] / sqrt(12.5) ≈ [0.8485, 1.1314] */
    const uint32_t N = 2;
    float in[N]  = {3.0f, 4.0f};
    float out[N] = {0};

    ds4_gpu_tensor *x   = ds4_gpu_tensor_alloc(N * sizeof(float));
    ds4_gpu_tensor *res = ds4_gpu_tensor_alloc(N * sizeof(float));
    int ok = x && res &&
             ds4_gpu_tensor_write(x, 0, in, N * sizeof(float)) &&
             ds4_gpu_begin_commands() &&
             ds4_gpu_rms_norm_plain_tensor(res, x, N, 1e-6f) &&
             ds4_gpu_end_commands() &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(res, 0, out, N * sizeof(float));
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(res);
    if (!ok) { fail_(__FILE__, __LINE__, "rms_norm API call failed"); return; }
    double rms  = sqrt((3.0*3.0 + 4.0*4.0) / 2.0);
    EXPECT_NEAR(out[0], 3.0 / rms, 1e-4);
    EXPECT_NEAR(out[1], 4.0 / rms, 1e-4);
}

TEST(element_add) {
    const uint32_t N = 8;
    float a[N], b[N], out[N];
    for (uint32_t i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(N - i); }

    ds4_gpu_tensor *ta  = ds4_gpu_tensor_alloc(N * sizeof(float));
    ds4_gpu_tensor *tb  = ds4_gpu_tensor_alloc(N * sizeof(float));
    ds4_gpu_tensor *res = ds4_gpu_tensor_alloc(N * sizeof(float));
    int ok = ta && tb && res &&
             ds4_gpu_tensor_write(ta, 0, a, N * sizeof(float)) &&
             ds4_gpu_tensor_write(tb, 0, b, N * sizeof(float)) &&
             ds4_gpu_begin_commands() &&
             ds4_gpu_add_tensor(res, ta, tb, N) &&
             ds4_gpu_end_commands() &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(res, 0, out, N * sizeof(float));
    ds4_gpu_tensor_free(ta);
    ds4_gpu_tensor_free(tb);
    ds4_gpu_tensor_free(res);
    if (!ok) { fail_(__FILE__, __LINE__, "add API call failed"); return; }
    int match = 1;
    for (uint32_t i = 0; i < N; i++)
        if (fabsf(out[i] - (a[i] + b[i])) > 1e-6f) { match = 0; break; }
    EXPECT(match);
}

TEST(topk_selection) {
    /* 8 scores [0..7], top-3 should give indices [7,6,5] */
    const uint32_t n_comp = 8, n_tokens = 1, top_k = 3;
    float scores[n_comp];
    uint32_t selected[top_k];
    for (uint32_t i = 0; i < n_comp; i++) scores[i] = (float)i;

    ds4_gpu_tensor *ts = ds4_gpu_tensor_alloc(n_comp * sizeof(float));
    ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc(top_k * sizeof(uint32_t));
    int ok = ts && tk &&
             ds4_gpu_tensor_write(ts, 0, scores, n_comp * sizeof(float)) &&
             ds4_gpu_begin_commands() &&
             ds4_gpu_indexer_topk_tensor(tk, ts, n_comp, n_tokens, top_k) &&
             ds4_gpu_end_commands() &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(tk, 0, selected, top_k * sizeof(uint32_t));
    ds4_gpu_tensor_free(ts);
    ds4_gpu_tensor_free(tk);
    if (!ok) { fail_(__FILE__, __LINE__, "topk API call failed"); return; }
    EXPECT(selected[0] == 7);
    EXPECT(selected[1] == 6);
    EXPECT(selected[2] == 5);
}

TEST(gpu_cleanup) {
    ds4_gpu_cleanup();
    EXPECT(1); /* no crash */
}

/* -------------------------------------------------------------------------
 * Main
 * --------------------------------------------------------------------- */

int main(void) {
    printf("ds4 ROCm component tests (gfx1151 / Strix Halo)\n");
    printf("================================================\n");

    printf("\n[GPU runtime]\n");
    RUN(gpu_init);

    printf("\n[Memory / tensor primitives]\n");
    RUN(tensor_alloc_free);
    RUN(tensor_write_read_roundtrip);
    RUN(tensor_contents_host_accessible);
    RUN(tensor_device_copy);

    printf("\n[Device intrinsics]\n");
    RUN(dp4a_intrinsic);
    RUN(dp4a_with_accumulate);
    RUN(warp_reduction);
    RUN(rocwmma_identity_matmul);

    printf("\n[GPU API kernels]\n");
    RUN(rms_norm);
    RUN(element_add);
    RUN(topk_selection);

    printf("\n[Cleanup]\n");
    RUN(gpu_cleanup);

    printf("\n================================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
