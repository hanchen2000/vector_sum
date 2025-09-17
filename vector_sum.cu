#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N (10 * 1000 * 1000)   // 10M elements
#define THREADS_PER_BLOCK 256

// GPU kernel: C = A + B
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {

    // Allocate unified memory
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));


    // Initialize data
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // -------- CPU run --------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU time: " << cpu_time << " ms\n";

    // Kernel config
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // -------- Single GPU run (timed) --------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(A, B, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_single = 0.0f;
    cudaEventElapsedTime(&gpu_time_single, start, stop);
    std::cout << "GPU single kernel time: " << gpu_time_single << " ms\n";

    // -------- Another GPU run (timed) --------
    cudaEventRecord(start);
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(A, B, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_2 = 0.0f;
    cudaEventElapsedTime(&gpu_time_2, start, stop);
    std::cout << "GPU after 1 warm-up runs: " << gpu_time_2 << " ms\n";

    // -------- 256 warm-up runs (no timing) --------
    for (int i = 0; i < 256; i++) {
        vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(A, B, C, N);
    }
    cudaDeviceSynchronize();

    // -------- Another GPU run (timed) --------
    cudaEventRecord(start);
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(A, B, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_post = 0.0f;
    cudaEventElapsedTime(&gpu_time_post, start, stop);
    std::cout << "GPU after 256 warm-up runs: " << gpu_time_post << " ms\n";

    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

