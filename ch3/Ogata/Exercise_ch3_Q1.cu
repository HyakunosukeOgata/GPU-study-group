#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAT_M 1024
#define MAT_N 1024
#define MAT_K 1024

#define BLOCK_SIZE 256
#define NUM_RUNS 50


// =====================================================
// Row Kernel: one thread computes one ROW
// =====================================================
__global__
void matMulRow(const float* A, const float* B, float* C,
               int M, int N, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) return;

    for (int col = 0; col < N; col++) {

        float sum = 0.0f;

        for (int kk = 0; kk < K; kk++) {
            sum += A[row * K + kk] * B[kk * N + col];
        }

        C[row * N + col] = sum;
    }
}


// =====================================================
// Column Kernel: one thread computes one COLUMN
// =====================================================
__global__
void matMulCol(const float* A, const float* B, float* C,
               int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= N) return;

    for (int row = 0; row < M; row++) {

        float sum = 0.0f;

        for (int kk = 0; kk < K; kk++) {
            sum += A[row * K + kk] * B[kk * N + col];
        }

        C[row * N + col] = sum;
    }
}


// =====================================================
// Initialize matrices
// =====================================================
void initMatrix(float* mat, int size)
{
    for (int i = 0; i < size; i++)
        mat[i] = (float)(rand() % 5);
}


// =====================================================
// Verify results
// =====================================================
void verify(float* A, float* B, int size)
{
    for (int i = 0; i < size; i++) {

        if (fabs(A[i] - B[i]) > 1e-2) {
            printf("Mismatch at %d : %f vs %f\n", i, A[i], B[i]);
            return;
        }
    }

    printf("Results match!\n");
}


int main()
{
    size_t sizeA = MAT_M * MAT_K * sizeof(float);
    size_t sizeB = MAT_K * MAT_N * sizeof(float);
    size_t sizeC = MAT_M * MAT_N * sizeof(float);

    float *h_A, *h_B, *h_C_row, *h_C_col;

    h_A = (float*)malloc(sizeA);
    h_B = (float*)malloc(sizeB);
    h_C_row = (float*)malloc(sizeC);
    h_C_col = (float*)malloc(sizeC);

    initMatrix(h_A, MAT_M * MAT_K);
    initMatrix(h_B, MAT_K * MAT_N);

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed;


    // =====================================================
    // Row Kernel Timing
    // =====================================================

    int gridRow = (MAT_M + BLOCK_SIZE - 1) / BLOCK_SIZE;

    matMulRow<<<gridRow, BLOCK_SIZE>>>(d_A, d_B, d_C,
                                       MAT_M, MAT_N, MAT_K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < NUM_RUNS; i++)
        matMulRow<<<gridRow, BLOCK_SIZE>>>(d_A, d_B, d_C,
                                           MAT_M, MAT_N, MAT_K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    printf("Row kernel avg time: %f ms\n", elapsed / NUM_RUNS);

    cudaMemcpy(h_C_row, d_C, sizeC, cudaMemcpyDeviceToHost);


    // =====================================================
    // Column Kernel Timing
    // =====================================================

    int gridCol = (MAT_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    matMulCol<<<gridCol, BLOCK_SIZE>>>(d_A, d_B, d_C,
                                       MAT_M, MAT_N, MAT_K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < NUM_RUNS; i++)
        matMulCol<<<gridCol, BLOCK_SIZE>>>(d_A, d_B, d_C,
                                           MAT_M, MAT_N, MAT_K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    printf("Column kernel avg time: %f ms\n", elapsed / NUM_RUNS);

    cudaMemcpy(h_C_col, d_C, sizeC, cudaMemcpyDeviceToHost);


    verify(h_C_row, h_C_col, MAT_M * MAT_N);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C_row);
    free(h_C_col);

    return 0;
}
