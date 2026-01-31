#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__
void matVecMulKernel(float* A, const float* B, const float* C, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N)
    {
        float sum = 0.0f;

        for (int j = 0; j < N; ++j)
        {
            sum += B[row * N + j] * C[j];
        }

        A[row] = sum;
    }
}

// Host stub function
void matVecMul(float* h_A, const float* h_B, const float* h_C, int N)
{
    float *d_A, *d_B, *d_C;

    size_t matrixSize = N * N * sizeof(float);
    size_t vectorSize = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, vectorSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, vectorSize);

    // Copy data to device
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, vectorSize, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    matVecMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Wait for GPU
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_A, d_A, vectorSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    int N = 4;  // matrix dimension

    size_t matrixSize = N * N * sizeof(float);
    size_t vectorSize = N * sizeof(float);

    float* B = (float*)malloc(matrixSize);
    float* C = (float*)malloc(vectorSize);
    float* A = (float*)malloc(vectorSize);

    // Initialize matrix B
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B[i * N + j] = i + j; // simple pattern
        }
    }

    // Initialize vector C
    for (int i = 0; i < N; ++i)
    {
        C[i] = 1.0f;
    }

    // Call GPU function
    matVecMul(A, B, C, N);

    // Print result
    cout << "Result vector A:\n";
    for (int i = 0; i < N; ++i)
    {
        cout << A[i] << " ";
    }
    cout << endl;

    free(A);
    free(B);
    free(C);

    return 0;
}
