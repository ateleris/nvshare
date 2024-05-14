#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

#define DEV
#define GLOBAL_MEMORY

constexpr int BLOCK_SIZE = 32;

using namespace std;

#define gpuErrCheck(ans)                      \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << endl;
        if (abort)
        {
            exit(code);
        }
    }
}

void populateMatrixBuffer(float *buffer, int dimSize)
{
    for (int i = 0; i < dimSize; i++)
    {
        for (int j = 0; j < dimSize; j++)
        {
            buffer[i * dimSize + j] = 1.0f / (j + 1);
        }
    }
}

__global__ void matMulGPU(const float *matrixA, const float *matrixB, float *matrixC, int dimSize)
{
    const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (x < dimSize && y < dimSize)
    {
        float product = 0.0f;
        for (int i = 0; i < dimSize; i++)
        {
            product += matrixA[x * dimSize + i] * matrixB[i * dimSize + y];
        }
        matrixC[x * dimSize + y] = product;
    }
}

int main()
{
    int MAX_DIM_SIZE = 16384;

    for (int DIM_SIZE = 64; DIM_SIZE <= MAX_DIM_SIZE; DIM_SIZE <<= 1)
    {
        std::cout << "-----------------------------------------------" << endl;
        std::cout << "DIM_SIZE: " << DIM_SIZE << endl
                  << endl;

        // Allocate host memory
        float *h_matrixA = new float[DIM_SIZE * DIM_SIZE];
        float *h_matrixB = new float[DIM_SIZE * DIM_SIZE];
        float *h_matrixC = new float[DIM_SIZE * DIM_SIZE];
        populateMatrixBuffer(h_matrixA, DIM_SIZE);
        populateMatrixBuffer(h_matrixB, DIM_SIZE);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        float matMulCPUNaiveTimeMs = -1.0f;
        float matMulCPUIntelMKLTimeMs = -1.0f;
        float matMulGPUMs = -1.0f;
        float matMulGPUSharedMemMs = -1.0f;
        float matMulGPUcuBLASMs = -1.0f;

        // Allocate device memory
        float *d_matrixA;
        float *d_matrixB;
        float *d_matrixC;
        gpuErrCheck(cudaMalloc(&d_matrixA, DIM_SIZE * DIM_SIZE * sizeof(float)));
        gpuErrCheck(cudaMalloc(&d_matrixB, DIM_SIZE * DIM_SIZE * sizeof(float)));
        gpuErrCheck(cudaMalloc(&d_matrixC, DIM_SIZE * DIM_SIZE * sizeof(float)));

        // Copy data from host to device
        gpuErrCheck(cudaMemcpy(d_matrixA, h_matrixA, DIM_SIZE * DIM_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrCheck(cudaMemcpy(d_matrixB, h_matrixB, DIM_SIZE * DIM_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Init CUDA events used to meassure timings
        cudaEvent_t startEvent, stopEvent;
        gpuErrCheck(cudaEventCreate(&startEvent));
        gpuErrCheck(cudaEventCreate(&stopEvent));

        // Define grids, blocks and threads
        const int GRID_SIZE = (DIM_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 gridDim(GRID_SIZE, GRID_SIZE); // 2D Grid
        dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);

        {
            // Call kernel (global-memory)
            gpuErrCheck(cudaEventRecord(startEvent, 0));
            matMulGPU<<<gridDim, blockDim>>>(d_matrixA, d_matrixB, d_matrixC, DIM_SIZE);
            gpuErrCheck(cudaEventRecord(stopEvent, 0));
            gpuErrCheck(cudaEventSynchronize(stopEvent));
            gpuErrCheck(cudaEventElapsedTime(&matMulGPUMs, startEvent, stopEvent));
            gpuErrCheck(cudaPeekAtLastError());
            std::cout << "GPU_GLOBAL time [ms]: " << matMulGPUMs << endl;
        }

        // Free memory
        gpuErrCheck(cudaFree(d_matrixA));
        gpuErrCheck(cudaFree(d_matrixB));
        gpuErrCheck(cudaFree(d_matrixC));
        delete[] h_matrixA;
        delete[] h_matrixB;
        delete[] h_matrixC;
    }

    return 0;
}