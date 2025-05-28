#include <cuda_runtime.h>
#include "gameOfLife/cuda.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

__global__ void gameOfLifeKernel(int* In, int* Out, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) {
        int curr_row = idx / m;
        int curr_col = idx % m;
        int neighbor_count = 0;

        for (int row = curr_row - 1; row <= curr_row + 1; row++) {
            if (row < 0 || row >= n) continue;
            for (int col = curr_col - 1; col <= curr_col + 1; col++) {
                if (col < 0 || col >= m) continue;
                if (row == curr_row && col == curr_col) continue;
                if (In[row * m + col]) neighbor_count++;
            }
        }

        int current = In[curr_row * m + curr_col];
        int new_cell = (current && (neighbor_count == 2 || neighbor_count == 3)) || (!current && neighbor_count == 3);
        Out[curr_row * m + curr_col] = new_cell;
    }
}


GameOfLifeCuda::GameOfLifeCuda(std::vector<std::vector<int>>& grid_) : grid(grid_) {
    n = grid.size();
    m = grid[0].size();
    N_ELEMENTS = n * m;
    blocksPerGrid = (N_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;

    hostIn.resize(N_ELEMENTS);
    hostOut.resize(N_ELEMENTS);

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            hostIn[i * m + j] = grid[i][j];

    cudaMalloc(&deviceIn, N_ELEMENTS * sizeof(int));
    cudaMalloc(&deviceOut, N_ELEMENTS * sizeof(int));

    cudaMemcpy(deviceIn, hostIn.data(), N_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
}

void GameOfLifeCuda::tick() {
    gameOfLifeKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceIn, deviceOut, n, m);
    cudaDeviceSynchronize();

    cudaMemcpy(hostOut.data(), deviceOut, N_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            grid[i][j] = hostOut[i * m + j];

    std::swap(deviceIn, deviceOut);
    std::swap(hostIn, hostOut);

    cudaMemcpy(deviceIn, hostIn.data(), N_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
}

const std::vector<std::vector<int>> GameOfLifeCuda::get_grid() const {
    return grid;
}

