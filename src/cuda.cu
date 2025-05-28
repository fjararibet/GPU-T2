#include <cuda_runtime.h>
#include <iostream>
#include "gameOfLife/cuda.hpp"

__global__ void gameOfLifeKernel(int* In, int* Out, int n, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= m) return;

    int neighbor_count = 0;
    for (int i = row - 1; i <= row + 1; i++) {
        if (i < 0 || i >= n) continue;
        for (int j = col - 1; j <= col + 1; j++) {
            if (j < 0 || j >= m) continue;
            if (i == row && j == col) continue;
            if (In[i * m + j]) neighbor_count++;
        }
    }

    int current = In[row * m + col];
    int new_cell = current && (neighbor_count == 2 || neighbor_count == 3);
    if (neighbor_count == 3) {
      new_cell = 1;
    }
    // Out[row * m + col] = new_cell;
    Out[row * m + col] = current;
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

    bool ok = true;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
          ok = hostIn[i * m + j] != hostOut[i * m + j] ? false : ok;
    std::cout << (ok ? "ok" : "not") << std::endl;

    std::swap(deviceIn, deviceOut);
    std::swap(hostIn, hostOut);

    cudaMemcpy(deviceIn, hostIn.data(), N_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
}

const std::vector<std::vector<int>> GameOfLifeCuda::get_grid() const {
    return grid;
}

