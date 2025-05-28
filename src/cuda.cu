#include "gameOfLife/cuda.hpp"

__global__ void gameOfLifeKernel(int *In, int *Out, int n, int m) {
  int curr_col = blockIdx.x * blockDim.x + threadIdx.x;
  int curr_row = blockIdx.y * blockDim.y + threadIdx.y;
  if (curr_row < n && curr_col < m) {
    int neighbor_count = 0;
    for (int row = curr_row - 1; row <= curr_row + 1; row++) {
      if (row < 0 || row >= n)
        continue;
      for (int col = curr_col - 1; col <= curr_col + 1; col++) {
        if (col < 0 || col >= m)
          continue;
        if (row == curr_row && col == curr_col)
          continue;
        if (In[row * m + col]) {
          neighbor_count++;
        }
      }
    }

    int new_cell = In[curr_row * m + curr_col] && (neighbor_count == 2 || neighbor_count == 3);
    if (neighbor_count == 3) {
      new_cell = 1;
    }
    Out[curr_row * m + curr_col] = new_cell;
  }
}

GameOfLifeCuda::GameOfLifeCuda(std::vector<std::vector<int>> &grid_) : grid(grid_) {
  n = grid.size();
  m = grid[0].size();
  int N_ELEMENTS = n * m;

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
  int N_ELEMENTS = n * m;
  threadsPerBlockX = 16;
  threadsPerBlockY = 16;
  dim3 blockDim(threadsPerBlockX, threadsPerBlockY);
  dim3 gridDim((m + threadsPerBlockX - 1) / threadsPerBlockX, (n + threadsPerBlockY - 1) / threadsPerBlockY);

  gameOfLifeKernel<<<gridDim, blockDim>>>(deviceIn, deviceOut, n, m);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA launch error: " << cudaGetErrorString(err) << std::endl;
  }

  cudaDeviceSynchronize();

  err = cudaMemcpy(hostOut.data(), deviceOut, N_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
  }

  cudaMemcpy(hostOut.data(), deviceOut, N_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < m; ++j)
      grid[i][j] = hostOut[i * m + j];

  std::swap(deviceIn, deviceOut);
  std::swap(hostIn, hostOut);

  cudaMemcpy(deviceIn, hostIn.data(), N_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
}

const std::vector<std::vector<int>> GameOfLifeCuda::get_grid() const { return grid; }
