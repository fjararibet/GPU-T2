#include "gameOfLife/cuda.hpp"
#include <iostream>

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
__global__ void gameOfLifeKernelLocalMem(int *In, int *Out, int n, int m) {
  // Block and thread indices
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + tx;
  int row = blockIdx.y * blockDim.y + ty;

  // Shared memory tile with 1-cell halo
  extern __shared__ int tile[];

  // Shared memory width includes 2 halo cells (1 on each side)
  int shared_width = blockDim.x + 2;
  int shared_x = tx + 1;
  int shared_y = ty + 1;

  // Load current cell into shared memory, and also halo region
  if (row < n && col < m) {
    tile[shared_y * shared_width + shared_x] = In[row * m + col];
  } else {
    tile[shared_y * shared_width + shared_x] = 0;
  }

  // Load halo cells
  if (tx == 0 && col > 0) {
    tile[shared_y * shared_width + 0] = (row < n) ? In[row * m + (col - 1)] : 0;
  }
  if (tx == blockDim.x - 1 && col < m - 1) {
    tile[shared_y * shared_width + (shared_x + 1)] = (row < n) ? In[row * m + (col + 1)] : 0;
  }
  if (ty == 0 && row > 0) {
    tile[0 * shared_width + shared_x] = (col < m) ? In[(row - 1) * m + col] : 0;
  }
  if (ty == blockDim.y - 1 && row < n - 1) {
    tile[(shared_y + 1) * shared_width + shared_x] = (col < m) ? In[(row + 1) * m + col] : 0;
  }

  // Corners
  if (tx == 0 && ty == 0 && col > 0 && row > 0)
    tile[0 * shared_width + 0] = In[(row - 1) * m + (col - 1)];
  if (tx == 0 && ty == blockDim.y - 1 && col > 0 && row < n - 1)
    tile[(shared_y + 1) * shared_width + 0] = In[(row + 1) * m + (col - 1)];
  if (tx == blockDim.x - 1 && ty == 0 && col < m - 1 && row > 0)
    tile[0 * shared_width + (shared_x + 1)] = In[(row - 1) * m + (col + 1)];
  if (tx == blockDim.x - 1 && ty == blockDim.y - 1 && col < m - 1 && row < n - 1)
    tile[(shared_y + 1) * shared_width + (shared_x + 1)] = In[(row + 1) * m + (col + 1)];

  // Synchronize to make sure the tile is fully loaded
  __syncthreads();

  if (row < n && col < m) {
    int neighbor_count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0)
          continue;
        neighbor_count += tile[(shared_y + dy) * shared_width + (shared_x + dx)];
      }
    }

    int current = tile[shared_y * shared_width + shared_x];
    int new_cell = (current && (neighbor_count == 2 || neighbor_count == 3)) || (!current && neighbor_count == 3);
    Out[row * m + col] = new_cell;
  }
}

GameOfLifeCuda::GameOfLifeCuda(std::vector<std::vector<int>> &grid_, int workgroup_x, int workgroup_y, bool local)
    : grid(grid_), workgroup_x(workgroup_x), workgroup_y(workgroup_y), local(local) {
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
  int threadsPerBlockX = workgroup_x;
  int threadsPerBlockY = workgroup_y;
  dim3 blockDim(threadsPerBlockX, threadsPerBlockY);
  int TILE_WIDTH = blockDim.x;
  int TILE_HEIGHT = blockDim.y;
  int sharedMemSize = (TILE_WIDTH + 2) * (TILE_HEIGHT + 2) * sizeof(int);

  dim3 gridDim((m + threadsPerBlockX - 1) / threadsPerBlockX, (n + threadsPerBlockY - 1) / threadsPerBlockY);

  if (local) {
    gameOfLifeKernelLocalMem<<<gridDim, blockDim, sharedMemSize>>>(deviceIn, deviceOut, n, m);
  } else {
    gameOfLifeKernel<<<gridDim, blockDim>>>(deviceIn, deviceOut, n, m);
  }
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
