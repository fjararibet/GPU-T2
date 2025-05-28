#include "gameOfLife/cuda.hpp"

GameOfLifeCuda::GameOfLifeCuda(std::vector<std::vector<int>> &grid_) : grid(grid_) {
  int N = 1 << 14;

  float *A = new float[N * N];
  float *B = new float[N * N];
  float *C = new float[N * N];

  // Initialize A and B matrices on the host
  for (int i = 0; i < N * N; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }

  // Allocate device memory for matrices A, B, and C
  float *dA, *dB, *dC;
  cudaMalloc((void **)&dA, N * N * sizeof(float));
  cudaMalloc((void **)&dB, N * N * sizeof(float));
  cudaMalloc((void **)&dC, N * N * sizeof(float));
}

void GameOfLifeCuda::tick() {}

const std::vector<std::vector<int>> GameOfLifeCuda::get_grid() const { return grid; }
