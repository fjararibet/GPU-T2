#define __CL_ENABLE_EXCEPTIONS
#include "gameOfLife/opencl.hpp"
#include <CL/opencl.hpp>
#include <iostream>
#include <utility>

std::string kernel = R"(
kernel void vecadd( global int* In, global int* Out) {
    const int idx = get_global_id(0);
    Out[idx] = In[idx];
}
)";

GameOfLifeOpenCL::GameOfLifeOpenCL(std::vector<std::vector<int>> &grid_) : grid(grid_) {
  try {
    // Query for platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices;
    // Select the platform.
    size_t platform_id = 0;
    platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

    // Create a context
    cl::Context context(devices);

    size_t device_id = 0;
    cl::CommandQueue queue = cl::CommandQueue(context, devices[device_id]);

    // Create the memory buffers
    size_t N_ELEMENTS = grid.size() * grid[0].size();
    cl::Buffer bufferIn = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(int));
    cl::Buffer bufferOut = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(int));

    // Copy the input data to the input buffers using the command queue.
    std::vector<int> In(grid.size() * grid[0].size());
    for (size_t i = 0; i < grid.size(); i++) {
      for (size_t j = 0; j < grid[0].size(); j++) {
        In[i * grid[0].size() + j] = grid[i][j];
      }
    }
    queue.enqueueWriteBuffer(bufferIn, CL_FALSE, 0, N_ELEMENTS * sizeof(int), In.data());
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

  } catch (cl::Error err) {
    std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
  }
}

void GameOfLifeOpenCL::tick() {
  std::vector<std::vector<int>> next_grid(grid.size(), std::vector(grid[0].size(), 0));
  for (std::size_t i = 0; i < grid.size(); i++) {
    for (std::size_t j = 0; j < grid[0].size(); j++) {
      std::vector<std::pair<std::size_t, std::size_t>> neighbors = {
          {i - 1, j}, {i - 1, j - 1}, {i, j - 1}, {i + 1, j - 1},
          {i + 1, j}, {i + 1, j + 1}, {i, j + 1}, {i - 1, j + 1},
      };
      int neighbor_count = 0;
      for (auto [ni, nj] : neighbors) {
        if (ni < 0 || nj < 0 || ni >= grid.size() || nj >= grid[0].size())
          continue;

        if (grid[ni][nj]) {
          neighbor_count++;
        }
      }
      next_grid[i][j] = (neighbor_count == 2 || neighbor_count == 3) && grid[i][j];
      if (neighbor_count == 3) {
        next_grid[i][j] = 1;
      }
    }
  }
  grid = next_grid;
}

const std::vector<std::vector<int>> GameOfLifeOpenCL::get_grid() const { return grid; }
