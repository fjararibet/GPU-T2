#define __CL_ENABLE_EXCEPTIONS
#include "gameOfLife/opencl.hpp"
#include <CL/opencl.hpp>
#include <iostream>
#include <utility>

std::string kernel = R"(
kernel void gameOfLife(global int* In, global int* Out, int n, int m) {
    const int idx = get_global_id(0);
    if (idx < n * m) {
        Out[idx] = In[idx];
        int row = idx / m;
        int col = idx % m;
        Out[row * m + col] = In[row * m + col];
    }
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
    cl::Program::Sources source{{kernel.c_str(), kernel.length()}};
    cl::Program program;
    cl_int err;
    try {
      program = cl::Program(context, source);
      err = program.build(devices);
    } catch (cl::BuildError &e) {
      std::cerr << "Build failed: " << e.what() << std::endl;
      for (const auto &pair : e.getBuildLog()) {
        std::cerr << "Device: " << pair.first.getInfo<CL_DEVICE_NAME>() << "\n";
        std::cerr << "Build log:\n" << pair.second << std::endl;
      }
    }
    cl::Kernel gol_kernel(program, "gameOfLife");
    gol_kernel.setArg(0, bufferIn);
    gol_kernel.setArg(1, bufferOut);
    gol_kernel.setArg(2, (int)grid.size());
    gol_kernel.setArg(3, (int)grid[0].size());

    // Execute the kernel
    size_t local_size = 256;
    size_t global_size = ((N_ELEMENTS + local_size - 1) / local_size) * local_size;
    cl::NDRange global(global_size);
    cl::NDRange local(local_size);
    queue.enqueueNDRangeKernel(gol_kernel, cl::NullRange, global, local);

    // Copy the output data back to the host
    std::vector<int> Out(grid.size() * grid[0].size());
    queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, N_ELEMENTS * sizeof(int), Out.data());
    bool ok = true;
    for (size_t i = 0; i < grid.size(); i++) {
      for (size_t j = 0; j < grid[0].size(); j++) {
        ok = Out[i * grid[0].size() + j] != In[i * grid[0].size() + j] ? false : ok;
      }
    }
    std::cout << (ok ? "ok" : "not ok") << std::endl;

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
