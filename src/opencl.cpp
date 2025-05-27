#define __CL_ENABLE_EXCEPTIONS
#include "gameOfLife/opencl.hpp"
#include <iostream>
#include <utility>

std::string kernel = R"(
kernel void gameOfLife(global int* In, global int* Out, int n, int m) {
    const int idx = get_global_id(0);
    if (idx < n * m) {
        int curr_row = idx / m;
        int curr_col = idx % m;
        int neighbor_count = 0;
        for(int row = curr_row - 1; row <= curr_row + 1; row++) {
          if (row < 0 || row >= n) continue;
          for(int col = curr_col - 1; col <= curr_col + 1; col++) {
            if (col < 0 || col >= m) continue;
            if (row == curr_row && col == curr_col) continue;
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
)";

GameOfLifeOpenCL::GameOfLifeOpenCL(std::vector<std::vector<int>> &grid_) : grid(grid_) {
  n = grid.size();
  m = grid[0].size();
  size_t N_ELEMENTS = n * m;

  try {
    // Query for platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

    context = cl::Context(devices);
    queue = cl::CommandQueue(context, devices[0]);

    bufferIn = cl::Buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(int));
    bufferOut = cl::Buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(int));

    bufferInHost.resize(N_ELEMENTS);
    bufferOutHost.resize(N_ELEMENTS);

    // Copy initial grid to bufferInHost
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        bufferInHost[i * m + j] = grid[i][j];
      }
    }

    cl::Program::Sources source{{kernel.c_str(), kernel.length()}};
    cl::Program program(context, source);
    program.build(devices);

    gol_kernel = cl::Kernel(program, "gameOfLife");

    auto max_work_group_size = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    if (local_size > max_work_group_size) {
      local_size = max_work_group_size;
    }
    global_size = ((N_ELEMENTS + local_size - 1) / local_size) * local_size;

    // Write initial data once
    queue.enqueueWriteBuffer(bufferIn, CL_TRUE, 0, N_ELEMENTS * sizeof(int), bufferInHost.data());
    queue.finish();
  } catch (cl::Error err) {
    std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
  }
}

void GameOfLifeOpenCL::tick() {
  size_t N_ELEMENTS = n * m;

  try {
    // Set kernel arguments for this run
    gol_kernel.setArg(0, bufferIn);
    gol_kernel.setArg(1, bufferOut);
    gol_kernel.setArg(2, (int)n);
    gol_kernel.setArg(3, (int)m);

    // Run the kernel
    queue.enqueueNDRangeKernel(gol_kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

    // Read output buffer back to host
    queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, N_ELEMENTS * sizeof(int), bufferOutHost.data());
    queue.finish();

    // Update grid from bufferOutHost
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        grid[i][j] = bufferOutHost[i * m + j];
      }
    }

    // Swap buffers for next iteration:
    std::swap(bufferIn, bufferOut);
    std::swap(bufferInHost, bufferOutHost);

    // Write new input buffer for next kernel execution
    queue.enqueueWriteBuffer(bufferIn, CL_TRUE, 0, N_ELEMENTS * sizeof(int), bufferInHost.data());
    queue.finish();
  } catch (cl::Error err) {
    std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
  }
}

const std::vector<std::vector<int>> GameOfLifeOpenCL::get_grid() const { return grid; }
