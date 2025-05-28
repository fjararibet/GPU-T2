#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "gameOfLife/opencl.hpp"
#include <iostream>
#include <utility>

std::string kernel = R"(
kernel void gameOfLife(global int* In, global int* Out, int n, int m) {
    const int curr_col = get_global_id(0);
    const int curr_row = get_global_id(1);
    if (curr_row >= 0 && curr_col < n && curr_col >= 0 && curr_col < m) {
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

GameOfLifeOpenCL::GameOfLifeOpenCL(std::vector<std::vector<int>> &grid, int workgroup_x, int workgroup_y)
    : grid(grid), workgroup_x(workgroup_x), workgroup_y(workgroup_y) {
  n = grid.size();
  m = grid[0].size();
  size_t N_ELEMENTS = n * m;

  try {
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

    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        bufferInHost[i * m + j] = grid[i][j];
      }
    }

    cl::Program::Sources source{{kernel.c_str(), kernel.length()}};
    cl::Program program(context, source);
    program.build(devices);

    gol_kernel = cl::Kernel(program, "gameOfLife");

    // check for max work group size
    // auto max_work_group_size = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    // if (local_size > max_work_group_size) {
    //   local_size = max_work_group_size;
    // }

    queue.enqueueWriteBuffer(bufferIn, CL_TRUE, 0, N_ELEMENTS * sizeof(int), bufferInHost.data());
    queue.finish();
  } catch (cl::Error err) {
    std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
  }
}

void GameOfLifeOpenCL::tick() {
  size_t N_ELEMENTS = n * m;

  try {
    gol_kernel.setArg(0, bufferIn);
    gol_kernel.setArg(1, bufferOut);
    gol_kernel.setArg(2, (int)n);
    gol_kernel.setArg(3, (int)m);

    size_t local_x = workgroup_x, local_y = workgroup_y;
    size_t global_x = ((m + local_x - 1) / local_x) * local_x;
    size_t global_y = ((n + local_y - 1) / local_y) * local_y;

    cl::NDRange local_size(local_x, local_y);
    cl::NDRange global_size(global_x, global_y);
    queue.enqueueNDRangeKernel(gol_kernel, cl::NullRange, global_size, local_size);

    queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, N_ELEMENTS * sizeof(int), bufferOutHost.data());
    queue.finish();

    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        grid[i][j] = bufferOutHost[i * m + j];
      }
    }

    std::swap(bufferIn, bufferOut);
    std::swap(bufferInHost, bufferOutHost);

    queue.enqueueWriteBuffer(bufferIn, CL_TRUE, 0, N_ELEMENTS * sizeof(int), bufferInHost.data());
    queue.finish();
  } catch (cl::Error err) {
    std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
  }
}

const std::vector<std::vector<int>> GameOfLifeOpenCL::get_grid() const { return grid; }
