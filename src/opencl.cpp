#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "gameOfLife/opencl.hpp"
#include <CL/opencl.hpp>
#include <iostream>
#include <utility>

std::string kernel = R"(
kernel void gameOfLife(global int* In, global int* Out, int n, int m) {
    const int curr_col = get_global_id(0);
    const int curr_row = get_global_id(1);
    if (curr_row >= 0 && curr_row < n && curr_col >= 0 && curr_col < m) {
        int neighbor_count = 0;
        for(int row = curr_row - 1; row <= curr_row + 1; row++) {
          if (row < 0 || row >= n) continue;
          for(int col = curr_col - 1; col <= curr_col + 1; col++) {
            if (col < 0 || col >= m) continue;
            if (row == curr_row && col == curr_col) continue;
            neighbor_count += In[row * m + col];
          }
        }

        int new_cell = (In[curr_row * m + curr_col] && (neighbor_count == 2 || neighbor_count == 3)) || neighbor_count == 3;
        Out[curr_row * m + curr_col] = new_cell;
    }
}
)";
std::string kernel_local_memory = R"(
__kernel void gameOfLifeLocalMem(global int* In, global int* Out, const int n, const int m, local int* buf) {
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lsize_x = get_local_size(0);
    const int lsize_y = get_local_size(1);

    int local_index = (ly + 1) * (lsize_x + 2) + (lx + 1);
    if (gx < m && gy < n) {
        buf[local_index] = In[gy * m + gx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int neighbor_count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = lx + 1 + dx;
            int ny = ly + 1 + dy;
            neighbor_count += buf[ny * (lsize_x + 2) + nx];
        }
    }

    int curr = buf[local_index];
    int new_cell = (curr && (neighbor_count == 2 || neighbor_count == 3)) || neighbor_count == 3;

    if (gx < m && gy < n) {
        Out[gy * m + gx] = new_cell;
    }
}
)";

GameOfLifeOpenCL::GameOfLifeOpenCL(std::vector<std::vector<int>> &grid, int workgroup_x, int workgroup_y, bool local)
    : grid(grid), workgroup_x(workgroup_x), workgroup_y(workgroup_y), local(local) {
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
    cl::Program::Sources source;
    if (local) {
      source = {{kernel_local_memory.c_str(), kernel_local_memory.length()}};
    } else {
      source = {{kernel.c_str(), kernel.length()}};
    }
    cl::Program program(context, source);
    program.build(devices);

    if (local) {
      gol_kernel = cl::Kernel(program, "gameOfLifeLocalMem");
    } else {
      gol_kernel = cl::Kernel(program, "gameOfLife");
    }

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

    size_t local_x = workgroup_x, local_y = workgroup_y;
    size_t global_x = ((m + local_x - 1) / local_x) * local_x;
    size_t global_y = ((n + local_y - 1) / local_y) * local_y;

    size_t local_mem_size = (local_x + 2) * (local_y + 2) * sizeof(int);

    gol_kernel.setArg(0, bufferIn);
    gol_kernel.setArg(1, bufferOut);
    gol_kernel.setArg(2, (int)n);
    gol_kernel.setArg(3, (int)m);
    if (local) {
      gol_kernel.setArg(4, cl::Local(local_mem_size));
    }

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
    // std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
  }
}

const std::vector<std::vector<int>> GameOfLifeOpenCL::get_grid() const { return grid; }
