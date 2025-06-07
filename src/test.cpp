#include "gameOfLife/cpu.hpp"
#include "gameOfLife/cuda.hpp"
#include "gameOfLife/interface.hpp"
#include "gameOfLife/opencl.hpp"
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

int N = 100;
int M = 100;

// cli args
bool cuda = false;
bool opencl = false;
bool cpu = true;
int workgroup_x = 16;
int workgroup_y = 16;
bool local = false;
int seconds = 10;
int main(int argc, char **argv) {
  for (int i = 0; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--workgroup-x") {
      std::string val = argv[i + 1];
      workgroup_x = std::stoi(val);
    }
    if (arg == "--workgroup-y") {
      std::string val = argv[i + 1];
      workgroup_y = std::stoi(val);
    }
    if (arg == "--local") {
      local = true;
    }
    if (arg == "--seconds") {
      std::string val = argv[i + 1];
      seconds = std::stoi(val);
    }
    if (arg == "--cuda") {
      cuda = true;
      cpu = false;
      opencl = false;
    }
    if (arg == "--opencl") {
      opencl = true;
      cuda = false;
      cpu = false;
    }
    if (arg == "--cpu") {
      cpu = true;
      opencl = false;
      cuda = false;
    }
  }
    std::vector<std::vector<int>> grid(N, std::vector<int>(M, 1));

    // random grid
    grid.resize(N);
    for (auto &v : grid) {
      v.resize(M);
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        grid[i][j] = rand() % 2 == 0;
      }
    }

    std::unique_ptr<GameOfLifeInterface> gol;
    if (cpu)
      gol = std::make_unique<GameOfLifeCPU>(grid);
    else if (cuda)
      gol = std::make_unique<GameOfLifeCuda>(grid, workgroup_x, workgroup_y, local);
    else if (opencl)
      gol = std::make_unique<GameOfLifeOpenCL>(grid, workgroup_x, workgroup_y, local);
    auto start = std::chrono::steady_clock::now();
    long long loop_count = 0;
    while (true) {
      loop_count++;
      gol->tick();
      auto now = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed = now - start;
      if (elapsed.count() >= seconds) {
        break;
      }
      long long total_cells = loop_count * N * M;
      if (total_cells < 0) {
        std::cout << "overflow" << std::endl;
      }
    }
    long long cells_per_sec = loop_count * N * M / seconds;
    std::cout << loop_count * N * M << " cells" << std::endl;
    std::cout << cells_per_sec << " cells/s" << std::endl;
  return 0;
}
