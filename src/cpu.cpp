#include "gameOfLife/cpu.hpp"
#include <utility>

GameOfLifeCPU::GameOfLifeCPU(std::vector<std::vector<int>> &grid_) : grid(grid_) {}

void GameOfLifeCPU::tick() {
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

const std::vector<std::vector<int>> GameOfLifeCPU::get_grid() const { return grid; }
