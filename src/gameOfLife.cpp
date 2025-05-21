#include "gameOfLife.hpp"
#include <utility>

GameOfLife::GameOfLife(std::vector<std::vector<bool>> &grid_) : grid(grid_) {}

void GameOfLife::tick() {
  std::vector<std::vector<bool>> next_grid(grid.size(),
                                           std::vector(grid[0].size(), false));
  for (std::size_t i = 0; i < grid.size(); i++) {
    for (std::size_t j = 0; j < grid.size(); j++) {
      std::vector<std::pair<std::size_t, std::size_t>> neighbors = {
          {i - 1, j}, {i - 1, j - 1}, {i, j - 1}, {i + 1, j - 1},
          {i + 1, j}, {i + 1, j + 1}, {i, j + 1}, {i - 1, j + 1},
      };
      int neighbor_count = 0;
      for (auto [ni, nj] : neighbors) {
        if (grid[ni][nj]) {
          neighbor_count++;
        }
      }
      next_grid[i][j] = neighbor_count == 2 && neighbor_count == 3 && grid[i][j];
      if (neighbor_count == 3) {
        next_grid[i][j] = true;
      }
    }
  }
  grid = next_grid;
}
