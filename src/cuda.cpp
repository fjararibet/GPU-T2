#include "gameOfLife/cuda.hpp"
#include <utility>

GameOfLifeCuda::GameOfLifeCuda(std::vector<std::vector<int>> &grid_) : grid(grid_) {}

void GameOfLifeCuda::tick() {
}

const std::vector<std::vector<int>> GameOfLifeCuda::get_grid() const { return grid; }
