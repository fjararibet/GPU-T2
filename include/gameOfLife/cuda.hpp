#include "interface.hpp"

class GameOfLifeCuda : public GameOfLifeInterface {
private:
  std::vector<std::vector<int>> grid;
  std::vector<int> hostIn, hostOut;
  int *deviceIn;
  int *deviceOut;
  size_t n, m, N_ELEMENTS;
  int threadsPerBlock = 256;
  int blocksPerGrid;

public:
  GameOfLifeCuda(std::vector<std::vector<int>> &grid_);
  void tick();
  const std::vector<std::vector<int>> get_grid() const;
};
