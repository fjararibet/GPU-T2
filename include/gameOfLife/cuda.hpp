#include "interface.hpp"

class GameOfLifeCuda : public GameOfLifeInterface{
private:
  std::vector<std::vector<int>> grid;
public:
  GameOfLifeCuda(std::vector<std::vector<int>> &grid);
  void tick();
  const std::vector<std::vector<int>> get_grid() const;
};
