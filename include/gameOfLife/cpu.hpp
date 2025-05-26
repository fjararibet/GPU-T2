#include "interface.hpp"

class GameOfLifeCPU : public GameOfLifeInterface{
private:
  std::vector<std::vector<int>> grid;
public:
  GameOfLifeCPU(std::vector<std::vector<int>> &grid);
  void tick();
  const std::vector<std::vector<int>> get_grid() const;
};
