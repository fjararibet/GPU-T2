#include "interface.hpp"

class GameOfLifeGPU : public GameOfLifeInterface{
private:
  std::vector<std::vector<bool>> grid;
public:
  GameOfLifeGPU(std::vector<std::vector<bool>> &grid);
  void tick();
  const std::vector<std::vector<bool>> get_grid() const;
};
