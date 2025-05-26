#include "interface.hpp"

class GameOfLifeOpenCL : public GameOfLifeInterface{
private:
  std::vector<std::vector<int>> grid;
public:
  GameOfLifeOpenCL(std::vector<std::vector<int>> &grid);
  void tick();
  const std::vector<std::vector<int>> get_grid() const;
};
